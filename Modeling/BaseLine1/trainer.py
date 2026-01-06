import os
import torch
import torch.nn as nn
from typing import Dict, Any
import torchvision.transforms as T
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from Data_utils.data_loader import Group_activity_Dataset

#------------------------------------------------#
# Prepare Dataloader For Train or Eval Functions #
#------------------------------------------------#
def prepare_dataloader(dataset_root:str, model_config:Any, train_config:Any, split_type:str, device:str)->DataLoader:
  """
    Create a PyTorch DataLoader for the Group Activity dataset.

    This function builds the appropriate dataset and DataLoader for
    training, validation, or testing, including correct data augmentation,
    batching, and worker configuration.

    Parameters
    ----------
    dataset_root : str
        Path to the root directory of the dataset. Must contain a 'videos/'
        subdirectory.

    model_config : Any
        Configuration object that defines dataset splits.
        Expected attributes:
            - train_split
            - val_split
            - test_split

    train_config : Any
        Training configuration object.
        Expected attributes:
            - batch_size     (used for training)
            - val_bsize      (used for validation and testing)

    split_type : str
        Dataset split to load. Must be one of:
            - 'train'
            - 'eval'
            - 'test'

    device : str
        Target device. Must be either:
            - 'cuda'
            - 'cpu'
            - 'mps'

    Returns
    -------
    torch.utils.data.DataLoader
        Configured DataLoader for the requested dataset split.

    Notes
    -----
    - Training uses stochastic data augmentation:
        Resize → RandomResizedCrop → Rotation → ColorJitter.
    - Validation and testing use deterministic preprocessing:
        Resize → CenterCrop.
    - ImageNet normalization is applied (suitable for pretrained CNN backbones).
    - Persistent workers are enabled when num_workers > 0.
    - pin_memory is enabled automatically when using CUDA.
  """

  # Validate inputs
  if not os.path.exists(dataset_root):
    raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
  
  if split_type not in ['train', 'eval', 'test']:
    raise ValueError(f"split_type must be 'train', 'eval', or 'test', got {split_type}")
  
  if device not in ['cuda', 'cpu', 'mps']:
    raise ValueError(f"device must be 'cuda' or 'cpu' or 'mps', got {device}")

  # Perpare Data Augmentation
  train_transform = T.Compose([
      T.Resize((256, 256)), # Resize shorter side to 256 
      T.RandomResizedCrop(224), # Randomly crop and resize to 224x224
      T.RandomRotation(degrees=3),
      T.ColorJitter(0.3, 0.3, 0.3, 0.1),
      T.ToTensor(),
      T.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
  ])

  val_transform = T.Compose([
      T.Resize((256, 256)),
      T.CenterCrop(224),
      T.ToTensor(),
      T.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
  ])

  # Path to the videos directory inside the dataset
  videos_root = os.path.join(dataset_root,'videos')

  # Number of worker processes used for data loading
  # > 0 enables parallel data loading
  num_workers = min(4, os.cpu_count()or 1)

  # Pin memory is only useful when training on CUDA
  # It speeds up CPU → GPU memory transfers
  pin_memory = device == "cuda"

  loader_kwargs = dict(
    num_workers = num_workers,
    pin_memory = pin_memory,
    persistent_workers=num_workers > 0,
    # persistent_workers Keeps worker processes alive between epochs
    # Must be False when num_workers == 0
  )

  # prefetch_factor controls how many batches each worker preloads
  # It is only valid when num_workers > 0
  if num_workers > 0:
      loader_kwargs["prefetch_factor"] = 2
  

  # Training dataset (uses training split indices)
  if split_type=='train':
    dataset = Group_activity_Dataset(videos_root, model_config.train_split, train_transform)
    batch_size=train_config.batch_size
    shuffle=True
  
  elif split_type=='eval':
    # Validation dataset (no data augmentation, no shuffling)
    dataset = Group_activity_Dataset(videos_root, model_config.val_split, val_transform)
    batch_size=train_config.val_bsize
    shuffle=False
  
  else:
    # Test dataset (used only for final evaluation)
    dataset = Group_activity_Dataset(videos_root, model_config.test_split, val_transform)
    batch_size=train_config.val_bsize
    shuffle=False
  
  return DataLoader(dataset=dataset,
                    batch_size=batch_size,
                    shuffle=shuffle, 
                    **loader_kwargs)

#----------------------------#
# Create Training Function   #
#----------------------------#
def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               accuracy_fn,
               criterion: torch.nn.Module,
               optimizer : torch.optim.Optimizer,
               scaler: torch.cuda.amp.GradScaler, device:str)->Dict[str, Any]:
  """
    Executes one full training epoch for a given model.

    This function performs the standard training loop including:
    data transfer to device, mixed-precision forward pass (AMP),
    loss computation, gradient scaling, backward propagation,
    optimizer update, and metric aggregation.

    Args:
        model (nn.Module): Model to be trained.
        dataloader (DataLoader): Iterable over training batches.
        accuracy_fn (callable): Function to compute batch accuracy.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer used for parameter updates.
        scaler (GradScaler): Gradient scaler for mixed-precision training.
        device (str): Target device ('cuda' or 'cpu').

    Returns:
        dict: Dictionary containing model name, average training loss,
              and average training accuracy for the epoch.
  """
  from torch.cuda.amp import autocast

  total_train_loss, total_train_acc = 0.0, 0.0
  model.to(device)
  model.train()
  
  for inputs, targets in dataloader:
    inputs = inputs.to(device, non_blocking = True).float()# <-- important
    targets = targets.to(device, non_blocking = True)

    optimizer.zero_grad(set_to_none=True) # faster & safer

    # Forward and loss in mixed precision
    with autocast():
      logits = model(inputs)
      loss = criterion(logits, targets)
      train_preds = torch.softmax(logits, dim=1).argmax(dim=1) # Go from logits -> pred labels

    # Backward with scaled loss
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    total_train_loss += loss.item()
    total_train_acc+= accuracy_fn(y_true=targets, y_pred=train_preds)

  # Average Loss and accuracy for all batches per epoch
  total_train_loss /=len(dataloader)
  total_train_acc /=len(dataloader)

  return {
    "Model_name":model.__class__.__name__,
    "Train_loss":total_train_loss,
    "Train_acc":total_train_acc
  }


#----------------------------#
# Create Evaluation Function #
#----------------------------#
def eval_step(model: nn.Module,
              dataloader: torch.utils.data.DataLoader,
              accuracy_fn, criterion: torch.nn.Module, device:str)->Dict[str, Any]:
  """Returns a dictionary containing the results of model predicting on data_loader.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        dataloader (torch.utils.data.DataLoader): The target dataset to predict on.
        criterion (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.
        device: which device we use it.

    Returns:
        (dict): Results of model making predictions on dataloader.
  """
  from torch.cuda.amp import autocast
    
  model.to(device)
  model.eval()
    
  all_targets, all_preds = [], []
  total_test_loss, total_test_acc = 0.0, 0.0

  with torch.inference_mode():
    for inputs, targets in dataloader:
      inputs = inputs.to(device, non_blocking = True).float()# <-- important
      targets = targets.to(device, non_blocking = True)

      # Mixed precision inference
      with autocast():
        test_logits = model(inputs)
        test_loss = criterion(test_logits, targets)
        test_preds = test_logits.argmax(dim=1) # Go from logits -> pred labels

      total_test_loss +=test_loss.item()
      total_test_acc +=accuracy_fn(y_true=targets, y_pred=test_preds)
        
      all_preds.append(test_preds.cpu())
      all_targets.append(targets.cpu())

  # Average Loss and accuracy for all batches per epoch
  total_test_loss /=len(dataloader)
  total_test_acc /=len(dataloader)

  y_preds = torch.cat(all_preds)
  y_true = torch.cat(all_targets)
  macro_f1 = f1_score(y_true, y_preds, average='macro')

  return {
    "Model_name":model.__class__.__name__,
    "Test_loss":total_test_loss,
    "Test_acc":total_test_acc,
    "Macro_f1": macro_f1
  }