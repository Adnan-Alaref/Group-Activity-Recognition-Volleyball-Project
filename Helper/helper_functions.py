"""
A series of helper functions used throughout the project.

If a function gets defined once and could be used over and over, it'll go in here.
"""
import os
import torch
import random
from torch._prims_common import validate_exclusive_idx
import torchinfo
import numpy as np
from typing import List 
import matplotlib.pyplot as plt
from torchmetrics import ConfusionMatrix
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_confusion_matrix

#-----------------#
# Set All Seeds   #
#-----------------#
def set_all_seeds(seed:int=42):
  """
    Sets random seeds for reproducibility across Python, NumPy, and PyTorch.

    Ensures consistent results by seeding CPU and CUDA operations.
    CuDNN is configured for performance while maintaining acceptable
    reproducibility in most training scenarios.

    Args:
        seed (int): Random seed value to be set.
  """
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


#----------------------------#
# Define Accuracy Function   #
#----------------------------#
def accuracy_fn(y_true, y_pred):
  """Calculates accuracy between truth labels and predictions.

  Args:
      y_true (torch.Tensor): Truth labels for predictions.
      y_pred (torch.Tensor): Predictions to be compared to predictions.

  Returns:
      [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
  """
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct / len(y_pred)) * 100
  return acc


#-----------------------#
# Diplay Model Summary  #
#-----------------------#
def model_summary(model: torch.nn.Module, device:str):
  dummy_data = torch.zeros(1,3,224,224)
  return torchinfo.summary(model= model, input_data=(dummy_data), device = device)


#----------------------------------#
# Define Save Checkpoint Function  #
#----------------------------------#
def save_checkpoints(model: torch.nn.Module, scaler: torch.GradScaler|None,
                     optimizer: torch.optim.Optimizer, train_loss: float, val_loss: float, macro_f1:float, epoch: int, path: str)->None:
  # Define Chekpoint
  checkpoint = {
    "epochs": epoch,
    "macro_f1":float(macro_f1),
    "val_loss": float(val_loss),
    "train_loss": float(train_loss),
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict()
  }

  # add scaler ONLY if it exists
  if scaler is not None:
    checkpoint['scaler_state_dict'] = scaler.state_dict()

  # Create dir
  os.makedirs(os.path.dirname(path), exist_ok=True)
  torch.save(checkpoint, path)
  print(f"Saved checkpoint: {path}")


#----------------------------------#
# Define Load Checkpoint Function  #
#----------------------------------#
def load_checkpoint(path:str, device: str|None=None)->dict:
  if not os.path.isfile(path):
    raise FileNotFoundError(f"Checkpoint file does not found.\nPath: {path}")

  # Load Data
  checkpoint_data = torch.load(path, map_location=device)
  return checkpoint_data

#----------------------------------#
# Smooth Curves    #
#----------------------------------#
def smooth_curve(values, window=5):
    values = np.array(values)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window)/window, mode='valid')

#----------------------------------#
# Plot Loss and Accuracy Curves    #
#----------------------------------#
def Display_train_eval_curves(epochs:List[int], train_losses:List[float], val_losses:List[float], train_acc:List[float], val_acc:list[float]):
  """
    Plots training and testing loss curves over epochs.

    Args:
        epochs (list or array-like): Epoch numbers.
        train_losses (list or array-like): Training loss values per epoch.
        test_losses (list or array-like): Testing/validation loss values per epoch.

    Returns:
        None: Displays a matplotlib figure showing the loss curves.
  """
  plt.figure(figsize=(10,5) ,dpi=300)

  # -------- Loss Curves --------
  plt.subplot(1,2,1)
  plt.plot(epochs,train_losses,label = "Train Loss")
  plt.plot(epochs ,val_losses, label = "val Loss")
  plt.ylabel("Loss" ,fontsize = 12 ,weight='bold')
  plt.xlabel("Epochs" ,fontsize = 12 ,weight='bold')
  plt.title("Training and Validation Loss", fontsize=12, weight="bold")
  plt.legend(prop = {"size":12},loc = "best")
  plt.grid(True)

  # -------- Accuracy Curves --------
  plt.subplot(1,2,2)
  plt.plot(epochs,train_acc,label = "Train acc")
  plt.plot(epochs ,val_acc, label = "val acc")
  plt.ylabel("Accuracy" ,fontsize = 12 ,weight='bold')
  plt.xlabel("Epochs" ,fontsize = 12 ,weight='bold')
  plt.title("Training and Validation Accuracy", fontsize=12, weight="bold")
  plt.legend(prop = {"size":12},loc = "best")
  plt.grid(True)

  plt.tight_layout()
  # plt.axis("off")
  plt.show()

#---------------------------------------#
# Plot Loss and Accuracy Smooth Curves  #
#---------------------------------------#
def Display_train_eval_smooth_curves(epochs:List[int], train_losses:List[float], val_losses:List[float], train_acc:List[float], val_acc:list[float]):
  """
    Plots training and testing loss curves over epochs.

    Args:
        epochs (list or array-like): Epoch numbers.
        train_losses (list or array-like): Training loss values per epoch.
        test_losses (list or array-like): Testing/validation loss values per epoch.

    Returns:
        None: Displays a matplotlib figure showing the loss curves.
  """
  # Smoothing
  window = 5  # try 3–7 depending on noise
  train_losses = smooth_curve(train_losses, window)
  val_losses   = smooth_curve(val_losses, window)
  train_acc    = smooth_curve(train_acc, window)
  val_acc      = smooth_curve(val_acc, window)

  epochs = epochs[:len(train_losses)]

  plt.figure(figsize=(10,5) ,dpi=300)

  # -------- Loss Curves --------
  plt.subplot(1,2,1)
  plt.plot(epochs,train_losses,label = "Train Loss")
  plt.plot(epochs ,val_losses, label = "val Loss")
  plt.ylabel("Loss" ,fontsize = 12 ,weight='bold')
  plt.xlabel("Epochs" ,fontsize = 12 ,weight='bold')
  plt.title("Training and Validation Loss", fontsize=12, weight="bold")
  plt.legend(prop = {"size":12},loc = "best")
  plt.grid(True)

  # -------- Accuracy Curves --------
  plt.subplot(1,2,2)
  plt.plot(epochs,train_acc,label = "Train acc")
  plt.plot(epochs ,val_acc, label = "val acc")
  plt.ylabel("Accuracy" ,fontsize = 12 ,weight='bold')
  plt.xlabel("Epochs" ,fontsize = 12 ,weight='bold')
  plt.title("Training and Validation Accuracy", fontsize=12, weight="bold")
  plt.legend(prop = {"size":12},loc = "best")
  plt.grid(True)

  plt.tight_layout()
  # plt.axis("off")
  plt.show()


#-------------------------------#
# Display Classification Report #
#-------------------------------#
def Display_Classification_Report(y_true:torch.Tensor, y_pred:torch.Tensor, classes_names:List[str]):
  # Generate the classification report
  report = classification_report(y_true.cpu().numpy(), y_pred.cpu().numpy(), target_names=classes_names)
  print(f"Classification Report:\n{report}")


#----------------------------------------#
# Plot Confusion Matrix By Using Mlxtend #
#----------------------------------------#
def Display_Confusion_matrix(y_true: torch.Tensor, y_preds:torch.Tensor, classes_names:List[str]):
  """
    Plots a multiclass confusion matrix using torchmetrics and mlxtend.

    This function computes the confusion matrix from the predicted labels
    and true labels, and visualizes it using mlxtend's plot_confusion_matrix.

    Args:
        y_true (Tensor): Ground-truth labels (1D tensor).
        y_preds (Tensor): Predicted labels (1D tensor).
        classes_names (list): List of class names corresponding to label indices.

    Returns:
        None: Displays the confusion matrix plot using matplotlib.
  """
  confmat = ConfusionMatrix(task="multiclass", num_classes=len(classes_names))
  confmat_tensor = confmat(preds = y_preds, target=y_true)
  # 2. Plot the confusion matrix
  fig, ax = plot_confusion_matrix(conf_mat=confmat_tensor.cpu().numpy(), # matplotlib likes working with NumPy
                              class_names = classes_names,
                              figsize=(12,7))
  plt.tight_layout()
  plt.show()