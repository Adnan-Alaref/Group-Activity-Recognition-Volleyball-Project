import torch
from typing import Tuple
from tqdm.auto import tqdm
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader, device:str)->Tuple[torch.Tensor, torch.Tensor]:
  
  """
    Runs model evaluation and generates predictions on a dataset.

    Performs a forward-only pass in inference mode, collects predicted
    class labels and ground-truth targets, and returns them as tensors
    on the CPU for further metric computation.

    Args:
        model (nn.Module): Trained model to be evaluated.
        dataloader (DataLoader): Iterable over evaluation batches.
        device (str): Device used for inference ('cuda' or 'cpu').

    Returns:
        Tuple[Tensor, Tensor]: Ground-truth labels and predicted labels,
        both concatenated into CPU tensors.
  """
  
  model.to(device)
  model.eval()

  y_true, y_preds = [], []
  model.eval()
  with torch.inference_mode():
      for X, y in tqdm(dataloader, desc="Making predictions"):
          X = X.to(device, non_blocking=True)
          y = y.to(device, non_blocking=True)

          logits = model(X)
          preds = logits.argmax(dim=1)

          y_preds.append(preds.cpu())
          y_true.append(y.cpu())

  y_preds_tensor = torch.cat(y_preds)
  y_true_tensor  = torch.cat(y_true)

  return y_true_tensor, y_preds_tensor