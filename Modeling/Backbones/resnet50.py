import torch
import torch.nn as nn
from torchvision import models

class ResNet50(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    Weights = models.ResNet50_Weights.DEFAULT
    orignal_model = models.resnet50(weights = Weights)
    layers = list(orignal_model.children())[:-1]     # keep avgpool
    self.backbone = nn.Sequential(*layers)

  def forward(self, x:torch.Tensor)->torch.Tensor:
    x = self.backbone(x)
    return x