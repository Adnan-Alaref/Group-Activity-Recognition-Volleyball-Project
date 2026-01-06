import torch
import torch.nn as nn
class ExtendedModel_Baseline1(nn.Module):
  """
    Wraps a backbone model with a simple classification head.

    The backbone is expected to output pooled feature maps
    (e.g., [B, C, 1, 1]), which are flattened and passed through
    a dropout + linear layer for classification.
  """
  def __init__(self, backbone: nn.Module, num_features:int, num_classes: int, prop_dropout:float=0.5) -> None:
    super().__init__()
    self.backbone = backbone
    self.conv_layer = nn.Sequential(
      nn.Conv2d(num_features, num_classes, kernel_size = 1),
      nn.AdaptiveAvgPool2d((1, 1))
    )
    # self.fc_layer = nn.Sequential(
    #   nn.Dropout(prop_dropout),
    #   nn.Linear(num_features, num_classes)
    # )

  def forward(self, x:torch.Tensor)-> torch.Tensor:
    x = self.backbone(x)    # [B, C, 1, 1] because adativeavgpool is included
    x = self.conv_layer(x)  # [B, C, 1, 1]
    x = torch.flatten(x, 1) # [B, C]
    return x