import torch
from typing import List
from dataclasses import dataclass, field, asdict

@dataclass
class ModelConfig:
  # name: TYPE = VALUE
  dropout: float = 0.5
  num_classes : int = 8
  device: str|None = None
  # mutable → must use field
  train_split:List[int] = field(default_factory=lambda:[
      1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31,
      32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54
      ])

  test_split:List[int] = field(default_factory=lambda:[
      4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47
      ])

  val_split:List[int] = field(default_factory=lambda:[
      0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51
  ])

  classes_names:List[str] = field(default_factory=lambda:[
      "r_set", "r_spike", "r-pass", "r_winpoint", "l_winpoint", "l-pass", "l-spike", "l_set"
  ])

  def __post_init__(self):
    max_index = 54
    if self.device == None:
      self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    # validate splits
    for name, split in {
      "train": self.train_split,
      "test": self.test_split,
      "val": self.val_split,
      }.items():

      # empty
      if len(split) == 0:
        raise ValueError(f"{name}_split cannot be empty.")

      # uniqueness
      if len(split) != len(set(split)):
        raise ValueError(f"{name}_split contains duplicate indices.")

      # range
      if any(idx < 0 or idx > max_index for idx in split):
        raise ValueError(f"{name}_split indices must be in range [0, {max_index}]")

    # data leakage check
    all_index = set(self.train_split) | set(self.val_split) | set(self.test_split)
    if len(all_index) != (len(self.train_split) + len(self.val_split) + len(self.test_split)):
      raise ValueError("Splits overlap, There are data leakage problem!")

@dataclass
class TrainingConfig:
  learning_rate_backbone: float = 3e-4
  learning_rate_classifer: float = 6e-4
  val_bsize:int = 32
  batch_size: int = 32
  epochs: int = 50
  checkpoint_path: str = r'/content/drive/MyDrive/Volleyball-Project/Modeling/BaseLine1/chekpoints'
  print_every: int = 1
  momentum:float = 0.9
  eta_min: float = 1e-6
  weight_decay: float = 1e-4
  betas:tuple = field(default_factory=lambda:(0.9, 0.999))