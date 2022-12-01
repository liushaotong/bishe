#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：bishe 
@File    ：nin_experiment_config.py
@IDE     ：PyCharm 
@Author  ：lst
@Date    ：2022/11/27 11:04 
'''
from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
import hashlib
from pathlib import Path
import time
from typing import Dict, List, NamedTuple, Optional, Tuple


class DatasetType(Enum):
    CIFAR10 = (1, (3, 32, 32), 10)
    SVHN = (2, (3, 32, 32), 10)

    def __init__(self, id: int, image_shape: Tuple[int, int, int], num_classes: int):
        self.D = image_shape
        self.K = num_classes


class OptimizerType(Enum):
  SGD = 1
  SGD_MOMENTUM = 2
  ADAM = 3


# Hyperparameters that uniquely determine the experiment
@dataclass(frozen=True)
class HParams:
  seed: int = 0
  use_cuda: bool = True
  # Model
  model_depth: int = 4  # 2
  model_width: int = 14  # 8
  base_width: int = 25
  # Dataset
  dataset_type: DatasetType = DatasetType.CIFAR10
  data_seed: Optional[int] = 42
  train_dataset_size: Optional[int] = None
  test_dataset_size: Optional[int] = None
  # Training
  batch_size: int = 32
  epochs: int = 300
  optimizer_type: OptimizerType = OptimizerType.SGD_MOMENTUM
  lr: float = 0.01
  # Cross-entropy stopping criterion
  ce_target: Optional[float] = 0.01
  ce_target_milestones: Optional[List[float]] = field(default_factory=lambda: [0.05, 0.025, 0.015])