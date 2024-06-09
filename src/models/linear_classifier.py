from collections import OrderedDict
from src.models.model import Model
from typing import List, Tuple
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
import flwr as fl

from .. import PROJECT_PATH, DATA_PATH

os.chdir(PROJECT_PATH)

# pylint: disable-msg=unsubscriptable-object
class MLR(Model):
    def __init__(self, n_classes, n_inputs, cid) -> None:
        super().__init__(cid)
        self.fc = nn.Linear(n_inputs, n_classes)

    # pylint: disable-msg=arguments-differ,invalid-name
    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        x = self.fc(x)
        return x

def load_model(*args, **kwargs) -> nn.Module:
    return MLR(*args, **kwargs)
