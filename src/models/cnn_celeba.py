from collections import OrderedDict
from src.models.model import Model
from typing import List, Tuple
import os

import torch
import torch.nn as nn
from torch import Tensor
import flwr as fl

from .. import PROJECT_PATH

os.chdir(PROJECT_PATH)

# pylint: disable-msg=unsubscriptable-object
class CNNCeleba(Model):
    def __init__(self, n_classes, cid) -> None:
        super().__init__(cid)
        assert n_classes == 2
        # input size (N, 1, 84, 84)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.convs = nn.ModuleList(
            [nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=(3, 3),
            padding=(1, 1)  # "same" padding
        )] + [nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            padding=(1, 1)  # "same" padding
        ) for _ in range(3)])
        # input size (N, 32, 5, 5)
        self.fc = nn.Linear(5*5*32, n_classes)
        self.layernorms = nn.ModuleList([
            torch.nn.LayerNorm([32, 84, 84]),
            torch.nn.LayerNorm([32, 42, 42]),
            torch.nn.LayerNorm([32, 21, 21]),
            torch.nn.LayerNorm([32, 10, 10]),
        ])

    # pylint: disable-msg=arguments-differ,invalid-name
    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        for idx in range(4):
            x = self.convs[idx](x)
            x = self.layernorms[idx](x)
            x = self.pool(x)
            x = self.relu(x)
        x = torch.reshape(x, (-1, 5*5*32))
        x = self.fc(x)
        return x

def load_model(*args, **kwargs) -> nn.Module:
    return CNNCeleba(*args, **kwargs)
