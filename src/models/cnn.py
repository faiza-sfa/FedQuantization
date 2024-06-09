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
class CNN(Model):
    def __init__(self, n_classes, n_inputs, cid) -> None:
        super().__init__(cid)
        assert n_inputs == 784
        assert n_classes == 62
        # input size (N, 1, 28, 28)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(5, 5),
            padding=(2, 2)  # "same" padding
        )
        self.relu1 = nn.ReLU()
        # input size (N, 32, 28, 28)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # input size (N, 32, 14, 14)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(5, 5),
            padding=(2, 2)  # "same" padding
        )
        # input size (N, 64, 14, 14)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # input size (N, 64, 7, 7)
        self.fc1 = nn.Linear(7*7*64, 2048)
        self.fc2 = nn.Linear(2048, n_classes)

    # pylint: disable-msg=arguments-differ,invalid-name
    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        x = torch.reshape(x, (-1, 1, 28, 28))
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = torch.reshape(x, (-1, 7*7*64))
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def load_model(*args, **kwargs) -> nn.Module:
    return CNN(*args, **kwargs)
