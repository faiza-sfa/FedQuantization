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
class LSTMShakespeare(Model):
    def __init__(self, n_hidden, n_classes, cid) -> None:
        super().__init__(cid)
        self.n_hidden = n_hidden  # 256 in FedProx and LEAF
        self.n_classes = n_classes
        self.embedding_size = 8
        self.fc1 = nn.Linear(
            self.n_hidden,
            self.n_classes
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.n_hidden,
            num_layers=2,
            batch_first=True
        )
        self.embedding = nn.Embedding(self.n_classes, self.embedding_size)

    # pylint: disable-msg=arguments-differ,invalid-name
    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        # x: (num_batches, sequence_length)
        x = self.embedding(x)
        # x: (num_batches, sequence_length, embedding_size)
        _, (h, _) = self.lstm(x)
        # h:  (num_layers, num_batches, n_hidden)
        h = h[1,:,:]
        # h:  (num_batches, n_hidden)
        x = self.fc1(h)
        return x

def load_model(*args, **kwargs) -> nn.Module:
    return LSTMShakespeare(*args, **kwargs)
