from collections import OrderedDict
from typing import List, Tuple
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
import flwr as fl

import numpy as np

from .. import PROJECT_PATH, DATA_PATH

os.chdir(PROJECT_PATH)

class Model(nn.Module):
    def __init__(self, cid):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(int(cid)%torch.cuda.device_count()))
        else:
            self.device = torch.device(f"cpu")

    def get_weights(self) -> fl.common.Weights:
        """Get model weights as a list of NumPy ndarrays."""
        return [np.copy(val.cpu().numpy()) for _, val in self.state_dict().items()]

    def set_weights(self, weights: fl.common.Weights) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.tensor(v, dtype=torch.int64) if "num_batches_tracked" in k else torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)

    def get_weight_shapes(self) -> List[torch.Size]:
        return [weight.size() for weight in self.state_dict().values()]

    # ONLY USE THIS FOR TRAINING! REGULARIZATION SHOULD NOT BE APPLIED FOR TESTING!
    def train_step(self, data, mu, old_params):
        # Assume that model, old_params is already set to the correct device by the client handler.
        images, labels = data[0].to(self.device), data[1].to(self.device)
        outputs = self(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        proximal_term = torch.tensor(0., requires_grad = True)
        for old_p, p in zip(old_params, self.parameters()):
            proximal_term = proximal_term + 0.5*torch.pow(old_p - p, 2).sum()
        loss += mu * proximal_term
        return loss

    
    def test(self, testloader: torch.utils.data.DataLoader, reduction='mean') -> Tuple[float, float]:
        """Validate the network on the entire test set."""
        if len(testloader) == 0:
            return -1, -1
        correct = 0
        loss = 0.0
        n_samples = 0
        self.to(self.device)
        self.eval()
        loss_fun = nn.CrossEntropyLoss(reduction=reduction)
        with torch.no_grad():
            for idx, data in enumerate(testloader):
                # if idx % 1000 == 0:
                # print(f"processed idx {idx} / {len(testloader)}")
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self(images)
                loss_now = loss_fun(outputs, labels).item()
                loss += loss_now
                _, predicted = torch.max(outputs.data, 1)  # pylint: disable-msg=no-member
                correct += (predicted == labels).sum().item()
                n_samples += len(predicted)
        self.to(torch.device(f"cpu"))
        accuracy = correct / n_samples
        return loss, accuracy