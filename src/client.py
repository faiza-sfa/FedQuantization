import copy
import importlib
from src.compressors.qsgd import QSGD
import timeit
from typing import Tuple
from flwr.common.parameter import ndarray_to_bytes
from flwr.common.typing import Parameters
import numpy as np

import torch
import torchvision
import flwr as fl
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, Weights
from .scripts.visualizer import summarize_loss

from src import federated_dataset

from logging import ERROR
from flwr.common.logger import log

from random import randrange
import random

import sys

from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main

class PytorchClient(fl.client.Client):

    def __init__(
        self,
        cid: str,
        model,
        dataset_name,
        train_test_split,
        plot_detailed_training,
        quantizer,
        min_no_samples
    ) -> None:
        self.cid = cid
        self.model = model
        self.dataset_name = dataset_name
        self.min_no_samples = min_no_samples
        self.train_test_split = train_test_split
        self.first_client = None
        self.plot_detailed_training = plot_detailed_training
        self.quantizer = quantizer
        if torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(int(cid)%torch.cuda.device_count()))
        else:
            self.device = torch.device(f"cpu")

    def get_parameters(self) -> ParametersRes:
        print(f"Client {self.cid}: get_parameters")

        weights: Weights = self.model.get_weights()
        parameters = fl.common.weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)

    def _train(
        self,
        model,
        trainloader: torch.utils.data.DataLoader,
        epochs: int,
        steps_per_epoch: int,
        epoch_global: int,
        learning_rate: float,
        client_name: str,
        mu: float
    ) -> None:
        if self.first_client is None:
            self.first_client = client_name            
        """Train the network."""
        # Define loss and optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")
        old_params = copy.deepcopy(list(model.parameters()))
        # Train the network
        old_params = [p.to(self.device) for p in old_params]
        model.to(self.device)
        model.train()
        start_time = timeit.default_timer()
        for epoch in range(epochs):  # loop over the dataset multiple times
            if steps_per_epoch > 0:
                n_steps = min(steps_per_epoch, len(trainloader))
            else:
                n_steps = len(trainloader)
            for i, data in enumerate(trainloader, 0):
                # if i == 0 and epoch == 0:
                    # print("client_name", client_name, 'data[0].reshape(-1)[0:10]', data[0].reshape(-1)[0:10])
                if i == n_steps:
                    raise Exception("This should not happen!")
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                loss = model.train_step(data=data, mu=mu, old_params=old_params)
                loss.backward()
                optimizer.step()
                # print statistics
                # if i == 0:
                    # print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, loss.item()))
                step = (epoch_global-1)*epochs*n_steps + epoch*n_steps + i
                # Log detailed training data for the first client encountered.
                if client_name == self.first_client and self.plot_detailed_training:
                    summarize_loss(f"training-detailed/{client_name}", loss, step)
            if epochs > 100:
                total_seconds = timeit.default_timer() - start_time
                seconds_per_round = total_seconds/(epoch+1)
                etc_minutes = (epochs - (epoch+1)) * (seconds_per_round/60)
                print(f"[TIME] ETC: {etc_minutes} minutes. Finished epoch {epoch+1}")
        model.to(torch.device("cpu"))

    def _test(
        self,
        model,
        trainloader: torch.utils.data.DataLoader,
        quantization_level: int,
        orig_weights: Weights,
        client_name: str = None,
    ) -> None:
        # start_weights = model.get_weights()[0].reshape(-1)[1:2]
        """Test the network."""
        if quantization_level is not None:
            assert False
            orig_quantization_level = self.quantizer.s
            self.quantizer.s = quantization_level
            weights_prime = self.model.get_weights()
            weights_diff = [w_prime - w for (w_prime, w) in zip(weights_prime, orig_weights)]
            flat_weights_diff = np.concatenate([w.reshape(-1) for w in weights_diff])
            q_flat_weights_diff = self.quantizer.decode(self.quantizer.encode(flat_weights_diff, cid=self.cid, client_name=client_name))
            q_weights = []
            for s, orig_weight in zip(model.get_weight_shapes(), orig_weights):
                length = np.prod(s)
                q_weight_diff = np.reshape(q_flat_weights_diff[:length], s)
                q_flat_weights_diff = q_flat_weights_diff[length:]
                q_weight = orig_weight + q_weight_diff
                q_weights.append(q_weight)
            
            self.model.set_weights(q_weights)

        model.to(self.device)
        model.eval()
        loss, _ = model.test(trainloader, reduction='sum')
        model.train()
        model.to(torch.device("cpu"))

        if quantization_level is not None:
            assert False
            self.model.set_weights(weights_prime)
            self.quantizer.s = orig_quantization_level
        # print(f"client {client_name} | test loss {loss} | weights[1:2] {start_weights}")
        return loss

    def fit(self, ins: FitIns) -> FitRes:
        config = ins.config

        seed = int(config["seed"])
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        Main.eval(f"using Random; Random.seed!({seed})")


        weights: Weights = fl.common.parameters_to_weights(ins.parameters)[:int(config["lo_quant_weight_idx"])]
        lo_quant_weights: Weights = fl.common.parameters_to_weights(ins.parameters)[int(config["lo_quant_weight_idx"]):]
        fit_begin = timeit.default_timer()

        # Get training config
        epochs = int(config["epochs"])
        steps_per_epoch = int(config["steps_per_epoch"])
        batch_size = int(config["batch_size"])
        epoch_global = int(config["epoch_global"])
        learning_rate = float(config["learning_rate"])
        client_name = config["client_name"]
        mu = float(config["mu"])
        if config["quantization_level"] != "":
            quantization_level = int(config["quantization_level"])
            self.quantizer.s = quantization_level

        # Set model parameters
        self.model.set_weights(weights)

        # Train model
        trainset = federated_dataset.load_data(
            client_names=[client_name],
            train_test_split=self.train_test_split,
            dataset_name=self.dataset_name,
            type="train",
            min_no_samples=self.min_no_samples,
            is_embedded=bool(int(config["is_embedded"])))
        testset = federated_dataset.load_data(
            client_names=[client_name],
            train_test_split=self.train_test_split,
            dataset_name=self.dataset_name,
            type="test",
            min_no_samples=self.min_no_samples,
            is_embedded=bool(int(config["is_embedded"])))
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True
        )
        # self.model.set_weights(lo_quant_weights)
        lo_quant_loss = 1
        # lo_quant_loss = self._test(self.model, trainloader, quantization_level=None, orig_weights=None, client_name=client_name)
        # self.model.set_weights(weights)
        uncompressed_losses = np.zeros(2)
        compressed_losses = np.zeros(2)
        uncompressed_losses[0] = self._test(self.model, trainloader, quantization_level=None, orig_weights=weights, client_name=client_name)
        self._train(self.model, trainloader, epochs=epochs, steps_per_epoch=steps_per_epoch, epoch_global = epoch_global, learning_rate=learning_rate, client_name=client_name, mu=mu)
        uncompressed_losses[1] = self._test(self.model, trainloader, quantization_level=None, orig_weights=weights, client_name=client_name)
        # if type(self.quantizer) == QSGD:
            # compressed_losses[0] = self._test(self.model, trainloader, quantization_level=self.quantizer.s, orig_weights=weights, client_name=client_name)
            # compressed_losses[1] = self._test(self.model, trainloader, quantization_level=self.quantizer.s*2, orig_weights=weights, client_name=client_name)

        # Return the refined weights and the number of examples used for training
        weights_prime: Weights = self.model.get_weights()
        weights_diff = [w_prime - w for (w_prime, w) in zip(weights_prime, weights)]
        flat_weights_diff = np.concatenate([w.reshape(-1) for w in weights_diff]).astype(np.float32)
        # if np.isnan(flat_weights_diff).any():
        try:
            params_prime = Parameters(
                tensors=[
                    self.quantizer.encode(flat_weights_diff,cid=self.cid,client_name=client_name),
                    ndarray_to_bytes(uncompressed_losses),
                    ndarray_to_bytes(compressed_losses),
                ] + [ndarray_to_bytes(np.array([lo_quant_loss]))],
                tensor_type="numpy.nda")
        except Exception as e:
            print(e)
            log(ERROR, str(e))
            res = FitRes(
                Parameters(tensors=[b''], tensor_type="numpy.nda"),
                num_examples=-42,  # Use -42 as magic value to indicate NaN.
                num_examples_ceil=-42,  # Use -42 as magic value to indicate NaN.
                fit_duration=-42,
            )
            return res
        num_examples_train = len(trainset)
        fit_duration = timeit.default_timer() - fit_begin
        res = FitRes(
            parameters=params_prime,
            num_examples=num_examples_train,
            num_examples_ceil=num_examples_train,
            fit_duration=fit_duration,
        )
        return res

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        raise Exception("All evaluation is now centralized on the server. Client \
        evaluation should not be happening")
