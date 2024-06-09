import itertools
from logging import INFO
import logging
import signal

import copy
from src.compressors.uveqfed import UVeQFed
from src.compressors.gzip import GZip
from src.compressors.qsgd import QSGD

from flwr.common.parameter import bytes_to_ndarray
from src.compressors.compressor import Compressor
import math
import numpy as np
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import flwr
from flwr.common import (
    EvaluateRes,
    EvaluateIns,
    FitIns,
    FitRes,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from .scripts.visualizer import add_bytes, add_bytes_downlink, summarize_bytes_sent_per_round, summarize_bytes_sent_per_round_downlink, summarize_bytes_sent_total, summarize_bytes_sent_total_downlink, summarize_loss, summarize_other, summarize_weight_update


class FedLearn(flwr.server.strategy.Strategy):
    def __init__(self,
        checkpoint_interval: int = 10,
        checkpoint_dir: str = "",
        n_clients_per_round_fit: int = 2,
        client_names: List[str] = [],
        min_available_clients: int = 2,
        eval_fn: Optional[Callable[[Weights], Optional[Tuple[float, float]]]] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, str]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, str]]] = None,
        accept_failures: bool = False,
        epochs_per_round: int = None,
        stragglers: float = None,
        drop_stragglers: bool = None,
        quantizer: Compressor = None,
        dynamic_quantization = None,
        weight_shapes: List[torch.Size] = None,
        compress_downlink: bool = None,
        client_n_samples: List[int] = None,
        localized_quantization: bool = False,
        min_quantization_level: int = None,
        max_quantization_level: int = None,
        quantization_budget_divider: float = 1,
        constant_variance: bool = None,
    ) -> None:
        super().__init__()
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir
        self.n_clients_per_round_fit = n_clients_per_round_fit
        self.min_available_clients = min_available_clients
        self.eval_fn = eval_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.client_names = client_names
        self.client_n_samples = client_n_samples
        self.localized_quantization = localized_quantization
        self.epochs_per_round = epochs_per_round
        self.n_stragglers = int(stragglers * n_clients_per_round_fit)
        self.drop_stragglers = drop_stragglers
        self.quantizer = quantizer
        self.dynamic_quantization = dynamic_quantization
        self.weight_shapes = weight_shapes
        self.losses = []
        self.mov_avg_losses = []
        self.last_quantization_increase = 0
        self.compress_downlink = compress_downlink
        self.quantization_budget_divider = quantization_budget_divider
        self.min_quantization_level = min_quantization_level
        self.max_quantization_level = max_quantization_level
        self.prev_weights = None
        self.prev_clients = None
        self.f0 = None
        self.fw = None
        self.lo_norm_quant_diffs = []
        self.constant_variance = constant_variance
        log(INFO, f"Sampling {self.n_stragglers} stragglers per round.")

    # Centralized
    def evaluate(self, weights: Weights, rnd: int) -> Optional[Tuple[float, float]]:
        """Evaluate model weights using an evaluation function (if
        provided)."""
        if self.eval_fn is None:
            # No evaluation function provided
            return None
        return self.eval_fn(weights, rnd)

    def on_configure_evaluate(
        self, rnd: int, weights: Weights, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        return []

    def on_configure_fit(
        self, rnd: int, weights: Weights, lo_quant_weights: Weights, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        if rnd % self.checkpoint_interval == 0:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{rnd}.pt")
            log(INFO, f"Saving checkpoint {checkpoint_path}")
            torch.save({'rnd': rnd, 'weights': weights}, checkpoint_path)

        # Sample clients
        all_fit_ins = []
        clients = list(client_manager.all().values())
        clients.sort(key=lambda x: x.cid)
        # np.random.seed(rnd)
        client_name_sample = np.random.choice(self.client_names, self.n_clients_per_round_fit, replace=False)
        straggler_epochs = random.randint(1, self.epochs_per_round)
        log(INFO, f"Straggler epochs: {straggler_epochs}")
        straggler_names = np.random.choice(client_name_sample, self.n_stragglers, replace=False)

        is_weight_update = self.compress_downlink and ((rnd-1) % 4 != 0)
        parameter_bytes = None
        if is_weight_update:
            quantizer = QSGD(s=1024, zero_rle=True)
            weights_diff = [w1 - w2 for w1, w2 in zip(weights, self.prev_weights)]
            flat_weights_diff = np.concatenate([w.reshape(-1) for w in weights_diff])
            enc_flat_weights_diff = quantizer.encode(flat_weights_diff)
            parameter_bytes = len(enc_flat_weights_diff)
            q_flat_weights_diff = quantizer.decode(enc_flat_weights_diff)
            weights_to_communicate = []
            for s, orig_weight in zip(self.weight_shapes, self.prev_weights):
                length = np.prod(s)
                q_weight_diff = np.reshape(q_flat_weights_diff[:length], s)
                q_flat_weights_diff = q_flat_weights_diff[length:]
                q_weight = orig_weight + q_weight_diff
                weights_to_communicate.append(q_weight)
            client_name_sample = self.prev_clients
        else:
            weights_to_communicate = weights
            parameter_bytes = 4 * len(client_name_sample) * sum([np.prod(w.shape) for w in weights_to_communicate])
        self.prev_weights = copy.deepcopy(weights_to_communicate)
        self.prev_clients = client_name_sample

        parameters = weights_to_parameters(weights_to_communicate)
        parameters.tensors += weights_to_parameters(lo_quant_weights).tensors

        add_bytes_downlink(parameter_bytes)
        summarize_bytes_sent_per_round_downlink("data sent downlink in MB" , parameter_bytes, rnd)
        summarize_bytes_sent_total_downlink("cumsum data sent downlink in MB" , rnd)

        if self.dynamic_quantization is not None:
            if self.dynamic_quantization["name"] == "AdaQuantFL":
                if self.f0 == None:
                    self.quantizer.s = self.dynamic_quantization["s0"]
                else:
                    self.quantizer.s = math.ceil(math.sqrt(self.f0 / self.fw) * self.dynamic_quantization["s0"])
            elif self.dynamic_quantization["name"] == "DynamicQuantizer":
                n_losses = len(self.mov_avg_losses)
                block_length = self.dynamic_quantization['block_length']
                if n_losses >= self.last_quantization_increase + block_length:
                    if self.mov_avg_losses[-1] - self.mov_avg_losses[-block_length] > 0:
                        self.quantizer.s = min(self.quantizer.s*2, self.dynamic_quantization['max_level'])
                        self.last_quantization_increase = n_losses
            summarize_other("training/quantization_levels", self.quantizer.s, rnd)
        on_configure_fit_args = {"quantization_level": dict(), "seed": dict()}
        # quantization_budget = round(len(client_name_sample) * self.quantizer.s / self.quantization_budget_divider)
        # log(INFO, f"quantization_budget = {quantization_budget} | quantizer.s = {self.quantizer.s}")
        client_name_sample = sorted(client_name_sample, key=lambda x: (self.client_n_samples[x], x), reverse=True)

        if type(self.quantizer) in {QSGD, GZip} and self.localized_quantization:
            clients_now_n_samples = {k: self.client_n_samples[k] for k in client_name_sample}
            norm_factor = sum(clients_now_n_samples.values())
            weightings = {k: v / norm_factor for k,v in clients_now_n_samples.items()}
            quantization_budget = len(client_name_sample) * self.quantizer.s
            variance_unweighted_strategy = sum([(w/self.quantizer.s)**2 for w in weightings.values()])
            variance_weighted_strategy_unscaled = sum([(w/(w**(2/3)))**2 for w in weightings.values()])
            if self.constant_variance:
                quantization_levels = {k: (w**(2/3)) * math.sqrt(variance_weighted_strategy_unscaled / variance_unweighted_strategy) for k, w in weightings.items()}
            else:
                quantization_levels = {k: (w**(2/3))/sum([v**(2/3) for v in weightings.values()]) * quantization_budget for k, w in weightings.items()}
            variance_weighted_strategy = sum([(weightings[k]/s)**2 for k,s in quantization_levels.items()])
            log(INFO, f"quantization budget ratio = {sum(quantization_levels.values())/quantization_budget} |"
                    f" variance_unweighted_strategy = {variance_unweighted_strategy} | variance_weighted_strategy = {variance_weighted_strategy}")

        for idx, (client, client_name) in enumerate(zip(itertools.cycle(clients), client_name_sample)):
            if client_name in straggler_names and self.drop_stragglers == True:
                continue
            quantization_level = None
            if type(self.quantizer) in {QSGD, GZip}:
                if self.localized_quantization:
                    # Old strategy
                    # total_samples = sum([self.client_n_samples[n] for n in client_name_sample[idx:]])
                    # proposed_q_level = round(quantization_budget * self.client_n_samples[client_name] / total_samples)
                    # quantization_level = min(self.max_quantization_level, max(self.min_quantization_level, proposed_q_level))
                    # quantization_budget -= quantization_level
                    # New strategy
                    quantization_level = max(1, round(quantization_levels[client_name]))
                else:
                    quantization_level = self.quantizer.s
                log(INFO, f"client {client_name} | cid {client.cid} | {self.client_n_samples[client_name]} samples | s = {quantization_level}")
            seed = str(random.randrange(1_000_000_000))
            on_configure_fit_args["quantization_level"][client_name] = quantization_level
            on_configure_fit_args["seed"][client_name] = seed
            config = {
                "client_name": client_name,
                "epochs": str(straggler_epochs if client_name in straggler_names else self.epochs_per_round),
                "quantization_level": "" if quantization_level is None else str(quantization_level),
                "lo_quant_weight_idx": str(len(weights_to_communicate)),
                "seed": seed
            }
            # print(f"client_name {client_name} | seed {seed}")
            if self.on_fit_config_fn is not None:
                # Custom fit config function provided
                config = {**config, **self.on_fit_config_fn(rnd)}
            fit_ins = FitIns(parameters, config)
            all_fit_ins.append((client, fit_ins, client_name))

        # Return client/config pairs
        return all_fit_ins, on_configure_fit_args

    def on_aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes, str]],
        failures: List[BaseException],
        on_configure_fit_args: Dict[str, Any],
    ) -> Optional[Weights]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None
        weights_results = []
        lo_quant_weights_results = []
        uncompressed_losses = np.zeros(2)
        compressed_losses = np.zeros(2)
        lo_quant_loss = 0
        total_examples = 0
        total_uveqfed_bytes = 0
        for (idx, (client, fit_res, client_name)) in enumerate(results):
            if fit_res.num_examples == -42 and fit_res.fit_duration == -42:
                log(INFO, "Received shutdown signal (-42). shutting down...")
                os.kill(os.getppid(), signal.SIGUSR2)
            assert len(fit_res.parameters.tensors) == 4
            import time
            start = time.process_time()
            if type(self.quantizer) == UVeQFed:
                seed = on_configure_fit_args["seed"][client_name]
                flat_weight_result, uveqfed_bytes = self.quantizer.decode(fit_res.parameters.tensors[0], seed=seed)
                total_uveqfed_bytes += uveqfed_bytes
                # flat_weight_result_lo_quant = self.quantizer.decode(fit_res.parameters.tensors[0], seed=seed)
            else:
                prev_s = self.quantizer.s
                self.quantizer.s = on_configure_fit_args["quantization_level"][client_name]
                flat_weight_result = self.quantizer.decode(fit_res.parameters.tensors[0])
                # flat_weight_result_lo_quant = self.quantizer.decode(fit_res.parameters.tensors[0], use_lo_quant=True)
            # print(f"client {client_name} | flat_weights_resultf[0:10]", flat_weight_result[0:10])
                self.quantizer.s = prev_s
            uncompressed_loss = bytes_to_ndarray(fit_res.parameters.tensors[1])
            uncompressed_losses += uncompressed_loss
            compressed_loss = bytes_to_ndarray(fit_res.parameters.tensors[2])
            compressed_losses += compressed_loss
            lo_quant_loss += bytes_to_ndarray(fit_res.parameters.tensors[3])[0]
            total_examples += fit_res.num_examples
            weight_result = []
            weight_result_lo_quant = []
            for s in self.weight_shapes:
                if len(s) == 0:
                    length = 1
                else:
                    length = np.prod(s)
                w, flat_weight_result = np.reshape(flat_weight_result[:length], s), flat_weight_result[length:]
                # w_lo_quant, flat_weight_result_lo_quant = np.reshape(flat_weight_result_lo_quant[:length], s), flat_weight_result_lo_quant[length:]
                weight_result.append(w)
                # weight_result_lo_quant.append(w_lo_quant)
            weights_results.append([weight_result, fit_res.num_examples])
            # lo_quant_weights_results.append([weight_result_lo_quant, fit_res.num_examples])
            assert len(flat_weight_result) == 0
        
        uncompressed_losses /= total_examples
        compressed_losses /= total_examples
        lo_quant_loss /= total_examples
        self.lo_norm_quant_diffs.append(uncompressed_losses[0] - lo_quant_loss)

        # print("SERVER TOTAL compressed_losses:", compressed_losses)
        print("SERVER TOTAL uncompressed_losses:", uncompressed_losses)
        summarize_loss("training/lo_quant_loss", lo_quant_loss, rnd)
        summarize_loss("training/lo_quant_loss_and_uncompressed/difference", self.lo_norm_quant_diffs[-1], rnd)
        summarize_loss("training/lo_quant_loss_and_uncompressed/difference_accumulated", sum(self.lo_norm_quant_diffs), rnd)
        summarize_loss("training/uncompressed/beginning", uncompressed_losses[0], rnd)
        summarize_loss("training/uncompressed/end", uncompressed_losses[1], rnd)
        summarize_loss("training/uncompressed/difference", uncompressed_losses[1]-uncompressed_losses[0], rnd)
        # summarize_loss("training/compressed/difference", compressed_losses[0]-uncompressed_losses[0], rnd)
        # summarize_loss("training/more_compressed/difference", compressed_losses[1]-uncompressed_losses[0], rnd)
        # summarize_loss("training/compressed_and_more_compressed/difference", compressed_losses[1]-compressed_losses[0], rnd)
        # summarize_loss("training/compressed_ratio_more_compressed/difference",
        # (uncompressed_losses[0]-compressed_losses[1])/(uncompressed_losses[0]-compressed_losses[0]),
        # rnd)

        if self.f0 == None:
            self.f0 = uncompressed_losses[0]
        self.fw = uncompressed_losses[0]
        self.losses.append(uncompressed_losses[0])
        if self.dynamic_quantization is not None and self.dynamic_quantization["name"] == "DynamicQuantizer":
            if len(self.losses) == 1:
                self.mov_avg_losses.append(self.losses[-1])
            else:
                self.mov_avg_losses.append(
                    self.dynamic_quantization['moving_average'] * self.mov_avg_losses[-1] +
                    (1-self.dynamic_quantization['moving_average']) * self.losses[-1]
                )
            summarize_loss("training/moving_average", self.mov_avg_losses[-1], rnd)

        import datetime
        if type(self.quantizer) == UVeQFed:
            bytes = total_uveqfed_bytes
        else:
            bytes = sum([len(fit_res.parameters.tensors[0]) for _,fit_res,_ in results])
        add_bytes(bytes)
        summarize_bytes_sent_per_round("data sent (acc. over all clients) in MB" , bytes, rnd)
        summarize_bytes_sent_total("cumsum data sent (acc. over all clients) in MB" , rnd)
        flat_weight_diffs = np.concatenate([
            w.reshape(-1) for weights,_ in weights_results for w in weights])
        summarize_weight_update(f"weight-updates", flat_weight_diffs, rnd)
        res = aggregate(weights_results)
        lo_quant_res = aggregate(weights_results)
        return res, lo_quant_res

    def on_aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
        raise Exception("It should be impossible to invoke on_aggregate_evaluate.")

    def on_conclude_round(
        self, rnd: int, loss: Optional[float], acc: Optional[float]
    ) -> bool:
        """Always continue training."""
        return True
