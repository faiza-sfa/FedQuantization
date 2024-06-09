import argparse
import importlib
import logging
import math
import os
import signal
from src.compressors.uveqfed import UVeQFed
from src.compressors.gzip import GZip
from src.compressors.fp8 import FP8
from src.compressors.qsgd import QSGD
from src.compressors.no_compression import NoCompression
from src import federated_dataset
from typing import Callable, Dict, Optional, Tuple
import random
import time

import toml
import torch
import numpy as np
import torchvision
import flwr as fl

from . import DEFAULT_SERVER_ADDRESS, EXPERIMENTS_PATH, GOD_CLIENT_NAME
from .scripts.utils import get_log_filename, get_latest_experiment_dir
from .fed_learn import FedLearn
from .scripts.visualizer import init_summary_writer, visualize_model, summarize_accuracy, summarize_loss

# pylint: enable=no-member
CONFIG = None

def main() -> None:
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument('eid', metavar='experiment', type=str,
            help='name of the experiment to execute')
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--seed",
        help=f"random seed.",
        required=True,
        type=int,
    )
    args = parser.parse_args()

    # Randomly seed pytorch to get random model weights.
    # Don't use torch.seed() because of
    # https://github.com/pytorch/pytorch/issues/33546
    torch.manual_seed(args.seed)
    # torch.seed()
    # Make rest of experiments deterministic (almost, see
    # https://pytorch.org/docs/stable/notes/randomness.html)
    random.seed(args.seed)
    np.random.seed(args.seed)

    experiment_config_path = os.path.join(EXPERIMENTS_PATH, f"{args.eid}.toml")
    global CONFIG
    CONFIG = toml.load(experiment_config_path)

    # Configure logger
    fl.common.logger.configure(f"server", filename=get_log_filename(args.eid))

    fl.common.logger.log(logging.INFO, f"Experiment dir: {get_latest_experiment_dir(args.eid)}")

    # Initialize visualizer
    init_summary_writer(os.path.join(get_latest_experiment_dir(args.eid), "server"))

    model_module = importlib.import_module(f'.models.{CONFIG["model"]["model"]}', 'src')
    model = model_module.load_model(**CONFIG["model_args"], cid=0)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model parameters: ", total_params)
    print("Trainable model parameters: ", trainable_params)
    # Load evaluation data: Select a random subset of clients.
    if CONFIG["FedProx"]["n_clients_per_round_eval"] == -1:
        eval_client_names = [GOD_CLIENT_NAME]
    else:
        raise Exception("Training on a subset of all clients is not supported anymore.")
        # eval_client_names = random.sample(client_names, CONFIG["FedProx"]["n_clients_per_round_eval"])
    trainset = federated_dataset.load_data(
        client_names=eval_client_names,
        train_test_split=CONFIG["dataset"]["train_test_split"],
        dataset_name=CONFIG["dataset"]["dataset"],
        type="train",
        min_no_samples=CONFIG["dataset"]["min_no_samples"],
        is_embedded=CONFIG["plumbing"].get("is_embedded", False)
    )
    testset = federated_dataset.load_data(
        client_names=eval_client_names,
        train_test_split=CONFIG["dataset"]["train_test_split"],
        dataset_name=CONFIG["dataset"]["dataset"],
        type="test",
        min_no_samples=CONFIG["dataset"]["min_no_samples"],
        is_embedded=CONFIG["plumbing"].get("is_embedded", False)
    )
    print("centralized trainset length: ", len(trainset))
    print("centralized testset length: ", len(testset))
    # Sort for deterministic client subset selection.
    if CONFIG["FedProx"]["n_clients"] == "unfederated":
        client_names = [GOD_CLIENT_NAME]
        client_n_samples = {GOD_CLIENT_NAME: sum([len(v) for v in federated_dataset.FederatedDataset.clients.values()])}
    elif CONFIG["FedProx"]["n_clients"] == -1:
        client_names = sorted(federated_dataset.FederatedDataset.clients.keys())
        client_n_samples = {k: len(v) for k, v in federated_dataset.FederatedDataset.clients.items()}
    else:
        raise Exception("We only support training on all clients, because evaluation needs to be done on the same clients!!")
    images, _ = next(iter(torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([testset, trainset]), batch_size=32, shuffle=False)))
    visualize_model(model, images)

    dynamic_quantization = None
    try:
        quantizer_name = CONFIG["quantizer"]["name"]
    except KeyError:
        quantizer_name = "NoCompression"
    if quantizer_name == "NoCompression":
        quantizer = NoCompression()
    elif quantizer_name == "QSGD":
        s = CONFIG["quantizer"]["quantization_levels"]
        quantizer = QSGD(s=s, zero_rle=True,
        )
    elif quantizer_name == "UVeQFed":
        s_fDesRate = CONFIG["quantizer"]["s_fDesRate"]
        quantizer = UVeQFed(s_fDesRate=s_fDesRate)
    elif quantizer_name == "GZip":
        s = CONFIG["quantizer"]["quantization_levels"]
        quantizer = GZip(s=s)
    elif quantizer_name == "DynamicQuantizer":
        quantizer = QSGD(s=CONFIG["quantizer"]["min_level"], zero_rle=True)
        dynamic_quantization = CONFIG["quantizer"]
    elif quantizer_name == "AdaQuantFL":
        quantizer = QSGD(s=1, zero_rle=True)
        dynamic_quantization = CONFIG["quantizer"]
    elif quantizer_name == "FP8":
        implementation = CONFIG["quantizer"]["implementation"]
        if implementation == "search":
            quantizer = FP8()
        else:
            raise Exception(f"Implementation {implementation} not found.")
    elif quantizer_name == "LFL":
        quantizer = QSGD(type="LFL", s=None, zero_rle=None)
    # Create client_manager, strategy, and server
    client_manager = fl.server.SimpleClientManager()
    strategy = FedLearn(
        n_clients_per_round_fit=CONFIG["FedProx"]["n_clients_per_round_fit"],
        min_available_clients=CONFIG["FedProx"]["n_processes"],
        eval_fn=get_eval_fn(
            testset=testset,
            trainset=trainset,
            plot_training_loss=CONFIG["plumbing"].get("plot_training_loss", False),
            batch_size=CONFIG["plumbing"]['evaluation_batch_size']),
        on_fit_config_fn=lambda rnd: {
            "epoch_global": str(rnd),
            "steps_per_epoch": str(CONFIG["FedProx"]["steps_per_epoch"]),
            "batch_size": str(CONFIG["FedProx"]["batch_size"]),
            "learning_rate": str(CONFIG["FedProx"]["learning_rate"]),
            "mu": str(CONFIG["FedProx"]["mu"]),
            "is_embedded": str(1 if CONFIG["plumbing"].get("is_embedded", False) else 0)
        },
        checkpoint_dir=get_latest_experiment_dir(args.eid),
        checkpoint_interval=CONFIG["plumbing"]["checkpoint_interval"],
        client_names=client_names,
        epochs_per_round=CONFIG["FedProx"]["epochs_per_round"],
        stragglers=CONFIG["FedProx"]["stragglers"],
        drop_stragglers=CONFIG["FedProx"]["drop_stragglers"],
        quantizer=quantizer,
        dynamic_quantization = dynamic_quantization,
        weight_shapes=model.get_weight_shapes(),
        compress_downlink=CONFIG["quantizer"].get("compress_downlink", False),
        client_n_samples=client_n_samples,
        localized_quantization=CONFIG["quantizer"].get("localized_quantization", False),
        quantization_budget_divider=CONFIG["quantizer"].get("quantization_budget_divider", 1),
        min_quantization_level=CONFIG["quantizer"].get("min_quantization_level", None),
        max_quantization_level=CONFIG["quantizer"].get("max_quantization_level", None),
        constant_variance=CONFIG["quantizer"].get("constant_variance", True)
    )
    # Round 0 is only used for an initial evaluation.
    starting_round = 1
    if CONFIG["plumbing"]["checkpoint"]:
        checkpoint_path = os.path.join(os.path.join(EXPERIMENTS_PATH, str(args.eid)), CONFIG["checkpoint"])
        checkpoint = torch.load(checkpoint_path)
        starting_round = checkpoint['rnd'] + 1
    server = fl.server.Server(
        client_manager=client_manager,
        strategy=strategy,
        starting_round=starting_round)

    os.kill(os.getppid(), signal.SIGUSR1) 
    # Run server
    fl.server.start_server(
        args.server_address, server, config={"num_rounds": CONFIG["FedProx"]["n_rounds"]},
    )

def get_eval_fn(
    testset: torch.utils.data.Dataset,
    trainset: torch.utils.data.Dataset,
    batch_size: int,
    plot_training_loss: bool
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights, rnd: int) -> Optional[Tuple[float, float]]:
        rounds_between_evaluations = CONFIG["plumbing"].get("rounds_between_evaluations", 1)
        if rnd % rounds_between_evaluations == 0:
            module = importlib.import_module(f'.models.{CONFIG["model"]["model"]}', 'src')
            model = module.load_model(**CONFIG["model_args"], cid=0)
            model.set_weights(weights)
            # model = torch.nn.DataParallel(model)
            print("starting evaluation")
            eval_sets = [(testset, "testing")]
            if plot_training_loss:
                eval_sets.append((trainset, "training"))
            for (dataset, label) in eval_sets:
                testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
                loss, accuracy = model.test(testloader, reduction='sum')
                loss /= len(dataset)
                summarize_loss(label, loss, rnd)
                summarize_accuracy(label, accuracy, rnd)
                # if rnd > 0 and label == "testing":
                    # add_test_accuracy(accuracy, rnd)
                    # summarize_accuracy_over_bytes_sent("testing accuracy over data sent in MB", rnd)
            print("finished evaluation")
        else:
            print("skipping evaluation")

    return evaluate


if __name__ == "__main__":
    main()
