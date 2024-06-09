import argparse
import importlib
import logging
import math
import os
from src.compressors.uveqfed import UVeQFed
from src.compressors.gzip import GZip
from src.compressors.fp8 import FP8
from src.compressors.qsgd import QSGD
from src.compressors.no_compression import NoCompression
import time
import timeit
import toml
import random

import torch
import torch.nn
import torchvision
import flwr as fl
import numpy as np
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, Weights

from . import DEFAULT_SERVER_ADDRESS, EXPERIMENTS_PATH
from .client import PytorchClient
from .scripts.utils import get_log_filename, get_latest_experiment_dir
from .scripts.visualizer import init_summary_writer

import grpc

def main() -> None:
    """Load data, create and start PytorchClient."""
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
        "--cid", type=str, required=True, help="Client CID (no default)"
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

    # Configure logger
    fl.common.logger.configure(f"client_{args.cid}", filename=get_log_filename(args.eid))

    # Initialize visualizer

    # Start client
    experiment_config_path = os.path.join(EXPERIMENTS_PATH, f"{args.eid}.toml")
    config = toml.load(experiment_config_path)
    module = importlib.import_module(f'.models.{config["model"]["model"]}', 'src')
    model = module.load_model(**config["model_args"], cid=args.cid)
    if config["plumbing"]["plot_detailed_training"]:
        init_summary_writer(os.path.join(get_latest_experiment_dir(args.eid), f"client {args.cid}"))
    if config["plumbing"]["checkpoint"]:
        checkpoint_path = os.path.join(os.path.join(EXPERIMENTS_PATH, str(args.eid)), config["plumbing"]["checkpoint"])
        checkpoint = torch.load(checkpoint_path)
        model.set_weights(checkpoint["weights"])
    try:
        quantizer_name = config["quantizer"]["name"]
    except KeyError:
        quantizer_name = "NoCompression"
    if quantizer_name == "NoCompression":
        quantizer = NoCompression()
    elif quantizer_name == "QSGD":
        s = config["quantizer"]["quantization_levels"]
        quantizer = QSGD(s=s,zero_rle=True
        )
    elif quantizer_name == "UVeQFed":
        s_fDesRate = config["quantizer"]["s_fDesRate"]
        quantizer = UVeQFed(s_fDesRate=s_fDesRate)
    elif quantizer_name == "GZip":
        s = config["quantizer"]["quantization_levels"]
        quantizer = GZip(s=s)
    elif quantizer_name == "DynamicQuantizer":
        quantizer = QSGD(s=config["quantizer"]["min_level"], zero_rle=True
        )
    elif quantizer_name == "AdaQuantFL":
        quantizer = QSGD(s=1,zero_rle=True
        )
    elif quantizer_name == "LFL":
        quantizer = QSGD(s=None,zero_rle=True, type="LFL"
        )
    elif quantizer_name == "FP8":
        implementation = config["quantizer"]["implementation"]
        if implementation == "search":
            quantizer = FP8()
        else:
            raise Exception(f"Implementation {implementation} not found.")
    client = PytorchClient(args.cid, model, dataset_name=config["dataset"]["dataset"],
        train_test_split=config["dataset"]["train_test_split"],
        plot_detailed_training=config["plumbing"]["plot_detailed_training"],
        quantizer=quantizer,
        min_no_samples=config["dataset"]["min_no_samples"])
    try:
        fl.client.start_client(args.server_address, client)
    except Exception as e:
        if type(e) == grpc._channel._MultiThreadedRendezvous:
            print("Terminated with grpc._channel._MultiThreadedRendezvous")
        else:
            raise e


if __name__ == "__main__":
    main()
