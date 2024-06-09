# Save experiment output in experiments/123_456/

import argparse
import os
import signal
import subprocess
import time
import timeit
import toml
import random

# Insert 'dissertation' directory so that we can resolve the src imports.
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, src_path)
from src.scripts.utils import prep_experiment_dir
from src import EXPERIMENTS_PATH, PROJECT_PATH

os.chdir(PROJECT_PATH)

def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment runner")
    parser.add_argument('eid', metavar='experiment', type=str,
            help='name of the experiment to execute')
    parser.add_argument('--use_random_seed',
            action='store_true',
            help='generate a random seed instead of using the random seed in the config file.')
    args = parser.parse_args()

    experiment_config_path = os.path.join(EXPERIMENTS_PATH, f"{args.eid}.toml")
    config = toml.load(experiment_config_path)

    if not args.use_random_seed:
        random.seed(config['plumbing']['seed'])

    prep_experiment_dir(args.eid)

    server_seed = str(random.randint(0, 1_000_000_000))
    client_seed = str(random.randint(0, 1_000_000_000))

    process = subprocess.Popen(["python", "-u", "-m", "src.run_server", args.eid, "--seed", server_seed])
    child_processes = []
    def on_server_ready(signum, frame):
        assert signum == signal.SIGUSR1
        for cid in range(config['FedProx']['n_processes']):
            print(f"Starting process {cid} out of {config['FedProx']['n_processes']} client processes.")
            child_processes.append(subprocess.Popen(['python','-u', "-m", "src.run_client", args.eid, "--cid", str(cid), "--seed", client_seed]))
        print(f"Finished starting all {config['FedProx']['n_processes']} client processes.")

    def on_server_shutdown(signum, frame):
        assert signum == signal.SIGUSR2
        for cp in child_processes:
            cp.terminate()
            cp.wait()
        process.terminate()
        process.wait()
        exit(0)

    signal.signal(signal.SIGUSR1, on_server_ready)
    signal.signal(signal.SIGUSR2, on_server_shutdown)
    try:
        for cp in child_processes:
            cp.wait()
        process.wait()
    except KeyboardInterrupt:
        print("keyboard interrupt, terminating all processes...")
        for cp in child_processes:
            cp.terminate()
            cp.wait()
        process.terminate()
        process.wait()
        print("termination complete. exiting")
        exit(1)

if __name__ == "__main__":
    main()
