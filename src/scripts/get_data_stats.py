# Save experiment output in experiments/123_456/

import argparse
import os
import numpy as np

# Insert 'dissertation' directory so that we can resolve the src imports.
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, src_path)

from src import DATA_PATH, GOD_CLIENT_NAME
from src.federated_dataset import FederatedDataset

def main() -> None:
    parser = argparse.ArgumentParser(description="Stats")
    parser.add_argument('dataset', type=str)
    args = parser.parse_args()
    if args.dataset == "sent140":
        min_no_samples = 10
    else:
        min_no_samples = 0
    dataset = FederatedDataset(client_name=GOD_CLIENT_NAME,
                dataset_name=args.dataset,
                transform=None, test_train_split=1.0, type="train", min_no_samples=min_no_samples,
                is_embedded=False)

    n_devices = len(dataset.clients)
    n_samples_per_device = np.array([len(d) for _,d in FederatedDataset.clients.items()])
    n_samples = n_samples_per_device.sum()
    mean = n_samples_per_device.mean()
    min = n_samples_per_device.min()
    max = n_samples_per_device.max()
    stddev = n_samples_per_device.std()
    print("Dataset", args.dataset)
    print("Devices", n_devices)
    print("Samples", n_samples)
    print(f"Samples per device: Mean: {mean} Stddev: {stddev} Min: {min} Max: {max}")
    # import sys
    # import numpy
    # numpy.set_printoptions(threshold=sys.maxsize)
    # print(repr(np.sort(n_samples_per_device)))

if __name__ == "__main__":
    main()
