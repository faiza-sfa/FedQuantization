import os
import sys

# Insert 'dissertation' directory so that we can resolve the src imports.
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, src_path)
from src import DATA_PATH, FEDPROX_PATH
from src.scripts.make_data_utils import make_data

def extract_synthetic_data(data_type):
    synthetic_in_dir = os.path.join(FEDPROX_PATH, f"synthetic_FedProx_{data_type}", "data", "train")
    synthetic_out_dir = os.path.join(DATA_PATH, f'synthetic_FedProx_{data_type}')
    make_data(synthetic_in_dir, synthetic_out_dir)
    os.chdir(os.path.dirname(__file__))

if __name__ == "__main__":
    extract_synthetic_data(sys.argv[1])