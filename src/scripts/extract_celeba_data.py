import os

# Insert 'dissertation' directory so that we can resolve the src imports.
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, src_path)
from src import DATA_PATH, LEAF_PATH
from src.scripts.make_data_utils import make_data

def extract_celeba_data():
    celeba_in_dir = os.path.join(LEAF_PATH, 'data', 'celeba', 'data', 'all_data')
    celeba_out_dir = os.path.join(DATA_PATH, 'celeba')
    make_data(celeba_in_dir, celeba_out_dir)

if __name__ == "__main__":
    extract_celeba_data()