import os

# Insert 'dissertation' directory so that we can resolve the src imports.
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, src_path)
from src import DATA_PATH, LEAF_PATH
from src.scripts.make_data_utils import make_data

def extract_shakespeare_data():
    shakespeare_in_dir = os.path.join(LEAF_PATH, 'data', 'shakespeare', 'data', 'all_data')
    shakespeare_out_dir = os.path.join(DATA_PATH, 'shakespeare')
    make_data(shakespeare_in_dir, shakespeare_out_dir, is_embedded=True)

if __name__ == "__main__":
    extract_shakespeare_data()