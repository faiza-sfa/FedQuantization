import os

# Insert 'dissertation' directory so that we can resolve the src imports.
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, src_path)
from src import DATA_PATH, LEAF_PATH
from src.scripts.make_data_utils import make_data

def extract_sent140_data():
    sent140_in_dir = os.path.join(LEAF_PATH, 'data', 'sent140', 'data', 'all_data')
    sent140_out_dir = os.path.join(DATA_PATH, 'sent140')
    make_data(sent140_in_dir, sent140_out_dir)

if __name__ == "__main__":
    extract_sent140_data()