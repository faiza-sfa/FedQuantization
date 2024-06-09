import os
import torch

PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))
os.chdir(PROJECT_PATH)

DEFAULT_SERVER_ADDRESS = "[::]:8080"
EXPERIMENTS_PATH = os.path.abspath("experiments")
DATA_PATH = os.path.abspath("data")
LEAF_PATH = os.path.abspath("leaf")
FEDPROX_PATH = os.path.abspath("FedProx")
# Unique ID that represents a client with all the data.
GOD_CLIENT_NAME = "952630398097868223647162069900715440297608885786503411514402181337302872670061123373871861"
# pylint: disable=no-member