import os

from .. import EXPERIMENTS_PATH

def get_latest_experiment_dir(eid):
    experiment_base_dir = os.path.join(EXPERIMENTS_PATH, str(eid))
    experiment_dir = os.path.join(experiment_base_dir, max(os.listdir(experiment_base_dir)))
    return experiment_dir

def get_log_filename(eid):
    return os.path.join(get_latest_experiment_dir(eid), f"{eid}.log")

def prep_experiment_dir(eid):
    experiment_base_dir = os.path.join(EXPERIMENTS_PATH, str(eid))
    try:
        os.mkdir(experiment_base_dir)
    except FileExistsError as _:
        pass
    try:
        experiment_idx = max([int(max(n.lstrip("0"), "0")) for n in os.listdir(experiment_base_dir)]) + 1
    except ValueError as _:
        experiment_idx = 0
    experiment_dir = os.path.join(experiment_base_dir, str(experiment_idx).zfill(3))
    try:
        os.mkdir(experiment_dir)
    except FileExistsError as _:
        pass
    return experiment_dir