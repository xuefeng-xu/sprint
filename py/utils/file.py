import json
import numpy as np
from pathlib import Path


# https://stackoverflow.com/a/57915246/17175613
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def indices_path(dataset, n0, branch):
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    indices_dir = PROJECT_ROOT / f"dataset/{dataset}/indices"
    indices_dir.mkdir(parents=True, exist_ok=True)
    return indices_dir / f"n0_{n0}_branch_{branch}.json"


def save_indices(indices, dataset, n0, branch):
    indices_file = indices_path(dataset, n0, branch)
    indices_file.parent.mkdir(parents=True, exist_ok=True)

    with open(indices_file, "w") as f:
        json.dump(indices, f, cls=NpEncoder)


def load_indices(dataset, n0, branch):
    indices_file = indices_path(dataset, n0, branch)

    if not indices_file.exists():
        raise FileNotFoundError(f"Indices file not found: {indices_file}")

    with open(indices_file, "r") as f:
        indices = json.load(f)
    return indices
