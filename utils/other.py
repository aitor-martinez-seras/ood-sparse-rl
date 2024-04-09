import random
import collections
import json
from pathlib import Path

import numpy as  np
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synthesize(array):
    d = collections.OrderedDict()
    d["mean"] = np.mean(array)
    d["std"] = np.std(array)
    d["min"] = np.amin(array)
    d["max"] = np.amax(array)
    return d

## JSON
def read_json(path_to_file: Path):
    # for reading also binary mode is important
    with open(path_to_file, 'rb') as fp:
        an_object = json.load(fp)
        return an_object
    

def write_json(an_object, path_to_file: Path):
    print(f"Started writing object {type(an_object)} data into a json file")
    with open(path_to_file, "w") as fp:
        json.dump(an_object, fp)
        print(f"Done writing JSON data into {path_to_file} file")

## NPY
def read_npy(path_to_file: Path):
    # for reading also binary mode is important
    with open(path_to_file, 'rb') as fp:
        an_object = np.load(fp)
        return an_object
    

def write_npy(an_object, path_to_file: Path):
    print(f"Started writing object {type(an_object)} data into a .npy file")
    with open(path_to_file, "wb") as fp:
        np.save(fp, an_object)       
