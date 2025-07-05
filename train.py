import os
import sys
import yaml
import torch
from utils.config_utils import validate_config
from runners.kfold import run_kfold
from runners.hyperparameter_grid_runner import grid_search_runner
from runners.train_core import train_from_config

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    if len(sys.argv) < 2:
        print("Usage: python train.py <config_path_or_search_yaml>")
        sys.exit(1)

    input_path = sys.argv[1]
    parent_folder = os.path.basename(os.path.dirname(input_path))

    if parent_folder == "grid_search":
        search_spec = load_yaml(input_path)
        config_template_path = search_spec["config"]
        search_space = search_spec["search"]
        grid_search_runner(config_template_path, search_space, k=5)
    else:
        config = load_yaml(input_path)
        train_from_config(config)

if __name__ == "__main__":
    main()
