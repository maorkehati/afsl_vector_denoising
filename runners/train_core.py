import yaml
import torch
from utils.config_utils import validate_config
from runners.kfold import run_kfold
from runners.full_train_runner import run_full_train  # to be implemented

def train_from_config(config: dict):
    validate_config(config)

    # Set seed
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("\n🛠️  Loaded Configuration")
    print("──────────────────────────────────────")
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    print("──────────────────────────────────────\n")

    # Decide which runner to use
    runner_type = config.get("runner", "kfold")

    if runner_type == "full_train":
        return run_full_train(config=config, batch_size=128, seed=seed)
    elif runner_type == "kfold":
        return run_kfold(config=config, k=5, batch_size=128, seed=seed)
    else:
        raise ValueError(f"Unknown runner type: {runner_type}")
