from .basic_data_loader import load_basic_dataset
from .torch_dataset_loader import build_dataloader as torch_loader

# Mapping from mode name to callable
DATALOADER_REGISTRY = {
    "basic": lambda path, **kwargs: load_basic_dataset(path, **kwargs),
    "torch": lambda path, **kwargs: torch_loader(path, **kwargs),
}

def build_dataloader(mode='basic', path=None, is_test=False, **kwargs):
    """
    Load dataset according to specified mode.

    Args:
        mode (str): Type of dataloader ("basic", "torch", etc.)
        path (str): Path to data
        is_test (bool): Whether this is a test dataset (only inputs, no labels)
        **kwargs: Extra arguments passed to the loader (e.g., indices, batch_size)

    Returns:
        A dataset or DataLoader, depending on mode
    """
    if mode not in DATALOADER_REGISTRY:
        raise ValueError(f"Unknown dataloader mode: {mode}")

    return DATALOADER_REGISTRY[mode](path=path, is_test=is_test, **kwargs)
