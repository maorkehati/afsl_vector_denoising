import pickle, zlib
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_basic_dataset(path, indices=None, **kwargs):
    """
    Loads the basic dataset and returns X, Y tensors, optionally indexed.

    Args:
        path (str): Path to the pickle file.
        indices (array-like, optional): Indices to subset the data.

    Returns:
        dict: {"X": tensor, "Y": tensor}
    """
    with open(path, 'rb') as f:
        data = pickle.loads(zlib.decompress(f.read()))

    data = torch.tensor(data, dtype=torch.float32)
    X = data[:, 0]
    Y = data[:, 1]

    if indices is not None:
        X = X[indices]
        Y = Y[indices]

    dataset = torch.utils.data.TensorDataset(X, Y)
    full_batch_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    return full_batch_loader
