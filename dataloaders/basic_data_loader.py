import pickle, zlib
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_basic_dataset(path, indices=None, is_test=False, **kwargs):
    """
    Loads the basic dataset and returns a DataLoader with full batch size.

    Args:
        path (str): Path to the pickle file.
        indices (array-like, optional): Indices to subset the data.
        is_test (bool): If True, load only input features (no labels).

    Returns:
        DataLoader: A DataLoader over the dataset (with or without labels).
    """
    with open(path, 'rb') as f:
        data = pickle.loads(zlib.decompress(f.read()))

    data = torch.tensor(data, dtype=torch.float32)

    if is_test:
        X = data[:, 0]
        if indices is not None:
            X = X[indices]
        dataset = TensorDataset(X)
    else:
        X = data[:, 0]
        Y = data[:, 1]
        if indices is not None:
            X = X[indices]
            Y = Y[indices]
        dataset = TensorDataset(X, Y)

    full_batch_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    return full_batch_loader
