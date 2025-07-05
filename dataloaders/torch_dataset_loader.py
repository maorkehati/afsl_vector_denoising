import pickle, zlib
import torch
from torch.utils.data import Dataset, DataLoader, Subset

class TorchDataset(Dataset):
    def __init__(self, path):
        with open(path, 'rb') as f:
            data = pickle.loads(zlib.decompress(f.read()))
        data = torch.tensor(data, dtype=torch.float32)
        self.X = data[:, 0]
        self.Y = data[:, 1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def build_dataloader(path, batch_size=64, shuffle=True, num_workers=0, indices=None):
    """
    Builds a torch DataLoader from the full dataset or a subset.

    Args:
        path (str): Path to dataset
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data
        num_workers (int): Number of workers for DataLoader
        indices (list[int] or Tensor, optional): Indices to select a subset

    Returns:
        DataLoader
    """
    full_dataset = TorchDataset(path)
    if indices is not None:
        dataset = Subset(full_dataset, indices)
    else:
        dataset = full_dataset

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
