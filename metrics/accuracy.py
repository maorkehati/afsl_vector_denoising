import torch
import numpy as np

def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Computes binary accuracy.

    Args:
        preds: Tensor of shape [B, D], float32 values of 0.0 or 1.0
        targets: Tensor of shape [B, D], same shape as preds

    Returns:
        accuracy: scalar float in [0, 1]
    """
    correct = (preds == targets).float()
    acc = correct.mean().item()
    return acc, lambda acc: f"{acc:.4f}", np.mean, "max"
