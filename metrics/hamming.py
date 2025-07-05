import torch

def hamming_distance(preds, targets):
    """
    Computes the normalized Hamming distance between binary predictions and targets.

    Args:
        preds (Tensor): Predicted binary values, shape [B, S] or [N]
        targets (Tensor): Ground truth binary values, same shape as preds

    Returns:
        score (float): Normalized Hamming distance (lower is better)
        fmt (callable): Formatter function for printing
        agg (callable): Aggregation function over folds
        direction (str): "min" because lower Hamming distance is better
    """
    preds = preds.view(-1)
    targets = targets.view(-1)

    assert preds.shape == targets.shape, "Preds and targets must have the same shape"

    mismatches = (preds != targets).float().sum()
    total = preds.numel()
    score = (mismatches / total).item()

    fmt = lambda x: f"{x:.4f}"
    agg = lambda scores: sum(scores) / len(scores)

    return score, fmt, agg, "min"
