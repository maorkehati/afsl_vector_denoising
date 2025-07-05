import torch.nn.functional as F

def bce(logits, targets):
    """
    Binary Cross-Entropy Loss (expects raw logits)

    Args:
        logits: Tensor of shape [B, S] (raw model outputs)
        targets: Tensor of shape [B, S] with binary labels (0 or 1)

    Returns:
        Tuple of (score, format_fn, aggregation_fn, direction)
    """
    targets = targets.float()
    score = F.binary_cross_entropy_with_logits(logits, targets).item()

    fmt = lambda x: f"{x:.6f}"
    agg = lambda scores: sum(scores) / len(scores)
    direction = "min"  # Lower BCE is better

    return score, fmt, agg, direction
