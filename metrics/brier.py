def brier(pred_probs, targets):
    """
    Brier Score for binary probabilistic predictions.

    Args:
        pred_probs: Tensor of shape [B, S] with probabilities in [0, 1]
        targets: Tensor of shape [B, S] with binary labels (0 or 1)

    Returns:
        Tuple of (score, format_fn, aggregation_fn)
    """
    score = ((pred_probs - targets.float()) ** 2).mean().item()
    fmt = lambda x: f"{x:.6f}"
    agg = lambda scores: sum(scores) / len(scores)
    return score, fmt, agg, "min"
