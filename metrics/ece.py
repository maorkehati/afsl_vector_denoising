import torch

def expected_calibration_error(pred_probs, targets, n_bins=10):
    """
    Computes the Expected Calibration Error (ECE) for binary classification.
    
    Args:
        pred_probs (Tensor): [N] predicted probabilities (after sigmoid if logits were used).
        targets (Tensor): [N] binary ground truth labels.
        n_bins (int): Number of bins for calibration.

    Returns:
        score (float): ECE value.
        fmt (function): Formatter for display.
        agg (function): Aggregator across folds.
        direction (str): "min" since lower ECE is better.
    """
    pred_probs = pred_probs.flatten()
    targets = targets.flatten()

    bins = torch.linspace(0, 1, n_bins + 1, device=pred_probs.device)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]

    ece = torch.zeros(1, device=pred_probs.device)
    for lower, upper in zip(bin_lowers, bin_uppers):
        mask = (pred_probs > lower) & (pred_probs <= upper)
        if mask.any():
            bin_acc = targets[mask].float().mean()
            bin_conf = pred_probs[mask].mean()
            ece += (mask.float().mean()) * torch.abs(bin_acc - bin_conf)

    score = ece.item()
    fmt = lambda x: f"{x:.4f}"
    agg = lambda values: sum(values) / len(values)
    return score, fmt, agg, "min"
