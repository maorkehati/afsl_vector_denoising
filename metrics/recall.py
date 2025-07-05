from sklearn.metrics import recall_score
import numpy as np

def recall(preds, targets):
    preds = preds.cpu().numpy().astype(int)
    targets = targets.cpu().numpy().astype(int)
    score = recall_score(targets.flatten(), preds.flatten(), zero_division=0)
    fmt = lambda x: f"{x:.4f}"
    agg = lambda scores: sum(scores) / len(scores)
    return score, fmt, agg, "max"
