from .accuracy import accuracy
from .f1_score import f1_score
from .precision import precision
from .recall import recall
from .roc_auc import roc_auc
from .hamming import hamming_distance
from .bce import bce
from .brier import brier
from .ece import expected_calibration_error

METRIC_REGISTRY = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "roc_auc": roc_auc,
    'f1_score': f1_score,
    "hamming_distance": hamming_distance,
    "bce": bce,
    "brier": brier,
    "ece": expected_calibration_error,
}

def load_metric(name):
    if name not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: {name}")
    return METRIC_REGISTRY[name]
