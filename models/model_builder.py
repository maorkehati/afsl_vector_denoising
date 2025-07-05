from models.MAP_estimator import MAPEstimator
from models.linear_regression import LinearRegression
from models.bimodal_GMM_MAP_estimator import Bimodal_GMM_MAP_Estimator
from models.local_multivariate_MAP_estimator import LocalMAPEstimatorFullCov
from models.conv1d import Conv1DNet

# Add all supported models here
MODEL_REGISTRY = {
    "MAP_Estimator": [MAPEstimator, "basic"],
    "Linear_Regression": [LinearRegression, "basic"],
    "Bimodal_GMM_MAP_Estimator": [Bimodal_GMM_MAP_Estimator, "basic"],
    "LocalMAPEstimatorFullCov": [LocalMAPEstimatorFullCov, "basic"],
    "Conv1DNet": [Conv1DNet, "torch"],
}

def build_model(model_config, X=None, Y=None):
    """
    Build and return a model from config.

    Args:
        config: Dictionary with keys:
            - type: string name of model class
            - params: dict of init kwargs
        X, Y: Optional tensors (for models like MAPEstimator that need training data)

    Returns:
        Instantiated model
    """
    model_type = model_config["type"]
    model_reg = MODEL_REGISTRY.get(model_type, None)
    if model_reg is None:
        raise ValueError(f"Unknown model type: {model_type}")
    model_cls = model_reg[0]
    model_type = model_reg[1]

    if model_cls is None:
        raise ValueError(f"Unknown model type: {model_type}")

    # If the model requires full dataset (e.g. MAPEstimator)
    if model_type == "basic":
        if X is None or Y is None:
            raise ValueError("Basic Model type requires full X and Y")
        return model_cls(X, Y, **model_config.get("params", {}))

    # Otherwise, pass standard kwargs
    return model_cls(**model_config.get("params", {}))
