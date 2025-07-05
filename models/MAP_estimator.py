import torch
import torch.nn as nn
import numpy as np

class MAPEstimator(nn.Module):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, shift: int = 0, verbose: bool = True):
        """
        MAP estimator using single Gaussians per class.
        
        Args:
            X: [N, S] binary labels
            Y: [N, S] corrupted real-valued observations
            shift: int, use Y_{i+shift} to predict X_i
        """
        super().__init__()
        X = X.cpu().numpy().astype(np.int32)
        Y = Y.cpu().numpy().astype(np.float32)
        N, S = X.shape

        self.shift = shift
        self.S = S

        # Shift X to align with Y
        X_shifted = np.roll(X, shift=shift, axis=1)

        # Extract Y values for each class
        y_given_x0 = Y[X_shifted == 0]
        y_given_x1 = Y[X_shifted == 1]

        # Fit single Gaussians
        self.mu_0 = float(np.mean(y_given_x0))
        self.std_0 = float(np.std(y_given_x0) + 1e-8)

        self.mu_1 = float(np.mean(y_given_x1))
        self.std_1 = float(np.std(y_given_x1) + 1e-8)

        # Estimate priors
        p_0 = (X == 0).sum() / (X.size + 1e-8)
        p_1 = (X == 1).sum() / (X.size + 1e-8)
        self.log_p0 = np.log(p_0 + 1e-8)
        self.log_p1 = np.log(p_1 + 1e-8)

        if verbose:
            print(f"\nUnimodal Gaussian Parameters:")
            print(f"  X=0: μ = {self.mu_0:.4f}, σ = {self.std_0:.4f}")
            print(f"  X=1: μ = {self.mu_1:.4f}, σ = {self.std_1:.4f}")
            print(f"  log P(X=0): {self.log_p0:.4f}, log P(X=1): {self.log_p1:.4f}")

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Y: [B, S] corrupted input

        Returns:
            pred: [B, S] binary predictions
        """
        Y = Y.cpu().numpy()
        B, S = Y.shape

        # Align Y_{i+shift} with X_i
        Y_shifted = np.roll(Y, -self.shift, axis=1).reshape(-1)

        # Compute log-likelihoods under each Gaussian
        def log_gaussian(y, mu, std):
            return -0.5 * np.log(2 * np.pi * std**2) - ((y - mu)**2) / (2 * std**2)

        log_p_y_given_0 = log_gaussian(Y_shifted, self.mu_0, self.std_0).reshape(B, S)
        log_p_y_given_1 = log_gaussian(Y_shifted, self.mu_1, self.std_1).reshape(B, S)

        log_post_0 = log_p_y_given_0 + self.log_p0
        log_post_1 = log_p_y_given_1 + self.log_p1

        preds = (log_post_1 > log_post_0).astype(np.float32)
        return torch.tensor(preds)
