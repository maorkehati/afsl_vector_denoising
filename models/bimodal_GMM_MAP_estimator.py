import torch
import torch.nn as nn
import numpy as np
from sklearn.mixture import GaussianMixture

class Bimodal_GMM_MAP_Estimator(nn.Module):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, shift: int = 0, verbose: bool = True):
        """
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

        # Shift X so that we're trying to predict X_i from Y_{i+shift}
        X_shifted = np.roll(X, shift=shift, axis=1)

        # Extract Y values conditioned on shifted X
        y_given_x0 = Y[X_shifted == 0].reshape(-1, 1)
        y_given_x1 = Y[X_shifted == 1].reshape(-1, 1)

        # Fit 2-component GMMs
        self.gmm_0 = GaussianMixture(n_components=2, random_state=0).fit(y_given_x0)
        self.gmm_1 = GaussianMixture(n_components=2, random_state=0).fit(y_given_x1)

        # Estimate priors
        p_0 = (X == 0).sum() / (X.size + 1e-8)
        p_1 = (X == 1).sum() / (X.size + 1e-8)

        self.log_p0 = np.log(p_0 + 1e-8)
        self.log_p1 = np.log(p_1 + 1e-8)

        if verbose:
            self.print_gmm_params("X = 0", self.gmm_0)
            self.print_gmm_params("X = 1", self.gmm_1)
            print(f"📊 log P(X=0): {self.log_p0:.4f}, log P(X=1): {self.log_p1:.4f}")

    def print_gmm_params(self, label, gmm):
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        weights = gmm.weights_.flatten()

        print(f"\nGMM parameters for {label}:")
        for i, (w, mu, std) in enumerate(zip(weights, means, stds)):
            print(f"  Component {i+1}: weight = {w:.4f}, mean = {mu:.4f}, std = {std:.4f}")

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Y: [B, S] corrupted input

        Returns:
            pred: [B, S] binary prediction for each X_i
        """
        Y = Y.cpu().numpy()
        B, S = Y.shape

        # Roll Y to match X_i
        Y_shifted = np.roll(Y, -self.shift, axis=1).reshape(-1, 1)  # Flatten to [B*S, 1]

        # Compute log likelihoods
        log_p_y_given_0 = self.gmm_0.score_samples(Y_shifted).reshape(B, S)
        log_p_y_given_1 = self.gmm_1.score_samples(Y_shifted).reshape(B, S)

        # Compute log posteriors
        log_post_0 = log_p_y_given_0 + self.log_p0
        log_post_1 = log_p_y_given_1 + self.log_p1

        preds = (log_post_1 > log_post_0).astype(np.float32)
        return torch.tensor(preds)
