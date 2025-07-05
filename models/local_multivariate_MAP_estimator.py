import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LocalMAPEstimatorFullCov(nn.Module):
    """
    MAP estimator using multivariate Gaussian distributions with full covariance.
    Learns class-conditional statistics over local windows of size 2L+1.
    """
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, L: int = 3, verbose: bool = True):
        """
        Args:
            X: [N, S] binary labels
            Y: [N, S] real-valued corrupted observations
            L: int, half-window size (total window = 2L+1)
            verbose: Whether to print statistics
        """
        super().__init__()
        self.L = L
        K = 2 * L + 1  # Window size
        N, S = X.shape

        # Extract local windows from Y
        Y_padded = F.pad(Y, (L, L), mode='reflect')  # [N, S + 2L]
        Y_local = Y_padded.unfold(dimension=1, size=K, step=1)  # [N, S, K]

        # Flatten into [N*S, K], and corresponding labels [N*S]
        Y_flat = Y_local.reshape(-1, K)  # [N*S, K]
        X_flat = X.reshape(-1)          # [N*S]

        # Split data by class
        Y0 = Y_flat[X_flat == 0]
        Y1 = Y_flat[X_flat == 1]

        # Compute means and covariances
        mu_0 = Y0.mean(dim=0)
        mu_1 = Y1.mean(dim=0)

        cov_0 = torch.from_numpy(np.cov(Y0.T.cpu().numpy(), bias=True)).float().to(Y.device)
        cov_1 = torch.from_numpy(np.cov(Y1.T.cpu().numpy(), bias=True)).float().to(Y.device)

        # Add small regularization
        eps = 1e-5
        cov_0 += torch.eye(K, device=Y.device) * eps
        cov_1 += torch.eye(K, device=Y.device) * eps

        # Priors
        count_0 = (X_flat == 0).sum()
        count_1 = (X_flat == 1).sum()
        total = count_0 + count_1 + 1e-8

        p0 = count_0.float() / total
        p1 = count_1.float() / total

        # Store buffers
        self.register_buffer("mu_0", mu_0)
        self.register_buffer("mu_1", mu_1)
        self.register_buffer("cov_0", cov_0)
        self.register_buffer("cov_1", cov_1)
        self.register_buffer("inv_cov_0", torch.linalg.inv(cov_0))
        self.register_buffer("inv_cov_1", torch.linalg.inv(cov_1))

        _, logdet_0 = torch.slogdet(cov_0)
        _, logdet_1 = torch.slogdet(cov_1)
        self.register_buffer("logdet_0", logdet_0)
        self.register_buffer("logdet_1", logdet_1)

        self.register_buffer("log_p0", torch.log(p0.clamp(min=1e-8)))
        self.register_buffer("log_p1", torch.log(p1.clamp(min=1e-8)))

        if verbose:
            print(f"📐 Learned multivariate Gaussian stats (window size {K}):")
            print(f"  log P(X=0): {self.log_p0.item():.4f}, log P(X=1): {self.log_p1.item():.4f}")
            print(f"  logdet_0: {logdet_0.item():.4f}, logdet_1: {logdet_1.item():.4f}")

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Y: [B, S] input corrupted sequence

        Returns:
            preds: [B, S] binary predictions
        """
        B, S = Y.shape
        L = self.L
        K = 2 * L + 1

        # Pad and extract local windows
        Y_padded = F.pad(Y, (L, L), mode='reflect')  # [B, S + 2L]
        Y_local = Y_padded.unfold(dimension=1, size=K, step=1)  # [B, S, K]

        def log_mv_gaussian(y, mu, inv_cov, logdet, log_prior):
            delta = y - mu  # [B, S, K]
            mahal = torch.einsum("bsk,kl,bsl->bs", delta, inv_cov, delta)  # Mahalanobis distance
            return -0.5 * mahal - 0.5 * logdet + log_prior

        logp0 = log_mv_gaussian(Y_local, self.mu_0, self.inv_cov_0, self.logdet_0, self.log_p0)
        logp1 = log_mv_gaussian(Y_local, self.mu_1, self.inv_cov_1, self.logdet_1, self.log_p1)

        return (logp1 > logp0).float()
