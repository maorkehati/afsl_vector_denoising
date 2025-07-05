import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearRegression(nn.Module):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, L: int = 3, verbose: bool = True):
        """
        Shared-weight linear regression using least squares to fit 2L+1 local coefficients.
        Predicts all positions using partial weights at boundaries.
        """
        super().__init__()
        self.L = L
        N, S = X.shape
        K = 2 * L + 1

        # Padding and unfolding
        Y_padded = F.pad(Y, (L, L), mode='reflect')               # [N, S+2L]
        Y_unfold = Y_padded.unfold(dimension=1, size=K, step=1)   # [N, S, K]

        # Mask out-of-bounds positions in window
        mask = torch.ones(K, dtype=Y.dtype, device=Y.device)
        masks = []
        for i in range(S):
            valid_mask = mask.clone()
            if i < L:
                valid_mask[:L - i] = 0
            if i > S - L - 1:
                valid_mask[K - (i - (S - L - 1)):] = 0
            masks.append(valid_mask)
        mask_tensor = torch.stack(masks, dim=0)  # [S, K]
        mask_tensor = mask_tensor.unsqueeze(0).repeat(N, 1, 1)  # [N, S, K]

        Y_unfold_masked = Y_unfold * mask_tensor

        A = Y_unfold_masked.reshape(-1, K)         # [N*S, K]
        x = X.reshape(-1, 1)                       # [N*S, 1]

        # Add bias column
        ones = torch.ones(A.size(0), 1, dtype=A.dtype, device=A.device)
        A_aug = torch.cat([A, ones], dim=1)        # [N*S, K+1]

        # Solve least squares: A_aug @ w = x
        solution = torch.linalg.lstsq(A_aug, x).solution.squeeze()  # [K+1]
        weights = solution[:-1]  # [K]
        bias = solution[-1:]     # [1]

        self.register_buffer("weights", weights)
        self.register_buffer("bias", bias)

        if verbose:
            print(f"📐 Learned shared linear weights (L={L}): {weights}")
            print(f"📏 Learned bias: {bias.item():.4f}")

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Predicts X from Y using learned shared linear weights.

        Args:
            Y (torch.Tensor): [B, S] corrupted input

        Returns:
            Tensor: [B, S] binary predictions for all positions
        """
        L = self.L
        K = 2 * L + 1
        B, S = Y.shape

        Y_padded = F.pad(Y, (L, L), mode='reflect')               # [B, S+2L]
        Y_unfold = Y_padded.unfold(dimension=1, size=K, step=1)   # [B, S, K]

        # Create masking tensor [S, K]
        mask = torch.ones(K, dtype=Y.dtype, device=Y.device)
        masks = []
        for i in range(S):
            valid_mask = mask.clone()
            if i < L:
                valid_mask[:L - i] = 0
            if i > S - L - 1:
                valid_mask[K - (i - (S - L - 1)):] = 0
            masks.append(valid_mask)
        mask_tensor = torch.stack(masks, dim=0)  # [S, K]
        mask_tensor = mask_tensor.unsqueeze(0).repeat(B, 1, 1)  # [B, S, K]

        Y_masked = Y_unfold * mask_tensor

        logits = (Y_masked @ self.weights) + self.bias  # [B, S]
        return (logits > 0).float()
