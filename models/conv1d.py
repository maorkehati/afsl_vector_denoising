import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1DNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        kernel_size: int = 7,
        depth: int = 3,
        width: int = 32,
        activation: str = 'relu',
        bias: bool = True
    ):
        """
        A configurable 1D convolutional network for denoising or local modeling.

        Args:
            input_channels (int): Number of input channels (typically 1).
            output_channels (int): Number of output channels (typically 1 for regression).
            kernel_size (int): Size of the convolutional kernel (should be odd to preserve symmetry).
            depth (int): Number of layers.
            width (int): Number of channels in hidden layers.
            activation (str): Activation function ('relu', 'tanh', 'gelu', etc.).
            bias (bool): Whether to include bias in convolutions.
        """
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size should be odd for symmetric padding"

        # Define activation function
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(0.2)
        }
        assert activation in activations, f"Unsupported activation: {activation}"
        act = activations[activation]

        layers = []

        # Input layer
        layers.append(nn.Conv1d(input_channels, width, kernel_size, padding=kernel_size // 2, bias=bias))
        layers.append(act)

        # Hidden layers
        for _ in range(depth - 2):
            layers.append(nn.Conv1d(width, width, kernel_size, padding=kernel_size // 2, bias=bias))
            layers.append(act)

        # Output layer
        layers.append(nn.Conv1d(width, output_channels, kernel_size, padding=kernel_size // 2, bias=bias))

        self.net = nn.Sequential(*layers)

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Y: [B, S] input signal (e.g., corrupted real-valued observations)

        Returns:
            Tensor: [B, S] predicted output (e.g., binary or continuous signal)
        """
        if Y.ndim != 2:
            raise ValueError("Expected input of shape [B, S]")

        Y = Y.unsqueeze(1)  # [B, 1, S]
        out = self.net(Y)   # [B, 1, S]
        return out.squeeze(1)  # [B, S]
