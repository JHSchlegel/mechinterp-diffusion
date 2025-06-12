"""
Module for definining a CNN that acts as a probe for latent representations.
"""

# =========================================================================== #
#                           Packages and Presets                              #
# =========================================================================== #
import torch
import torch.nn as nn


# =========================================================================== #
#                             Probe Class Definition                          #
# =========================================================================== #
class LatentProbe(nn.Module):
    def __init__(
        self,
        n_latent_channels: int,
        n_hidden_channels: int = 64,
    ) -> None:
        """
        Initializes the LatentProbe with specified parameters.

        Args:
            n_latent_channels (int): Number of channels in the latent space.
            n_hidden_channels (int): Number of hidden channels in the probe.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_latent_channels, n_hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                n_hidden_channels, n_hidden_channels, 3, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(n_hidden_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the probe network.
        Args:
            x (torch.Tensor): Input tensor of shape
                (batch_size, latent_channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size).
        """
        return self.net(x).squeeze(-1)


if __name__ == "__main__":
    # Example
    probe = LatentProbe(n_latent_channels=128, n_hidden_channels=64)
    example_input = torch.randn(32, 128, 8, 8)  # [bs, c, h, w]
    output = probe(example_input)
    print(f"{output.shape=}")
