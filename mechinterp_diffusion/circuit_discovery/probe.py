"""
Module for definining a CNN that acts as a probe for latent representations.
"""

import sys
from pathlib import Path
from typing import List, Literal, Tuple

# =========================================================================== #
#                           Packages and Presets                              #
# =========================================================================== #
import torch
import torch.nn as nn
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from utils.reproducibility import set_all_seeds


# =========================================================================== #
#                             Probe Class Definition                          #
# =========================================================================== #
class LatentProbe(nn.Module):
    def __init__(
        self,
        n_latent_channels: int,
        n_hidden_channels: int = 64,
        probe_type: Literal["cnn", "linear"] = "cnn",
        spatial_resolution: Tuple[int, int] = (16, 16),
    ) -> None:
        """
        Initializes the LatentProbe with specified parameters.

        Args:
            n_latent_channels (int): Number of channels in the latent space.
            n_hidden_channels (int): Number of hidden channels in the probe.
        """
        super().__init__()
        self.spatial_resolution = spatial_resolution
        self.n_hidden_channels = n_hidden_channels
        self.probe_type = probe_type

        if probe_type == "cnn":
            self.net = nn.Sequential(
                nn.Conv2d(n_latent_channels, n_hidden_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(
                    n_hidden_channels,
                    n_hidden_channels,
                    3,
                    stride=2,
                    padding=1,
                ),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(n_hidden_channels, 1),
            )
        elif probe_type == "linear":
            n_spatial_points = spatial_resolution[0] * spatial_resolution[1]
            self.net = nn.Linear(n_spatial_points * n_latent_channels, 1)
        else:
            raise ValueError(
                f"Unknown probe type: {probe_type}. "
                "Choose either 'cnn' or 'linear'."
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


def train_probe(
    train_loader: torch.utils.data.DataLoader,
    probe: LatentProbe,
    lr: float = 1e-3,
    epochs: int = 10,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[LatentProbe, List[float]]:
    """
    Train the latent probe on a DataLoader.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader yielding
            (latents, labels) where latents have shape
            (batch_size, n_latent)
        probe (LatentProbe): Latent probe model to train.
        lr (float): Learning rate for optimization.
        epochs (int): Number of training epochs.
        seed (int): Random seed for reproducibility.
        device (str): Device to run the training on ("cuda" or "cpu").

    Returns:
        Tuple[LatentProbe, List[float]]: Trained probe and loss history.
    """
    set_all_seeds(seed)

    probe = probe.to(device)
    probe.train()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    losses = []

    for _ in tqdm(range(epochs)):
        for batch in train_loader:

            latents = batch["activations"]

            labels = batch["label"]

            bs, _, n_latent_channels = latents.shape

            if probe.probe_type == "cnn":
                latents = latents.reshape(
                    -1,
                    n_latent_channels,
                    probe.spatial_resolution[0],
                    probe.spatial_resolution[1],
                )  # [bs, c, h, w]
            elif probe.probe_type == "linear":
                latents = latents.reshape(bs, -1)  # [bs, c * h * w]
            latents = latents.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            logits = probe(latents)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    return probe, losses


@torch.no_grad()
def test_probe(
    probe: LatentProbe,
    test_loader: torch.utils.data.DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    """
    Test the probe on a DataLoader.

    Args:
        probe (LatentProbe): Trained latent probe.
        test_loader (torch.utils.data.DataLoader): DataLoader yielding
            (latents, labels).
        device (str): Device to run the testing on ("cuda" or "cpu").

    Returns:
        float: Accuracy of the probe on the test set.
    """
    probe.eval()
    correct = 0
    total = 0

    for batch in test_loader:
        latents = batch["activations"]
        labels = batch["label"]
        bs, _, n_latent_channels = latents.shape
        if probe.probe_type == "cnn":
            latents = latents.reshape(
                -1,
                n_latent_channels,
                probe.spatial_resolution[0],
                probe.spatial_resolution[1],
            )
        elif probe.probe_type == "linear":
            latents = latents.reshape(bs, -1)
        latents = latents.to(device)  # [bs, c, h, w]
        labels = labels.to(device).float()

        logits = probe(latents)
        preds = (logits > 0.0).long()

        correct += (preds == labels.long()).sum().item()
        total += labels.size(0)

    accuracy = correct / total

    return accuracy


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Train and test example
    # -------------------------------------------------------------------------
    from torch.utils.data import DataLoader

    # Create dummy data
    latents = torch.randn(
        100, 16 * 16, 1280
    )  # 100 samples, 1280 channels, 16x16
    labels = torch.randint(0, 2, (100,))  # Binary labels
    dataset = Dataset.from_dict(
        {"activations": latents.numpy(), "label": labels.numpy()}
    )
    dataset.set_format(
        type="torch",
        columns=["activations", "label"],
    )
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    probe = LatentProbe(
        n_latent_channels=1280,
        n_hidden_channels=64,
        probe_type="cnn",
        spatial_resolution=(16, 16),
    )

    trained_probe, loss_history = train_probe(
        train_loader,
        probe=probe,
        lr=1e-3,
        epochs=5,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"Trained probe: {trained_probe}")
    print(f"Loss history: {loss_history[:5]}...")  # Print first 5 losses

    accuracy = test_probe(
        trained_probe,
        test_loader,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"Test accuracy: {accuracy:.4f}")
