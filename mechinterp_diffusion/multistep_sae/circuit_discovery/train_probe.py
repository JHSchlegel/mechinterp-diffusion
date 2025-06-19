from dataclasses import dataclass

import torch
from datasets import load_from_disk
from safetensors.torch import load_file, save_file
from simple_parsing import Serializable, parse
from torch.utils.data import DataLoader


@dataclass
class ProbeConfig(Serializable):
    """Configuration for the probe training."""

    dataset_path: str = (
        "../../../data/activations/stable-diffusion-2-1/birds_vs_cats/train/subset_size-10000/25-inference-steps/timesteps-4/unet.down_blocks.2.attentions.0"  # noqa: E501
    )
    num_epochs: int = 20
    """Number of epochs to train the probe model."""

    batch_size: int = 128
    """Batch size for training the probe model."""

    learning_rate: float = 1e-3
    """Learning rate for the AdamW optimizer."""

    weight_decay: float = 1e-5
    """AdamW optimizer weight decay."""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """Device to use for training, either 'cuda' or 'cpu'."""

    save_path: str = "probe_model.safetensors"
    """Path to save the trained probe model."""


def main() -> None:
    config = parse(ProbeConfig)
    dataset = load_from_disk(config.dataset_path).shuffle(seed=42)
    dataset.set_format(
        type="torch",
        columns=["activations", "label"],
    )
    n_latent_channels = dataset.features["activations"].shape[1]
    dataset = dataset.select_columns(["activations", "label"])

    # train and validation split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = dataset.train_test_split(
        test_size=val_size, seed=42
    ).values()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    from probe import (
        LatentProbe,
        test_probe,
        train_probe,
    )

    trained_probe, loss_history = train_probe(
        train_loader,
        n_latent_channels=n_latent_channels,
        lr=config.learning_rate,
        epochs=config.num_epochs,
        n_hidden_channels=64,
        device=config.device,
    )
    accuracy = test_probe(trained_probe, val_loader, device=config.device)
    print(f"Validation Accuracy: {accuracy:.4f}")
    save_file(
        trained_probe.state_dict(),
        config.save_path,
    )

    # test whether the model can be loaded correctly
    loaded_probe = LatentProbe(
        n_latent_channels=n_latent_channels,
        n_hidden_channels=64,
    ).to(config.device)
    loaded_probe.load_state_dict(load_file(config.save_path))
    loaded_probe.eval()
    loaded_accuracy = test_probe(
        loaded_probe, val_loader, device=config.device
    )

    print(f"Loaded Model Validation Accuracy: {loaded_accuracy:.4f}")

    assert (
        abs(accuracy - loaded_accuracy) < 1e-4
    ), "Loaded model accuracy does not match original model accuracy."


if __name__ == "__main__":
    main()
