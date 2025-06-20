"""
Script for training a probe on latent representations.

Example usage:
    python train_probe.py --num_epochs 10 --batch_size 32 --learning_rate 0.001
"""

# =========================================================================== #
#                           Packages and Presets                              #
# =========================================================================== #

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from datasets import load_from_disk
from safetensors.torch import load_file, save_file
from simple_parsing import parse
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import ProbeConfig
from probe import LatentProbe, test_probe, train_probe


# =========================================================================== #
#                           Main Functionality                                #
# =========================================================================== #
def main() -> None:
    """
    Main function to train and evaluate a probe on latent representations.
    """
    config = parse(ProbeConfig)

    # Load and prepare training dataset
    train_dataset = load_from_disk(config.train_dataset_path).shuffle(
        seed=config.seed
    )
    train_dataset.set_format(
        type="torch",
        columns=["activations", "label"],
    )
    n_latent_channels = train_dataset.features["activations"].shape[1]
    train_dataset = train_dataset.select_columns(["activations", "label"])

    # Load and prepare test dataset
    test_dataset = load_from_disk(config.test_dataset_path)
    test_dataset.set_format(
        type="torch",
        columns=["activations", "label"],
    )
    test_dataset = test_dataset.select_columns(["activations", "label"])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create probe instance
    probe = LatentProbe(
        n_latent_channels=n_latent_channels,
        n_hidden_channels=config.n_hidden_channels,
        probe_type=config.probe_type,
        spatial_resolution=config.spatial_resolution,
    )

    # Train the probe
    trained_probe, loss_history = train_probe(
        train_loader=train_loader,
        probe=probe,
        lr=config.learning_rate,
        epochs=config.num_epochs,
        seed=config.seed,
        device=config.device,
    )

    train_accuracy = test_probe(
        trained_probe, train_loader, device=config.device
    )
    print(f"Training Accuracy: {train_accuracy:.4f}")

    # Evaluate the probe on test set
    test_accuracy = test_probe(
        trained_probe, test_loader, device=config.device
    )
    print(f"Test Accuracy: {test_accuracy:.4f}")

    save_file(
        trained_probe.state_dict(),
        config.save_path,
    )
    print(f"Model saved to: {config.save_path}")

    # Plot training loss
    plt.plot(loss_history)
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training Loss History")
    plt.grid()
    plt.show()

    # Test loading the model
    loaded_probe = LatentProbe(
        n_latent_channels=n_latent_channels,
        n_hidden_channels=config.n_hidden_channels,
        probe_type=config.probe_type,
        spatial_resolution=config.spatial_resolution,
    ).to(config.device)
    loaded_probe.load_state_dict(load_file(config.save_path))
    loaded_probe.eval()

    # Verify loaded model accuracy
    loaded_train_accuracy = test_probe(
        loaded_probe, train_loader, device=config.device
    )
    print(f"Loaded Model Training Accuracy: {loaded_train_accuracy:.4f}")

    loaded_test_accuracy = test_probe(
        loaded_probe, test_loader, device=config.device
    )

    print(f"Loaded Model Test Accuracy: {loaded_test_accuracy:.4f}")

    assert (
        abs(train_accuracy - loaded_train_accuracy) < 1e-4
    ), "Loaded model training accuracy does not match original model accuracy."

    assert (
        abs(test_accuracy - loaded_test_accuracy) < 1e-4
    ), "Loaded model testing accuracy does not match original model accuracy."


if __name__ == "__main__":
    main()
