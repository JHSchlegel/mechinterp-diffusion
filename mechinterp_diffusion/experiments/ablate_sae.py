"""
Hydra-based ablation study framework for Sparse Autoencoders.

Example usage:
    # Single run
    python ablate_sae_hydra.py sae=topk sae.k=20 trainer.lr=0.001

    # Grid sweep
    python ablate_sae_hydra.py -m sae=topk sae.k=10,20,40 trainer.lr=1e-4,1e-3
"""

import datetime

# =========================================================================== #
#                             Packages and Presets                            #
# =========================================================================== #
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import hydra
import torch
from datasets import Dataset, load_from_disk
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    AblationConfig,
    TopKSAEConfig,
    TrainerConfig,
    TrainingConfig,
)
from core.sae.base_sae import BaseSAE
from core.sae.topk_sae import TopKSAE
from core.sae.trainer import SAETrainer
from core.utils.reproducibility import set_all_seeds


# =========================================================================== #
#                             Main Hydra Script                               #
# =========================================================================== #
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: AblationConfig) -> None:
    """Main efunction for Hydra-based ablation study."""
    logger.info(f"Running with config:\n{OmegaConf.to_yaml(cfg)}")

    cfg.trainer.wandb_project = "sae_ablation_study"

    results = train_and_evaluate(cfg)

    # Save results
    results_file = "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_file}")
    logger.info(f"Test L2 Loss: {results['l2_loss']:.6f}")


@torch.no_grad()
def evaluate_sae(
    sae: BaseSAE,
    test_dataset: Dataset,
    batch_size: int = 128,
    num_samples_per_timestep: int = 10000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_batch_topk: bool = False,
) -> Dict[str, Any]:
    """Evaluate a Sparse Autoencoder on a test dataset.

    Args:
        sae (BaseSAE): Trained Sparse Autoencoder.
        test_dataset (Dataset): Huggingface Dataset containing test data.
            Should have "activations" and "timestep" columns.
        batch_size (int, optional): Batch size to use for evaluation.
            Defaults to 128.
        num_samples (int, optional): . Defaults to 10000.
        device (str, optional): Device to use. Defaults to "cuda" whenever it
            is available, otherwise "cpu".
            Defaults to True.

    Returns:
        Dict[str, Any]: Dictionary containing evaluation metrics.
    """
    sae.eval()
    sae.cfg.use_batch_topk = False  # Disable batch top-k for evaluation

    # Unique timesteps:
    timesteps = sorted(
        list(set([ts.item() for ts in test_dataset["timestep"]]))
    )

    # Subset test dataset:
    total_samples = min(
        len(test_dataset), num_samples_per_timestep * len(timesteps)
    )
    test_dataset = test_dataset.select(range(total_samples))

    timestep_metrics = {}
    for timestep in tqdm(timesteps, desc="Evaluating timesteps"):
        # Get indices for this timestep
        timestep_indices = [
            i
            for i, ts in enumerate(test_dataset["timestep"])
            if ts.item() == timestep
        ]

        if not timestep_indices:
            logger.warning(
                f"No samples found for timestep {timestep}. Skipping."
            )
            continue

        assert len(timestep_indices) == num_samples_per_timestep, (
            f"Expected {num_samples_per_timestep} samples for timestep "
            f"{timestep}, but found {len(timestep_indices)}."
        )

        timestep_dataset = test_dataset.select(timestep_indices)
        timestep_dataset.set_format(type="torch", columns=["activations"])

        # Create separate test loaders for each timestep
        # This way batch top k reconstruction can be compared with topk; also
        # this is consistent with how the SAE will be used during inference
        test_loader = DataLoader(
            timestep_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        # ---------------------------------------------------------------------
        # Extract Batchwise Statistics
        # ---------------------------------------------------------------------
        stats = {"l2": []}
        act_sum = 0.0
        act_sq_sum = 0.0
        n_total_values = 0

        for batch in test_loader:
            acts = batch["activations"].to(device)

            output = sae(acts)

            # sae_input, info = sae.preprocess_input(acts)
            # reconstruct = sae.postprocess_output(
            #     sae.decode(torch.nn.functional.relu(sae.encode(sae_input))),
            #     info,
            # )

            # Total MSE for this batch
            batch_mse = (output["sae_out"].view_as(acts) - acts).pow(2).sum()

            stats["l2"].append(batch_mse.detach().cpu().item())

            # Accumulate for variance (across all activation values)
            act_sum += acts.sum().item()
            act_sq_sum += acts.pow(2).sum().item()
            n_total_values += acts.numel()

            del acts, output

        torch.cuda.empty_cache()
        gc.collect()
        # ---------------------------------------------------------------------
        # Aggregate Statistics for Timestep
        # ---------------------------------------------------------------------
        mean_act = act_sum / n_total_values
        var = act_sq_sum / n_total_values - mean_act**2
        l2_mean = sum(stats["l2"]) / n_total_values

        timestep_metrics[timestep] = {}
        timestep_metrics[timestep].update(
            l2_loss=l2_mean,
            l2_loss_normalized=l2_mean / var if var > 0 else l2_mean,
            variance_explained=max(0, 1 - l2_mean / var) if var > 0 else 0,
            activation_variance=var,
            activation_mean=mean_act,
            activation_sum=act_sum,
            activation_sq_sum=act_sq_sum,
            activation_count=n_total_values,
            total_samples=n_total_values,
        )

    # -------------------------------------------------------------------------
    # Aggregate across all timesteps
    # -------------------------------------------------------------------------
    total_n = sum(r["total_samples"] for r in timestep_metrics.values())

    # Compute overall variance
    total_act_sum = sum(r["activation_sum"] for r in timestep_metrics.values())
    total_act_sq_sum = sum(
        r["activation_sq_sum"] for r in timestep_metrics.values()
    )
    total_act_count = sum(
        r["activation_count"] for r in timestep_metrics.values()
    )

    overall_mean = total_act_sum / total_act_count
    overall_variance = total_act_sq_sum / total_act_count - overall_mean**2

    # Total MSE
    total_mse_sum = sum(
        r["l2_loss"] * r["total_samples"] for r in timestep_metrics.values()
    )

    overall = {}
    overall.update(
        l2_loss=total_mse_sum / total_n,
        l2_loss_normalized=(
            total_mse_sum / total_n / overall_variance
            if overall_variance > 0
            else total_mse_sum / total_n
        ),
        variance_explained=(
            max(0, 1 - total_mse_sum / total_n / overall_variance)
            if overall_variance > 0
            else 0
        ),
        total_samples=total_n,
    )

    return dict(overall=overall, per_timestep=timestep_metrics)


def train_and_evaluate(cfg: AblationConfig) -> Dict[str, Any]:
    """_summary_

    Args:
        cfg (AblationConfig): Ablation configuration

    Returns:
        Dict[str, Any]: Dictionary containing evaluation metrics.
    """
    cfg.trainer.seed = cfg.seed
    set_all_seeds(cfg.trainer.seed)

    from icecream import ic

    ic(cfg.trainer.seed)

    if cfg.sae_type == "topk":
        # Convert from Hydra's DictConfig
        sae_config = TopKSAEConfig(**cfg.sae)

        SAEModelClass = TopKSAE
    else:
        raise ValueError(f"Unknown SAE type: {cfg.sae_type}")
    # Convert from Hydra's DictConfig
    trainer_config = TrainerConfig(**cfg.trainer)

    config = TrainingConfig(sae=sae_config, trainer=trainer_config)

    # Load datasets
    logger.info(f"Loading train dataset from: {cfg.train_dataset}")
    train_dataset = load_from_disk(cfg.train_dataset)
    train_dataset.set_format(type="torch", columns=["activations", "timestep"])

    logger.info(f"Loading test dataset from: {cfg.test_dataset}")
    test_dataset = load_from_disk(cfg.test_dataset)
    test_dataset.set_format(type="torch", columns=["activations", "timestep"])

    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae_model = SAEModelClass(sae_config).to(device)

    trainer = SAETrainer(
        config=config,
        sae_model=sae_model,
        dataset=train_dataset,
    )

    logger.info("Starting training...")
    start_time = datetime.datetime.now()
    trainer.train()
    train_time = (datetime.datetime.now() - start_time).total_seconds()

    # Clean up trainer to free memory before evaluation
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("Evaluating on test set...")
    metrics = evaluate_sae(
        sae_model,
        test_dataset,
        batch_size=128,
        num_samples_per_timestep=cfg.num_samples_per_timestep,
        device=device,
        use_batch_topk=cfg.sae.use_batch_topk,
    )

    # Add training info to overall metrics
    metrics["overall"]["train_time"] = train_time
    metrics["overall"]["num_params"] = sum(
        p.numel() for p in sae_model.parameters()
    )
    metrics["overall"]["num_dead_features"] = (
        (sae_model.num_tokens_inactive >= cfg.sae.num_tokens_dead_threshold)
        .sum()
        .item()
    )

    metrics["overall"]["perc_dead_features"] = (
        (sae_model.num_tokens_inactive >= cfg.sae.num_tokens_dead_threshold)
        .float()
        .mean()
    ).item()

    # Add config info to top level for compatibility
    metrics.update(
        {
            "sae_type": cfg.sae_type,
            "seed": cfg.trainer.seed,
            "train_time": train_time,
            "l2_loss": metrics["overall"]["l2_loss"],
            "l2_loss_normalized": metrics["overall"]["l2_loss_normalized"],
            "variance_explained": metrics["overall"]["variance_explained"],
            **{
                f"sae.{k}": v
                for k, v in OmegaConf.to_container(cfg.sae).items()
            },
            **{
                f"trainer.{k}": v
                for k, v in OmegaConf.to_container(cfg.trainer).items()
            },
        }
    )

    # Clean up
    del sae_model
    gc.collect()
    torch.cuda.empty_cache()

    return metrics


if __name__ == "__main__":
    main()
