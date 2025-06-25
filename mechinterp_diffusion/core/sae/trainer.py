"""
This module implements the SAETrainer class for training Sparse Autoencoders.

Handles the training loop, data loading, optimization, logging, plotting, and
checkpointing for SAE models.
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #


import gc
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset
from geom_median.torch import compute_geometric_median
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from tqdm.auto import tqdm
from transformers import SchedulerType, get_scheduler

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
from config import TrainerConfig, TrainingConfig
from utils.activations_iterator import CustomActivationsIterator

from .base_sae import BaseSAE

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =========================================================================== #
#                                SAETrainer Class                             #
# =========================================================================== #
class SAETrainer:
    """Handles the training process for Sparse Autoencoders."""

    def __init__(
        self,
        config: TrainingConfig,
        sae_model: BaseSAE,
        dataset: Dataset,
        optimizer: Optional[Adam] = None,
        lr_scheduler: Optional[LRScheduler | SchedulerType] = None,
    ) -> None:
        """Initializes the SAETrainer.

        Args:
            config (TrainingConfig): Configuration object for the training run.
            sae_model (BaseSAE): The Sparse Autoencoder model instance.
            dataset (Dataset): The dataset containing activation vectors.
            optimizer (Optional[Adam]): The optimizer. If None, Adam is
                created.
            lr_scheduler (Optional[LRScheduler | SchedulerType]): Learning rate
                scheduler. If None, one is created based on config.
        """
        assert isinstance(
            config, TrainingConfig
        ), "config must be a TrainingConfig instance"
        assert isinstance(
            config.trainer, TrainerConfig
        ), "config.trainer must be a TrainerConfig instance"
        assert isinstance(
            dataset, Dataset
        ), "dataset must be a Dataset instance"
        assert (
            "activations" in dataset.column_names
        ), "Dataset must contain 'activations' column"
        assert (
            "timestep" in dataset.column_names
        ), "Dataset must contain 'timestep' column"

        self.config: TrainingConfig = config
        self.trainer_cfg: TrainerConfig = config.trainer
        self.sae_model: BaseSAE = sae_model.to(sae_model.device)
        self.dataset: Dataset = dataset
        self.optimizer: Adam
        self.lr_scheduler: LambdaLR
        self.global_step: int = 0
        self.total_tokens_processed: int = 0

        self.device: torch.device = self.sae_model.device
        self.dtype: torch.dtype = self.sae_model.dtype
        logger.info(
            f"Trainer initialized on device: {self.device}, "
            f"dtype: {self.dtype}"
        )

        # ---------------------------------------------------------------------
        # Dataloader for Specified Timesteps
        # ---------------------------------------------------------------------
        self.train_indices: Optional[List[int]] = None
        if self.trainer_cfg.target_timesteps:
            logger.info(
                f"Filtering dataset for timesteps: "
                f"{self.trainer_cfg.target_timesteps}"
            )
            target_set = set(self.trainer_cfg.target_timesteps)

            self.train_indices = [
                i
                for i, ts in enumerate(self.dataset["timestep"])
                if ts.item() in target_set
            ]

            if not self.train_indices:
                raise ValueError(
                    "No data found for the specified target_timesteps."
                )

            logger.info(f"Filtered dataset size: {len(self.train_indices)}")
            self.train_dataset = self.dataset.select(self.train_indices)
        else:
            logger.info("Training on all timesteps.")
            self.train_dataset = self.dataset

        self.train_dataset.shuffle(seed=self.trainer_cfg.seed)

        self.dataloader = CustomActivationsIterator(
            dataset=self.train_dataset,
            batch_size=self.trainer_cfg.effective_batch_size,
            total_tokens=self.trainer_cfg.num_tokens,
            buffer_size=self.trainer_cfg.buffer_size,
        )

        # ---------------------------------------------------------------------
        # Initialize b_dec and mse_scale from data
        # ---------------------------------------------------------------------
        # initialize b_dec using geometric median of first 8 batches:
        tmp_dataloader = CustomActivationsIterator(
            dataset=self.train_dataset,
            batch_size=4096,
            total_tokens=32768,
            buffer_size=100,
        )

        # Temporary sample to calculate initial b_dec and mse scale
        stats_acts_sample = (
            torch.cat(
                [
                    next(tmp_dataloader.iterate())["activations"].to(
                        self.device, dtype=self.dtype
                    )
                    for _ in range(8)
                ],
                dim=0,
            )
            .float()
            .cpu()
        )

        if config.sae.standardize_input:
            # standardize the activations
            stats_acts_sample = (
                stats_acts_sample - stats_acts_sample.mean(dim=0)
            ) / (stats_acts_sample.std(dim=0) + 1e-8)

        self.sae_model.mse_scale = (
            1
            / (
                (
                    stats_acts_sample.float().mean(dim=0)
                    - stats_acts_sample.float()
                )
                ** 2
            ).mean()
        ).item()

        self.sae_model.b_dec.data = (
            compute_geometric_median(stats_acts_sample).median.cuda().float()
        )
        del tmp_dataloader, stats_acts_sample

        logging.info(
            f"b_dec initialized to geometric median of first 32768 samples: "
            f"{self.sae_model.b_dec.data}"
        )
        logging.info(f"mse_scale initialized to: {self.sae_model.mse_scale}")

        # ---------------------------------------------------------------------
        # Optimizer and Learning Rate Scheduler
        # ---------------------------------------------------------------------
        self.total_training_steps: int = math.ceil(
            self.trainer_cfg.num_tokens / self.trainer_cfg.effective_batch_size
        )

        if optimizer is None:
            self.optimizer = Adam(
                self.sae_model.parameters(),
                lr=self.trainer_cfg.lr,
                betas=(
                    self.trainer_cfg.adam_beta1,
                    self.trainer_cfg.adam_beta2,
                ),
            )
            logger.info("Adam optimizer initialized.")
        else:
            self.optimizer = optimizer

        if lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                name=self.trainer_cfg.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.trainer_cfg.warmup_steps,
                num_training_steps=self.total_training_steps,
            )
            logger.info("Learning rate scheduler initialized.")
        else:
            self.lr_scheduler = lr_scheduler

        # ---------------------------------------------------------------------
        # Checkpointing and Logging
        # ---------------------------------------------------------------------
        self._init_wandb()

        self.checkpoint_path = Path(self.trainer_cfg.checkpoint_path) / str(
            self.trainer_cfg.wandb_run_name or "default_run"
        )
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoints will be saved to: {self.checkpoint_path}")

        with open(self.checkpoint_path / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=4)
        logger.info(f"Saved config to: {self.checkpoint_path / 'config.json'}")

    def _init_wandb(self) -> None:
        """Initializes Weights & Biases if configured."""
        if self.trainer_cfg.wandb_project:
            wandb.init(
                project=self.trainer_cfg.wandb_project,
                entity=self.trainer_cfg.wandb_entity,
                name=self.trainer_cfg.wandb_run_name,
                config=self.config.to_dict(),  # Log the combined config
                save_code=self.trainer_cfg.wandb_log_code,
                dir=(
                    str(self.trainer_cfg.wandb_dir)
                    if self.trainer_cfg.wandb_dir
                    else None
                ),
            )
            # Watch the model gradients and parameters
            wandb.watch(
                self.sae_model, log_freq=self.trainer_cfg.log_frequency
            )
            logger.info(
                f"Weights & Biases initialized for run: "
                f"{self.trainer_cfg.wandb_run_name}"
            )
        else:
            logger.info(
                "wandb_project not specified. Skipping wandb initialization."
            )

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """Performs a single training step.

        Args:
            batch (Dict[str, Tensor]): A batch of data from the dataloader,
                                       containing 'activations' and 'timestep'.

        Returns:
            Dict[str, float]: Dictionary containing loss values for logging.
        """
        # assert as had problems with batch['activations'] being a list in past
        assert isinstance(
            batch["activations"], Tensor
        ), "batch['activations'] must be a Tensor"

        self.sae_model.train()
        activations = batch["activations"].to(self.device, dtype=self.dtype)

        sae_output = self.sae_model(activations)
        # sae_output = self.sae_model(activations)
        loss = sae_output["loss"]

        self.optimizer.zero_grad()
        loss.backward()

        if self.trainer_cfg.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self.sae_model.parameters(), self.trainer_cfg.max_grad_norm
            )

        # Keep unit norm for decoder columns if specified
        if self.sae_model.cfg.normalize_decoder:
            self.sae_model.remove_gradient_parallel_to_decoder_directions_()

        self.optimizer.step()
        self.lr_scheduler.step()

        # add log dict items to wandb log data
        log_data = {
            k: v.item() if isinstance(v, Tensor) else v
            for k, v in sae_output.items()
            if k not in ["sae_out", "feature_acts"]
        }
        log_data["lr"] = self.optimizer.param_groups[0]["lr"]
        log_data["total_tokens"] = self.total_tokens_processed

        if "feature_acts" in sae_output:
            feature_acts = sae_output["feature_acts"]
            l0_per_sample = (feature_acts > 0.0).float().sum(dim=-1)
            log_data["l0_per_sample_std"] = l0_per_sample.std().item()
            # Clean up intermediate tensors
            del feature_acts, l0_per_sample

        self.global_step += 1
        self.total_tokens_processed += activations.shape[0]

        # cleanup after step
        del activations, sae_output, loss

        return log_data

    def _log_metrics(self, log_data: Dict[str, float], step: int) -> None:
        """Log metrics to Weights & Biases.

        Args:
            log_data (Dict[str, float]): Metrics to log.
            step (int): Current training step.
        """
        if self.trainer_cfg.wandb_project and wandb.run:
            wandb.log(log_data, step=step)

    def _plot_feature_activation_histogram(self, step: int) -> None:
        """Log plot of histogram of L0 norms per sample to checkpoints and
        wandb

        Args:
            step (int): Current training step.
        """
        # Define variables outside try block for cleanup
        activations, sae_output, feature_acts, l0_per_sample, df, plot = (
            None,
        ) * 6
        try:
            self.sae_model.eval()

            # get 50k samples for plotting from dataset
            tmp_loader = CustomActivationsIterator(
                dataset=self.train_dataset,
                batch_size=self.trainer_cfg.effective_batch_size,
                total_tokens=50_000,
                buffer_size=self.trainer_cfg.buffer_size,
            )

            with torch.no_grad():
                sae_output = torch.cat(
                    [
                        self.sae_model(
                            batch["activations"].to(
                                self.device, dtype=self.dtype
                            )
                        )["feature_acts"]
                        for batch in tmp_loader.iterate()
                    ]
                )

                feature_acts = sae_output.detach().cpu()
                l0_per_sample = (feature_acts > 0.0).float().sum(dim=-1)
                df = pd.DataFrame({"l0_norm": l0_per_sample.numpy()})

                plot_dir = self.checkpoint_path / "plots"
                plot_dir.mkdir(parents=True, exist_ok=True)

                plot_filename = plot_dir / f"l0_histogram_step_{step}.png"

                median = np.median(df["l0_norm"])
                mean = np.mean(df["l0_norm"])
                std = np.std(df["l0_norm"])
                num_samples = len(df)

                fig, ax = plt.subplots(figsize=(10, 6))

                sns.histplot(
                    data=df,
                    x="l0_norm",
                    bins=100,
                    kde=False,
                    color="#00B5E2",
                    edgecolor="white",
                    linewidth=1,
                    alpha=0.8,
                    ax=ax,
                )

                # -------------------------------------------------------------
                # Mean/Median Lines
                # -------------------------------------------------------------
                ax.axvline(
                    median,
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"Median: {median:.2f}",
                )
                ax.axvline(
                    mean,
                    color="green",
                    linestyle="-.",
                    linewidth=1.5,
                    label=f"Mean: {mean:.2f}",
                )

                ax.set_title(
                    f"Distribution of Active Features per Sample (L0 Norm) at Step {step}",  # noqa: E501
                    fontsize=16,
                    fontweight="bold",
                )
                ax.set_xlabel(
                    "Number of Active Features (L0 Norm)", fontsize=14
                )
                ax.set_ylabel("Frequency", fontsize=14)

                # -------------------------------------------------------------
                # Statistics Text Box
                # -------------------------------------------------------------
                stats_text = (
                    f"Mean: {mean:.2f}\n"
                    f"Median: {median:.2f}\n"
                    f"Std Dev: {std:.2f}\n"
                    f"Samples: {num_samples}"
                )
                ax.text(
                    0.95,  # x
                    0.95,  # y
                    stats_text,
                    transform=ax.transAxes,
                    horizontalalignment="right",
                    verticalalignment="top",
                    fontsize=10,
                    bbox=dict(
                        boxstyle="round,pad=0.5", facecolor="white", alpha=0.7
                    ),
                )

                ax.legend()
                ax.grid(axis="y", alpha=0.4, linestyle="--")
                plt.tight_layout()

                plt.savefig(
                    plot_filename,
                    dpi=150,
                    bbox_inches="tight",
                )
                logger.info(
                    f"L0 activation histogram saved to {plot_filename}"
                )

                # if self.trainer_cfg.wandb_project and wandb.run:
                #     wandb.log(
                #         {
                #             "plots/l0_activation_histogram": wandb.Image(
                #                 str(plot_filename)
                #             )
                #         },
                #         step=step,
                #     )
                logger.info(
                    f"L0 activation histogram saved to {plot_filename} and "
                    f"logged to wandb."
                )
        except Exception as e:
            logger.error(f"Error during plotting: {e}", exc_info=True)

        finally:
            # -----------------------------------------------------------------
            # Clean up to avoid VRAM issues
            # -----------------------------------------------------------------
            # Delete references to large temporary tensors and objects
            del (
                activations,
                sae_output,
                feature_acts,
                l0_per_sample,
                df,
                plot,
            )

            logger.debug("Cleaning up plotting variables.")

            collected = gc.collect()
            logger.debug(f"Garbage collector collected {collected} objects.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.sae_model.train()

    def _save_checkpoint(self, step: int) -> None:
        """Save model and optimizer state to disk to directory specified in
        config.

        Args:
            step (int): Current training step.
        """
        checkpoint_subpath = self.checkpoint_path / f"step_{step}"
        self.sae_model.save_to_disk(str(checkpoint_subpath))
        torch.save(
            self.optimizer.state_dict(),
            checkpoint_subpath / "optimizer.pt",
        )
        torch.save(
            self.lr_scheduler.state_dict(),
            checkpoint_subpath / "scheduler.pt",
        )
        trainer_state = {
            "global_step": step,
            "total_tokens_processed": self.total_tokens_processed,
        }
        torch.save(trainer_state, checkpoint_subpath / "trainer_state.pt")
        logger.info(f"Checkpoint saved successfully to {checkpoint_subpath}")

    def train(self) -> None:
        """Runs the main training loop."""
        logger.info("Starting training...")
        start_time = time.time()

        pbar = tqdm(
            total=self.total_training_steps,
            desc="Training Progress",
        )

        for batch in self.dataloader.iterate():
            if self.global_step >= self.total_training_steps:
                logger.warning(
                    f"Global step {self.global_step} exceeds total "
                    f"training steps {self.total_training_steps}. Halting."
                )
                break

            log_data = self.train_step(batch)
            # -------------------------------------------------------------
            # Update progress bar
            # -------------------------------------------------------------
            if self.global_step % self.trainer_cfg.log_frequency == 0:
                self._log_metrics(log_data, self.global_step)

                pbar.set_description(
                    f"Step {self.global_step} | "
                    f"Tokens: {self.total_tokens_processed:,} | "
                    f"Loss: {log_data.get('loss', 0):.4f} | "
                    f"L0: {log_data.get('l0_loss', 0):.2f} | "
                    f"L2: {log_data.get('l2_loss', 0):.4f} | "
                    f"LR: {log_data.get('lr', 0):.2e}"
                )

            # -------------------------------------------------------------
            # Checkpointing and Plotting
            # -------------------------------------------------------------
            if (
                self.trainer_cfg.plot_frequency > 0
                and self.global_step % self.trainer_cfg.plot_frequency == 0
            ):
                self._plot_feature_activation_histogram(self.global_step)

            if (
                self.trainer_cfg.save_frequency > 0
                and self.global_step % self.trainer_cfg.save_frequency == 0
                and self.global_step > 0
            ):
                self._save_checkpoint(self.global_step)

            pbar.update(1)

            # Break outer loop if inner loop broke due to exceeding total steps
            if self.global_step >= self.total_training_steps:
                break

        pbar.close()
        total_duration = time.time() - start_time
        logger.info(
            f"Training finished in {total_duration:.2f} seconds. Global step: "
            f"{self.global_step}"
        )

        logger.info("Saving final model checkpoint...")
        self._save_checkpoint(self.global_step)

        if self.trainer_cfg.wandb_project and wandb.run:
            wandb.finish()
            logger.info("Weights & Biases run finished.")
