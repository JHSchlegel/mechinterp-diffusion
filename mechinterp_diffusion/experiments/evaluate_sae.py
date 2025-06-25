"""
Script for evaluating Sparse Autoencoders (SAEs) on test datasets.

Example usage:
    python evaluate_sae.py --model_path /path/to/sae \
        --dataset_path /path/to/dataset \
        --output_path /path/to/output
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #


import argparse
import gc
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from config import TopKSAEConfig
from core.sae.base_sae import BaseSAE
from core.sae.topk_sae import TopKSAE

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# TODO: include spatial activation in plotting
# TODO: density ridges
# TODO: r2; normalized mse; feature distribution; norms plot
# TODO: UMAP decoder
# =========================================================================== #
#                            Main Functionality                               #
# =========================================================================== #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Sparse Autoencoders (SAEs)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved SAE model",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset in Hugging Face format",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"../../results/evaluation_results/{datetime.now().strftime('%Y%m%d_%H%M%S')}",  # noqa: E501
        help="Path to save the evaluation results",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate per timestep. "
        "If None, evaluates all samples.",
    )

    args = parser.parse_args()

    # Load the SAE model
    sae = TopKSAE.load_from_disk(
        args.model_path,
        config_class=TopKSAEConfig,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    test_dataset = load_from_disk(args.dataset_path)

    evaluator = Evaluator(
        sae=sae,
        dataset=test_dataset,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_samples=args.num_samples,
        batch_size=100,
    )
    evaluator.run()


# =========================================================================== #
#                              Evaluator Class                                #
# =========================================================================== #


class Evaluator:
    def __init__(
        self,
        sae: BaseSAE,
        dataset: Dataset,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_samples: Optional[int] = None,
        batch_size: int = 100,
    ) -> None:
        """_summary_

        Args:
            sae (BaseSAE): _description_
            dataset (Dataset): _description_
            device (str, optional): _description_. Defaults to "cuda" if it is
                available, otherwise "cpu".
            num_samples (Optional[int], optional): _description_. Defaults to
                None.
            batch_size (int, optional): Batch size to use for test loaders.
                Defaults to 100.
        """
        num_timesteps = len(np.unique(dataset["timestep"]))
        assert (
            num_samples is None or num_samples <= len(dataset) // num_timesteps
        ), "num_samples must be less than or equal to number of test examples"

        self.num_samples = num_samples
        self.sae = sae.to(device)
        self.dataset = dataset
        self.device = torch.device(device)
        self.timesteps = sorted(
            list(set([ts.item() for ts in dataset["timestep"]]))
        )

        # Subset test dataset:
        if self.num_samples is None:
            self.total_samples = len(self.dataset)
            self.num_samples = len(self.dataset) // len(self.timesteps)
        else:
            self.total_samples = min(
                len(self.dataset), self.num_samples * len(self.timesteps)
            )

        self.dataset = self.dataset.select(range(self.total_samples))

        # avoid distribution of across time
        self.batch_size = 1 if self.sae.use_batch_topk else batch_size

        self.sae.eval()

    def run(self) -> None:
        """Run the evaluation process."""
        self._evaluate_reconstruction()
        self._evaluate_features()
        self._plot_results()

        self._save_results()
        raise NotImplementedError("This method is not implemented yet.")

    def _save_results(self) -> None:
        raise NotImplementedError("This method is not implemented yet.")

    def _plot_results(self) -> None:
        raise NotImplementedError("This method is not implemented yet.")

    @torch.no_grad()
    def _evaluate_reconstruction(
        self,
    ) -> Dict[str, Dict[str, Union[float, int]]]:
        timestep_metrics = {}
        for timestep in tqdm(self.timesteps, desc="Evaluating timesteps"):
            # Get indices for this timestep
            timestep_indices = [
                i
                for i, ts in enumerate(self.dataset["timestep"])
                if ts.item() == timestep
            ]

            if not timestep_indices:
                logger.warning(
                    f"No samples found for timestep {timestep}. Skipping."
                )
                continue

            timestep_dataset = self.dataset.select(timestep_indices)
            timestep_dataset.set_format(type="torch", columns=["activations"])

            # Create separate test loaders for each timestep
            test_loader = DataLoader(
                timestep_dataset,
                batch_size=self.batch_size,
                shuffle=False,
            )

            # -----------------------------------------------------------------
            # Extract Batchwise Statistics
            # -----------------------------------------------------------------
            stats = {"l2": []}
            act_sum = 0.0
            act_sq_sum = 0.0
            n_total_values = 0

            for batch in test_loader:
                acts = batch["activations"].to(self.device)

                output = self.sae(acts)

                # sae_input, info = sae.preprocess_input(acts)
                # reconstruct = sae.postprocess_output(
                #     sae.decode(
                #         torch.nn.functional.relu(sae.encode(sae_input))
                #     ),
                #     info,
                # )

                # Total MSE for this batch
                batch_mse = (
                    (output["sae_out"].view_as(acts) - acts).pow(2).sum()
                )

                stats["l2"].append(batch_mse.detach().cpu().item())

                # Accumulate for variance (across all activation values)
                act_sum += acts.sum().item()
                act_sq_sum += acts.pow(2).sum().item()
                n_total_values += acts.numel()

                del acts, output

            torch.cuda.empty_cache()
            gc.collect()
            # -----------------------------------------------------------------
            # Aggregate Statistics for Timestep
            # -----------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # Aggregate across all timesteps
        # ---------------------------------------------------------------------
        total_n = sum(r["total_samples"] for r in timestep_metrics.values())

        # Compute overall variance
        total_act_sum = sum(
            r["activation_sum"] for r in timestep_metrics.values()
        )
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
            r["l2_loss"] * r["total_samples"]
            for r in timestep_metrics.values()
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

    def _evaluate_features(self) -> pd.DataFrame:
        raise NotImplementedError("This method is not implemented yet.")
