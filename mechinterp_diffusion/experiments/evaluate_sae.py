"""
Script for evaluating Sparse Autoencoders (SAEs) on test datasets.

Example usage:
    python evaluate_sae.py \
        --model_path /path/to/sae \
        --dataset_path /path/to/dataset \
        --output_path /path/to/output
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #


import argparse
import gc
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from umap import UMAP

sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import seaborn as sns
from config import TopKSAEConfig
from core.sae.base_sae import BaseSAE
from core.sae.topk_sae import TopKSAE

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


PAPER_COLORS = {
    # see: https://nanx.me/ggsci/reference/pal_jama.html
    "jama": [
        "#DF8F44FF",
        "#00A1D5FF",
        "#B24745FF",
        "#374E55FF",
        "#79AF97FF",
        "#6A6599FF",
        "#80796BFF",
    ],
    # see: https://nanx.me/ggsci/reference/pal_aaas.html
    "aaas": [
        "#3B4992FF",
        "#EE0000FF",
        "#008B45FF",
        "#631879FF",
        "#008280FF",
        "#BB0021FF",
        "#5F559BFF",
        "#A20056FF",
        "#808180FF",
        "#1B1919FF",
    ],
}
plt.style.use("default")

# TODO: r2; normalized mse; feature distribution; norms plot
# TODO: UMAP decoder
# TODO: Allow for multiple SAEs to be evaluated at once; or maybe save to SAE path? # noqa: E501
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

    parser.add_argument(
        "--colors",
        type=str,
        choices=["jama", "aaas"],
        default="aaas",
        help="Color palette to use for plots",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for evaluation.",
    )

    parser.add_argument(
        "--spatial_resolution",
        type=int,
        nargs=2,
        default=[16, 16],
        help="Spatial resolution of latents (height, width).",
    )

    args = parser.parse_args()
    print(args)

    sns.set_palette(PAPER_COLORS[args.colors])

    logging.info("Starting evaluation with the following parameters:")

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
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_samples=args.num_samples,
        resolution=args.spatial_resolution,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
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
        model_path: str,
        dataset_path: str,
        resolution: List[int],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_samples: Optional[int] = None,
        batch_size: int = 100,
        output_dir: Union[str, Path] = "../../results/evaluation_results",
    ) -> None:
        """
        Initialize the Evaluator for Sparse Autoencoders (SAEs).

        Args:
            sae (BaseSAE): Sparse Autoencoder to evaluate.
            dataset (Dataset): Test dataset in Hugging Face format.
            model_path (str): Path to the saved SAE model.
            dataset_path (str): Path to the dataset in Hugging Face format.
            device (str, optional): Device to run evaluaiton on. Defaults to
                "cuda" if it is available, otherwise "cpu".
            num_samples (Optional[int], optional): Number of test samples to
                use for evaluation. Defaults to None.
            resolution (List[int], optional): Spatial resolution of the latents
                (height, width). Defaults to [16, 16].
            batch_size (int, optional): Batch size to use for test loaders.
                Defaults to 100.
            output_dir (Union[str, Path], optional): Directory to save the
                evaluation results. Defaults to
                "../../results/evaluation_results".
        """
        num_timesteps = len(np.unique(dataset["timestep"]))
        self.max_timestep = max(dataset["timestep"])
        assert (
            num_samples is None or num_samples <= len(dataset) // num_timesteps
        ), "num_samples must be less than or equal to number of test examples"

        self.resolution = resolution
        self.num_samples = num_samples
        self.sae = sae.to(device=device, dtype=torch.float32)
        self.dataset = dataset
        self.device = torch.device(device)
        self.timesteps = sorted(list(set([ts for ts in dataset["timestep"]])))
        # Subset test dataset:
        if self.num_samples is None:
            self.total_samples = len(self.dataset)
            self.num_samples = len(self.dataset) // len(self.timesteps)
        else:
            self.total_samples = min(
                len(self.dataset), self.num_samples * len(self.timesteps)
            )

        self.dataset = self.dataset.select(range(self.total_samples))
        self.dataset_path = dataset_path
        self.model_path = model_path

        # avoid distribution of across time
        self.batch_size = 1 if self.sae.cfg.use_batch_topk else batch_size

        self.sae.eval()

        self.output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    def run(self) -> None:
        """Run the evaluation process."""
        logger.info("-" * 50)
        logger.info("Starting evaluation".center(50))
        logger.info("-" * 50)

        self._evaluate_reconstruction()
        self._plot_reconstruction()
        self._evaluate_features()
        self._save_config()

        logger.info("-" * 50)
        logger.info("Evaluation Completed".center(50))
        logger.info("-" * 50)

    def _save_config(self) -> None:
        """Save Configuration of Evaluation as JSON."""
        summary = {
            "sae_type": type(self.sae).__name__,
            "sae_path": self.model_path,
            "dataset_path": self.dataset_path,
            "num_samples": self.num_samples,
            "resolution": self.resolution,
            "num_timesteps": len(self.timesteps),
            "max_timestep": self.max_timestep,
        }

        import json

        with open(self.output_dir / "evaluation_config.json", "w") as f:
            json.dump(summary, f, indent=4)
        logger.info("Saved evaluation configuration.")

    @torch.no_grad()
    def _evaluate_reconstruction(
        self,
    ) -> None:
        """
        Evaluate the reconstruction performance of the SAE across all
        timesteps.
        """
        timestep_metrics = {}

        for timestep in tqdm(self.timesteps, desc="Evaluating timesteps"):
            # Get indices for this timestep
            timestep_indices = [
                i
                for i, ts in enumerate(self.dataset["timestep"])
                if ts == timestep
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
                acts = batch["activations"].to(
                    dtype=torch.float32, device=self.device
                )

                output = self.sae(acts)

                # Total MSE for this batch
                batch_l2 = (
                    (output["sae_out"].view_as(acts) - acts).pow(2).sum()
                )

                stats["l2"].append(batch_l2.detach().cpu().item())

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
            mse = sum(stats["l2"]) / n_total_values

            timestep_metrics[timestep] = {}
            timestep_metrics[timestep].update(
                mse=mse,
                mse_normalized=mse / var if var > 0 else mse,
                variance_explained=max(0, 1 - mse / var) if var > 0 else 0,
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
        total_mse = (
            sum(
                r["mse"] * r["total_samples"]
                for r in timestep_metrics.values()
            )
            / total_n
        )

        feature_is_inactive = (
            self.sae.num_tokens_inactive
            >= self.sae.cfg.num_tokens_dead_threshold
        )

        overall = {}
        overall.update(
            mse=total_mse,
            l2_loss_normalized=(
                total_mse / overall_variance
                if overall_variance > 0
                else total_mse
            ),
            variance_explained=(
                max(0, 1 - total_mse / overall_variance)
                if overall_variance > 0
                else 0
            ),
            total_samples=total_n,
            num_dead_features=feature_is_inactive.sum().item(),
            perc_dead_features=feature_is_inactive.float().mean().item(),
        )

        self.timestep_df = pd.DataFrame.from_dict(
            timestep_metrics, orient="index"
        )
        self.timestep_df.reset_index(inplace=True)
        self.timestep_df.rename(columns={"index": "timestep"}, inplace=True)
        self.overall_df = pd.DataFrame([overall])

        self.timestep_df.to_csv(
            Path(self.output_dir) / "timestep_metrics.csv", index=False
        )
        self.overall_df.to_csv(
            Path(self.output_dir) / "overall_metrics.csv", index=False
        )

    def _convert_timestep_to_diffusion_time(self, timestep: int) -> float:
        """
        Convert discrete timestep to normalized diffusion time for plotting.

        Args:
            timestep (int): The current timestep (1-indexed).

        Returns:
            float: Normalized diffusion time in the range [0, 1].
        """
        return (timestep - 1) / (self.max_timestep - 1)

    def _plot_reconstruction(self) -> None:
        self.timestep_df["diffusion_time"] = self.timestep_df[
            "timestep"
        ].apply(self._convert_timestep_to_diffusion_time)

        metrics = [
            ("mse", "MSE"),
            ("mse_normalized", "Normalized MSE"),
            ("variance_explained", "Fraction of Variance Explained"),
        ]

        for metric, label in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot the line
            ax.plot(
                self.timestep_df["diffusion_time"],
                self.timestep_df[metric],
                linewidth=1.5,
                marker="o",
                markersize=4,
                markeredgewidth=0,
            )

            ax.set_xlabel("Diffusion Time", fontsize=12)
            ax.set_ylabel(label, fontsize=12)
            # Reverse x-axis to match diffusion time convention
            ax.set_xlim(1, 0)
            ax.set_xticks(np.arange(0, 1.1, 0.2))

            # Set y-axis to start at 0
            ax.set_ylim(bottom=0, top=1.1 * self.timestep_df[metric].max())

            # Special formatting for R2
            if metric == "variance_explained":
                ax.set_ylim(0, 1)
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda y, _: f"{y:.0%}")
                )

            ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
            ax.set_axisbelow(True)

            sns.despine()

            plt.tight_layout()

            output_path = Path(self.output_dir) / f"timestep_{metric}.pdf"
            fig.savefig(output_path, dpi=300)
            plt.close()

            logger.info(f"Saved {label} plot to {output_path}")

    def _evaluate_features(self) -> None:
        gc.collect()
        torch.cuda.empty_cache()

        # [1280, 5120] -> [1280, 2]
        W_dec = self.sae.W_dec.weight.detach().cpu().numpy()

        logger.info(
            f"Fitting UMAP for Decoder with weight shape: {W_dec.shape}"
        )
        reducer = UMAP(n_components=2, metric="cosine", random_state=42)

        embedding = reducer.fit_transform(W_dec)

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.scatter(
            embedding[:, 0], embedding[:, 1], c="#191970", s=10, alpha=0.6
        )

        # Labels
        ax.set_xlabel("UMAP 1", fontsize=12)
        ax.set_ylabel("UMAP 2", fontsize=12)
        ax.set_title("UMAP of SAE Features", fontsize=14)

        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)

        sns.despine()

        plt.tight_layout()

        output_path = Path(self.output_dir) / "decoder_umap.pdf"
        fig.savefig(output_path, dpi=300)
        plt.close()

        logger.info(f"Saved decoder UMAP to {output_path}")


if __name__ == "__main__":
    main()
