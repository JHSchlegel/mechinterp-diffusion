"""
Module for analyzing SAE feature activity across diffusion model blocks.
"""

# =========================================================================== #
#                               Packages and Presets                          #
# =========================================================================== #

import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, load_from_disk
from simple_parsing import Serializable, parse
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import TopKSAEConfig
from core.sae.topk_sae import TopKSAE
from core.utils.analysis_utils import (
    convert_timestep_to_diffusion_time,
    get_block_label,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",  # noqa: E501
)
logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-paper")

PAPER_COLORS = {
    "jama": [
        "#DF8F44FF",
        "#00A1D5FF",
        "#B24745FF",
        "#374E55FF",
        "#79AF97FF",
        "#6A6599FF",
        "#80796BFF",
    ],
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

# =========================================================================== #
#                       Feature Activity Analysis Configuration               #
# =========================================================================== #


@dataclass
class FeatureActivityConfig(Serializable):
    # -------------------------------------------------------------------------
    # Model paths configuration
    # -------------------------------------------------------------------------
    sae_model_paths: List[str] = field(
        default_factory=lambda: [
            "../../checkpoints/sae/down_blocks.2.attentions.0/TopKSAE_dsae-5120_timesteps-all_20250816_083716/step_488282",  # noqa: E501
            "../../checkpoints/sae/up_blocks.1.attentions.1/TopKSAE_dsae-5120_timesteps-all_20250815_224124/step_488282",  # noqa: E501
        ]
    )
    """List of paths to the trained SAE model directories."""

    activation_dataset_paths: List[str] = field(
        default_factory=lambda: [
            "../../data/activations/stable-diffusion-2-1/laion/test/subset_size-10000/25-inference-steps/every-1-steps/unet.down_blocks.2.attentions.0",
            "../../data/activations/stable-diffusion-2-1/laion/test/subset_size-10000/25-inference-steps/every-1-steps/unet.up_blocks.1.attentions.1",
        ]
    )
    """
    List of paths to activation datasets, corresponding to the SAE model paths.
    """

    output_dir: str = "../../results/feature_activity"
    """Directory to save analysis plots and reports."""

    # -------------------------------------------------------------------------
    # Processing configuration
    # -------------------------------------------------------------------------
    spatial_size: Tuple[int, int] = (16, 16)
    """Spatial resolution of the latents (height, width)."""

    batch_size: int = 256
    """Batch size for processing per-timestep stats."""

    num_samples: int = 10000
    """Limit samples to process per timestep."""

    compute_summary_stats: bool = True
    """Whether to compute cross-timestep summary statistics."""

    def __post_init__(self):
        """Validate configuration after initialization."""
        if len(self.sae_model_paths) != len(self.activation_dataset_paths):
            raise ValueError(
                f"Number of SAE model paths ({len(self.sae_model_paths)}) must"
                f" match number of activation dataset paths "
                f"({len(self.activation_dataset_paths)})"
            )


# =========================================================================== #
#                                 Main Function                               #
# =========================================================================== #


def main():
    config = parse(FeatureActivityConfig)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results, all_summary_results = [], []

    logger.info(
        f"Starting analysis of {len(config.sae_model_paths)} SAE models"
    )
    logger.info(f"{'-'*60}\n{'SAE FEATURE ACTIVITY ANALYSIS':^60}\n{'-'*60}")

    for i, (sae_path_str, dataset_path_str) in enumerate(
        zip(
            config.sae_model_paths,
            config.activation_dataset_paths,
            strict=False,
        ),
        1,
    ):
        logger.info(
            f"Processing SAE {i}/{len(config.sae_model_paths)}: "
            f"{Path(sae_path_str).name}"
        )
        analyzer = FeatureActivityAnalyzer(
            sae_path=sae_path_str,
            dataset_path=dataset_path_str,
            spatial_size=config.spatial_size,
            batch_size=config.batch_size,
            num_samples=config.num_samples,
            compute_summary_stats=config.compute_summary_stats,
        )
        result_df, summary_stats = analyzer.run()
        all_results.append(result_df)
        all_summary_results.append(summary_stats)

    combined_df = pd.concat(all_results, ignore_index=True)
    generate_plots(combined_df, output_dir)
    csv_path = output_dir / "feature_activity_full_report.csv"
    combined_df.to_csv(csv_path, index=False)
    logger.info(f"Reports saved to {csv_path}.")
    logger.info(f"Analysis complete! Results saved to: {output_dir}")


# =========================================================================== #
#                         Feature Activity Analyzer                           #
# =========================================================================== #


class FeatureActivityAnalyzer:
    def __init__(
        self,
        sae_path: str,
        dataset_path: str,
        spatial_size: Tuple[int, int],
        batch_size: int,
        num_samples: Optional[int] = None,
        compute_summary_stats: bool = True,
        device: str = "cuda",
    ) -> None:
        """Feature Activity Analyzer for analyzing SAE feature activity.

        Args:
            sae_path (str): Path to Sparse Autoencoder.
            dataset_path (str): Path to dataset.
            spatial_size (Tuple[int, int]): Spatial size (height, width).
            batch_size (int): Batch size for dataloader.
            num_samples (Optional[int], optional): Number of samples to
                analyze. Defaults to None.
            compute_summary_stats (bool, optional): Whether to compute summary
                statistics. Defaults to True.
            device (str, optional): Device to use for computation.
                Defaults to "cuda".
        """
        self.sae_path = sae_path
        self.dataset_path = dataset_path
        self.spatial_h, self.spatial_w = spatial_size
        self.batch_size = batch_size
        self.compute_summary_stats = compute_summary_stats
        self.device = device if torch.cuda.is_available() else "cpu"
        self.block_label = get_block_label(Path(sae_path))

        self._load_models()
        self._load_dataset(num_samples)

    def _load_models(self) -> None:
        """
        Load trained TopKSAEs from disk.
        """
        logger.info(
            f"[{self.block_label}] Loading SAE from {self.sae_path}..."
        )
        self.sae = TopKSAE.load_from_disk(
            self.sae_path, config_class=TopKSAEConfig, device=self.device
        )
        self.sae.eval()

    def _load_dataset(self, num_samples: Optional[int]) -> None:
        """Load dataset from disk.

        Args:
            num_samples (Optional[int]): Number of samples to load. If
                None, load all samples.
        """
        logger.info(
            f"[{self.block_label}] Loading dataset from {self.dataset_path}..."
        )
        if not Path(self.dataset_path).exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.dataset_path}"
            )

        dataset = load_from_disk(self.dataset_path)
        if num_samples:
            dataset = self._subsample_dataset(dataset, num_samples)

        logger.info(
            f"[{self.block_label}] Dataset size: {len(dataset)} samples"
        )
        dataset.set_format(type="torch", columns=["activations", "timestep"])
        self.dataset = dataset
        self.unique_timesteps = sorted(
            torch.unique(self.dataset["timestep"]).tolist()
        )
        self.max_timestep = (
            max(self.unique_timesteps) if self.unique_timesteps else 1
        )
        self.num_timesteps = len(self.unique_timesteps)

    def _subsample_dataset(
        self, dataset: Dataset, num_samples: int
    ) -> Dataset:
        """Subsample the dataset to a fixed number of samples per timestep.

        Args:
            dataset (Dataset): The dataset to subsample.
            num_samples (int): The number of samples to keep per timestep.

        Returns:
            Dataset: The subsampled dataset.
        """
        unique_timesteps_list = sorted(list(set(dataset["timestep"])))
        subsampled_datasets = []
        for ts in unique_timesteps_list:
            indices_for_ts = [
                i for i, t in enumerate(dataset["timestep"]) if t == ts
            ]
            selected_indices = indices_for_ts[
                : min(num_samples, len(indices_for_ts))
            ]
            subsampled_datasets.append(dataset.select(selected_indices))
        return concatenate_datasets(subsampled_datasets)

    @torch.no_grad()
    def _calculate_activity_stats(
        self,
    ) -> Tuple[defaultdict, defaultdict, defaultdict]:
        """
        Calculate nr. of distinct active features, average active locations,
        and mean activation values per timestep.

        Returns:
            Tuple[defaultdict, defaultdict, defaultdict]: distinct counts,
                average locations, and mean values per timestep
        """
        dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False
        )
        stats = {
            "distinct": defaultdict(list),
            "locs": defaultdict(list),
            "vals": defaultdict(list),
        }
        pbar = tqdm(
            dataloader, desc=f"Analyzing {self.block_label}", leave=False
        )

        for batch in pbar:
            activations = batch["activations"].to(
                self.device, non_blocking=True
            )
            timesteps = batch["timestep"]
            bs = timesteps.shape[0]

            sae_output = self.sae(activations)
            sparse_acts = sae_output["feature_acts"]
            # [bs, d_sae]
            sparse_acts_spatial = sparse_acts.view(
                bs, self.spatial_h, self.spatial_w, -1
            )
            # [bs, h, w, d_sae]
            active_mask = sparse_acts_spatial > 0.0

            feature_is_active = active_mask.any(dim=1).any(dim=1)
            # [bs, d_sae]
            distinct_counts = feature_is_active.sum(dim=1)

            total_active_locs = (
                active_mask.sum(dim=(1, 2)) * feature_is_active
            ).sum(dim=1)
            avg_locs_per_feature = torch.nan_to_num(
                total_active_locs / distinct_counts
            )

            # For each active feature, compute its mean activation across space
            # Then average those means
            mean_activation_values = sparse_acts_spatial.mean(
                dim=(1, 2, 3)
            )  # [batch_size]

            for i in range(bs):
                ts = timesteps[i].item()
                stats["distinct"][ts].append(distinct_counts[i].item())
                stats["locs"][ts].append(avg_locs_per_feature[i].item())
                stats["vals"][ts].append(mean_activation_values[i].item())

        return stats["distinct"], stats["locs"], stats["vals"]

    @torch.no_grad()
    def _calculate_summary_stats(self) -> dict:
        """
        Calculates summary stats by processing ONE prompt at a time to ensure
        a constant and minimal memory footprint, preventing OOM errors.
        """
        logger.info(
            f"[{self.block_label}] Computing cross-timestep summary statistics"
        )
        num_unique_samples = len(self.dataset) // self.num_timesteps

        results = {"total_feats": [], "mean_feats": [], "mean_locs": []}
        pbar = tqdm(
            range(num_unique_samples),
            desc="Computing summaries (prompt-by-prompt)",
            leave=False,
        )

        for sample_idx in pbar:
            # Get all timesteps for this single sample; note that
            # test set is ordered -> simple to extract all activations for a
            # given prompt sample
            indices = [
                sample_idx * self.num_timesteps + t
                for t in range(self.num_timesteps)
            ]

            # Load data for just this one sample, creating a small batch
            activations = self.dataset.select(indices)["activations"].to(
                self.device
            )

            sparse_acts = self.sae(activations)[
                "feature_acts"
            ]  # Shape: [num_timesteps, d_sae]

            active_mask = sparse_acts > 0.0

            # Total unique features used for this image
            results["total_feats"].append(active_mask.any(dim=0).sum().item())

            # Mean features per timestep for this image
            results["mean_feats"].append(
                active_mask.sum(dim=1).float().mean().item()
            )

            # Mean spatial locations per feature firing
            sparse_acts_spatial = sparse_acts.view(
                self.num_timesteps, self.spatial_h, self.spatial_w, -1
            )
            locs_per_firing = (sparse_acts_spatial > 0.0).sum(dim=(1, 2))

            total_locs = locs_per_firing.sum()
            total_firings = (locs_per_firing > 0).sum()
            avg_locs = total_locs / total_firings if total_firings > 0 else 0.0
            results["mean_locs"].append(avg_locs.item())

        return {
            "block_label": self.block_label,
            "mean_total_features": np.mean(results["total_feats"]),
            "std_total_features": np.std(results["total_feats"]),
            "mean_features_per_timestep": np.mean(results["mean_feats"]),
            "std_features_per_timestep": np.std(results["mean_feats"]),
            "mean_spatial_locs": np.mean(results["mean_locs"]),
            "std_spatial_locs": np.std(results["mean_locs"]),
        }

    def run(self) -> Tuple[pd.DataFrame, Optional[dict]]:
        """
        Runs the analysis and generates the report DataFrame and summary
        statistics.

        Returns:
            Tuple[pd.DataFrame, Optional[dict]]: The report DataFrame and
                summary statistics.
        """
        logger.info(f"Starting analysis for block: {self.block_label}")
        distinct_stats, spatial_stats, activation_stats = (
            self._calculate_activity_stats()
        )

        # ---------------------------------------------------------------------
        # Aggregate and summarize results
        # ---------------------------------------------------------------------
        report_data = []
        for timestep in sorted(list(self.unique_timesteps)):
            distinct_counts = np.array(distinct_stats.get(timestep, [0]))
            report_data.append(
                {
                    "Timestep": timestep,
                    "Diffusion Time": convert_timestep_to_diffusion_time(
                        timestep, self.max_timestep
                    ),
                    "Mean Distinct Active Features": np.mean(distinct_counts),
                    "Std Dev Distinct Active Features": np.std(
                        distinct_counts
                    ),
                    "Mean Spatial Locs per Active Feature": np.mean(
                        spatial_stats.get(timestep, [0])
                    ),
                    "Std Dev Spatial Locs per Active Feature": np.std(
                        spatial_stats.get(timestep, [0])
                    ),
                    "Mean Spatial Activation Value": np.mean(
                        activation_stats.get(timestep, [0])
                    ),
                    "Std Dev Spatial Activation Value": np.std(
                        activation_stats.get(timestep, [0])
                    ),
                    "Num Samples": len(distinct_counts),
                }
            )

        report_df = pd.DataFrame(report_data)
        report_df["Block Label"] = self.block_label
        report_df["Total SAE Features"] = self.sae.cfg.d_sae

        summary_stats = (
            self._calculate_summary_stats()
            if self.compute_summary_stats
            else None
        )

        logger.info(f"Finished analysis for block: {self.block_label}")
        return report_df, summary_stats


# =========================================================================== #
#                         Plotting and Reporting                              #
# =========================================================================== #


def create_single_plot(
    df: pd.DataFrame, y_col: str, std_col: str, y_label: str, output_path: Path
) -> None:
    """Create a single plot for the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to plot.
        y_col (str): The name of the column to plot on the y-axis.
        std_col (str): The name of the column containing the standard
            deviation.
        y_label (str): The label for the y-axis.
        output_path (Path): The path where the plot will be saved.
    """
    # Set the global font properties
    plt.rcParams.update(
        {
            "font.size": 18,
            "font.weight": "bold",
            "axes.labelsize": 26,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "legend.fontsize": 24,
        }
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    # Configure tick parameters
    ax.tick_params(axis="both", which="major", labelsize=18, width=2, length=8)

    # Make tick labels bold
    for label in ax.get_xticklabels():
        label.set_fontsize(18)
        label.set_fontweight("bold")

    for label in ax.get_yticklabels():
        label.set_fontsize(18)
        label.set_fontweight("bold")

    colors = PAPER_COLORS["aaas"]
    for i, label in enumerate(sorted(df["Block Label"].unique())):
        block_df = df[df["Block Label"] == label].sort_values("Diffusion Time")
        color = colors[i % len(colors)]
        ax.plot(
            block_df["Diffusion Time"],
            block_df[y_col],
            marker="o",
            markersize=6,
            linewidth=2.5,
            label=label,
            color=color,
        )
        ax.fill_between(
            block_df["Diffusion Time"],
            block_df[y_col] - block_df[std_col],
            block_df[y_col] + block_df[std_col],
            color=color,
            alpha=0.2,
            linewidth=0,
        )

    ax.set_xlabel("Diffusion Time", fontsize=26, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=26, fontweight="bold")
    ax.set_xlim(1, 0)
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Configure legend
    legend = ax.legend(
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
        borderpad=1,
        columnspacing=1.2,
        handletextpad=0.8,
        loc="best",
    )
    legend.get_title().set_fontweight("bold")
    legend.get_title().set_fontsize(22)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, format="pdf", bbox_inches="tight")
    logger.info(f"Plot saved to {output_path}")
    plt.close(fig)


def generate_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate plots for the given DataFrame and save them to the specified
    output directory.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to plot.
        output_dir (Path): The directory where the plots will be saved.
    """
    create_single_plot(
        df=df,
        y_col="Mean Distinct Active Features",
        std_col="Std Dev Distinct Active Features",
        y_label="Active Features",
        output_path=output_dir / "distinct_feature_activity_vs_time.pdf",
    )
    create_single_plot(
        df=df,
        y_col="Mean Spatial Locs per Active Feature",
        std_col="Std Dev Spatial Locs per Active Feature",
        y_label="Spatial Locations per Feature",
        output_path=output_dir
        / "spatial_locations_per_active_feature_vs_time.pdf",
    )
    create_single_plot(
        df=df,
        y_col="Mean Spatial Activation Value",
        std_col="Std Dev Spatial Activation Value",
        y_label="Mean Activation Magnitude",
        output_path=output_dir / "mean_spatial_activation_values_vs_time.pdf",
    )


if __name__ == "__main__":
    main()
