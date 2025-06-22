"""
Visualization script for SAE ablation results with per-timestep analysis.

Example usage:
    python visualize_ablation_results.py --results_dir /path/to/ablation
    # auto-detects latest results
    python visualize_ablation_results.py
    # use ±SD instead of 95% CI
    python visualize_ablation_results.py --use_std
    # use AAAS color palette instead of JAMA
    python visualize_ablation_results.py --color aaas
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotnine as pn

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

# Academic theme for publication-quality plots
THEME_ACADEMIC = pn.theme_bw() + pn.theme(
    # text=element_text(family="monospace"),
    plot_title=pn.element_text(weight="bold", size=14, ha="center"),
    legend_text=pn.element_text(size=12),
    legend_title=pn.element_text(size=12, hjust=0.5),
    panel_background=pn.element_rect(fill="white", color="black"),
    panel_border=pn.element_rect(color="black", size=1),
    axis_ticks=pn.element_line(color="black", size=0.5),
    panel_grid_major=pn.element_line(color="grey", size=0.3, alpha=0.5),
    panel_grid_minor=pn.element_line(color="grey", size=0.2, alpha=0.3),
    legend_background=pn.element_blank(),  # no box around the legend panel
    legend_key=pn.element_blank(),  # no box behind each key
    legend_key_size=18,
    plot_margin=0.02,
    figure_size=(8, 5),
    axis_title=pn.element_text(size=14, weight="bold"),
    axis_text=pn.element_text(size=12),
    panel_spacing=0.05,
)


# =========================================================================== #
#                            Main Visualization Logic                         #
# =========================================================================== #
def main():
    parser = argparse.ArgumentParser(
        description="Visualize SAE ablation results"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory containing ablation (auto-detects latest by default)",
    )
    parser.add_argument(
        "--use_std",
        action="store_true",
        help="Use ±SD instead of 95% CI for error bars",
    )
    parser.add_argument(
        "--color",
        choices=["jama", "aaas"],
        default="jama",
        help="Color palette for plots (default: 'jama')",
    )
    args = parser.parse_args()

    colors = PAPER_COLORS[args.color]

    # -------------------------------------------------------------------------
    # Find and sanity check results directory
    # -------------------------------------------------------------------------
    if args.results_dir:
        results_dir = Path(args.results_dir)

        assert [
            d
            for d in results_dir.iterdir()
            if d.is_dir() and any(d.rglob("results.json"))
        ], "No ablation runs found in specified results directory"

    else:
        results_dir = find_latest_ablation_dir()
        logger.info(f"Auto-detected results directory: {results_dir}")

    df_overall, df_timestep = load_results(results_dir)

    if df_overall.empty:
        logger.error("No overall results found in the directory")
        return

    logger.info(
        f"Loaded {len(df_overall)} runs with "
        f"{len(df_timestep)} timestep measurements"
    )

    # Find ablated parameters
    exclude_cols = [
        "l2_loss",
        "l2_loss_normalized",
        "l0_loss",
        "variance_explained",
        "num_dead_features",
        "perc_dead_features",
        "seed",
    ]
    param_cols = [
        col
        for col in df_overall.columns
        if col not in exclude_cols and df_overall[col].nunique() > 1
    ]

    if not param_cols:
        logger.error("No varying parameters found!")
        return

    logger.info(f"Found ablated parameters: {param_cols}")

    # -------------------------------------------------------------------------
    # Create and save visualizations
    # -------------------------------------------------------------------------
    ablation_name = results_dir.name  # e.g., "topk_oaat_20250521_232450"
    output_dir = results_dir / f"visualization_{ablation_name}"
    output_dir.mkdir(exist_ok=True)

    # Tables for thesis
    create_paper_table(df_overall, param_cols, output_dir)

    error_type = "SD" if args.use_std else "95% CI"
    logger.info(f"\nCreating timestep plots with {error_type} error bars...")

    for param in param_cols:
        for metric in [
            "l2_loss",
            "l2_loss_normalized",
            "l0_loss",
            "variance_explained",
        ]:
            if (
                metric in df_timestep.columns
                and df_timestep[metric].notna().any()
            ):
                p = plot_timestep_curves(
                    df_timestep, param, metric, args.use_std, colors=colors
                )
                filename = (
                    output_dir
                    / f'timestep_{metric}_by_{param.replace(".", "_")}.pdf'
                )
                p.save(filename, dpi=300)
                logger.debug(f"Saved {filename}")

    logger.info(f"\nAll visualizations saved to {output_dir}")


# --------------------------------------------------------------------------- #
#                            Helper Functions                                 #
# --------------------------------------------------------------------------- #


def convert_timestep_to_diffusion_time(
    timestep: int, max_timestep: int = 961
) -> float:
    """
    Convert discrete timestep to normalized diffusion time for plotting.

    Args:
        timestep (int): The current timestep (1-indexed).
        max_timestep (int): The maximum timestep (default is 961).

    Returns:
        float: Normalized diffusion time in the range [0, 1].
    """
    return (timestep - 1) / (max_timestep - 1)


def parse_run_params(run_name: str) -> Dict[str, str]:
    """
    Parse parameter values from run directory name.

    Args:
        run_name (str): Name of the run directory, e.g.,
            "topk_oaat_20250521_232450".
    Returns:
        Dict[str, str]: Dictionary of parameter names and their values.
    """
    params = {}
    if "=" not in run_name:
        return params

    for part in run_name.split(","):
        if "=" in part:
            key, value = part.split("=", 1)
            params[key] = value
    return params


def load_results(results_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both overall and per-timestep results from ablation directory.

    Args:
        results_dir (Path): Path to the directory containing ablation results.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames containing overall and
            per-timestep results.
    """
    overall_data = []
    timestep_data = []

    for results_file in results_dir.rglob("results.json"):
        with open(results_file) as f:
            data = json.load(f)

        params = parse_run_params(results_file.parent.name)

        # Extract overall metrics
        overall = {
            "l2_loss": data.get("overall", {}).get("l2_loss"),
            "l2_loss_normalized": data.get("overall", {}).get(
                "l2_loss_normalized"
            ),
            "l0_loss": data.get("overall", {}).get("l0_loss"),
            "variance_explained": data.get("overall", {}).get(
                "variance_explained"
            ),
            "num_dead_features": data.get("overall", {}).get(
                "num_dead_features"
            ),
            "perc_dead_features": data.get("overall", {}).get(
                "perc_dead_features"
            ),
            **params,
        }
        overall_data.append(overall)

        # Extract per-timestep metrics
        per_timestep = data.get("per_timestep", {})
        max_timestep = (
            max(int(t) for t in per_timestep.keys() if t.isdigit())
            if per_timestep
            else 1
        )
        for timestep_str, metrics in per_timestep.items():
            timestep = int(timestep_str)
            entry = {
                "timestep": timestep,
                "diffusion_time": convert_timestep_to_diffusion_time(
                    timestep, max_timestep=max_timestep
                ),
                "l2_loss": metrics.get("l2_loss"),
                "l2_loss_normalized": metrics.get("l2_loss_normalized"),
                "l0_loss": metrics.get("l0_loss"),
                "variance_explained": metrics.get("variance_explained"),
                **params,
            }
            timestep_data.append(entry)

    return pd.DataFrame(overall_data), pd.DataFrame(timestep_data)


def get_clean_param_name(param: str) -> str:
    """
    Get clean parameter name for plots and tables.

    Args:
        param (str): The parameter name, e.g., "sae.k", "trainer.lr",
            "sae.d", etc.
    Returns:
        str: Cleaned parameter name suitable for display, e.g., "Top-K",
            "Learning Rate", "SAE Width", etc.
    """
    clean = param.replace("sae.", "").replace("trainer.", "").replace("_", " ")

    name_map = {
        "k": "Top-K",
        "lr": "Learning Rate",
        "d sae": "SAE Width",
        "effective batch size": "Batch Size",
        "use batch topk": "Batch TopK",
        "standardize input": "Standardize",
    }

    return name_map.get(clean, clean.title())


def get_metric_label(metric: str) -> str:
    """
    Get clean metric label for plots.

    Args:
        metric (str): The metric name, e.g., "l2_loss", "l2_loss_normalized",
            "l0_loss", etc.
    Returns:
        str: Cleaned metric label suitable for display, e.g., "MSE",
            "Normalized MSE", "L0 (Sparsity)", etc.
    """
    labels = {
        "l2_loss": "MSE",
        "l2_loss_normalized": "Normalized MSE",
        "l0_loss": "L0 (Sparsity)",
        "variance_explained": "Fraction of Variance Explained",
        "num_dead_features": "Number of Dead Features",
        "perc_dead_features": "Percentage of Dead Features",
    }
    return labels.get(metric, metric.replace("_", " ").title())


def find_latest_ablation_dir() -> Path:
    """
    Find the most recent ablation results directory.

    Returns:
        Path: Path to the latest ablation results directory.
    """
    base = Path("../../results/ablation")

    # Look for actual ablation run directories
    candidates = []
    for pattern in ["topk_*", "ablation_*", "*_*_*"]:  # timestamp patterns
        for path in base.glob(pattern):
            if path.is_dir() and any(path.rglob("results.json")):
                candidates.append(path)

    # Find the most recent one
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    raise ValueError(
        "No ablation results found. Please specify --results_dir or "
        "ensure results are in '../../results/ablation'"
    )


# --------------------------------------------------------------------------- #
#                       Plotting and Table Functions                          #
# --------------------------------------------------------------------------- #
def plot_timestep_curves(
    df: pd.DataFrame,
    param: str,
    metric: str,
    use_std: bool = False,
    colors: List[str] = PAPER_COLORS["jama"],
) -> pn.ggplot:
    """
    Create timestep curve plot for a given parameter and metric.

    Args:
        df (pd.DataFrame): DataFrame containing per-timestep results.
        param (str): The parameter to plot, e.g., "sae.k", "trainer.lr", etc.
        metric (str): The metric to plot, e.g., "l2_loss",
            "l2_loss_normalized", "l0_loss", "variance_explained", etc.
        use_std (bool): Whether to use ±SD instead of 95% CI for error bounds.
        colors (List[str]): List of colors for the parameter levels.

    Returns:
        pn.ggplot: The generated plot object.
    """
    # Calculate statistics
    stats = (
        df.groupby(["diffusion_time", param])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Calculate error bounds
    if use_std:
        stats["error_lower"] = stats["mean"] - stats["std"]
        stats["error_upper"] = stats["mean"] + stats["std"]
    else:
        stats["se"] = stats["std"] / np.sqrt(stats["count"])
        stats["error_lower"] = stats["mean"] - 1.96 * stats["se"]
        stats["error_upper"] = stats["mean"] + 1.96 * stats["se"]

    # Create plot
    p = (
        pn.ggplot(stats, pn.aes(x="diffusion_time", y="mean", color=param))
        + pn.geom_ribbon(
            pn.aes(ymin="error_lower", ymax="error_upper", fill=param),
            alpha=0.2,
        )
        + pn.geom_line(size=1.5)
        + pn.geom_point(size=2)
        + pn.scale_color_manual(values=colors[: df[param].nunique()])
        + pn.scale_fill_manual(values=colors[: df[param].nunique()])
        + pn.scale_x_reverse(breaks=np.arange(0, 1.1, 0.2), limits=[1, 0])
        + pn.scale_y_continuous(limits=(0, None), expand=(0, 0.05))
        + pn.labs(
            x="Diffusion Time",
            y=get_metric_label(metric),
            color=get_clean_param_name(param),
            fill=get_clean_param_name(param),
        )
        + THEME_ACADEMIC
        + pn.theme(figure_size=(10, 6), legend_position="right")
        + pn.guides(fill=False)  # Hide fill legend (redundant with color)
    )

    # Special formatting for variance explained
    if metric == "variance_explained":
        p = p + pn.scale_y_continuous(
            limits=[0, 1], labels=lambda x: [f"{val:.0%}" for val in x]
        )

    return p


def create_paper_table(
    df: pd.DataFrame, param_cols: List[str], output_dir: Path
) -> None:
    """
    Create publication-ready table with ablation results.

    Args:
        df (pd.DataFrame): DataFrame containing overall ablation results.
        param_cols (List[str]): List of parameter columns to include in the
            table.
        output_dir (Path): Directory to save the table files.

    """
    # Group by parameters and calculate statistics
    grouped = df.groupby(param_cols)

    table_data = []
    for params, group in grouped:
        if isinstance(params, str):
            params = [params]

        row = {
            col: params[i] if len(param_cols) > 1 else params
            for i, col in enumerate(param_cols)
        }

        # Calculate mean ± std for each metric
        metrics = [
            "l2_loss",
            "l2_loss_normalized",
            "l0_loss",
            "variance_explained",
            "num_dead_features",
            "perc_dead_features",
        ]
        for metric in metrics:
            if metric in group.columns:
                values = group[metric].dropna()
                mean = values.mean()
                std = values.std()
                if metric == "num_dead_features":
                    # Integer metric - round to nearest int
                    row[metric] = (
                        f"{mean:.0f} ± {std:.0f}"
                        if len(values) > 1
                        else f"{mean:.0f}"
                    )
                elif metric == "perc_dead_features":
                    # Percentage - always in [0,1] range
                    row[metric] = (
                        f"{mean:.1%} ± {std:.1%}"
                        if len(values) > 1
                        else f"{mean:.1%}"
                    )
                else:
                    row[metric] = (
                        f"{mean:.4f} ± {std:.4f}"
                        if len(values) > 1
                        else f"{mean:.4f}"
                    )

        row["n_seeds"] = len(group)
        table_data.append(row)

    # Create DataFrame and sort by normalized MSE
    table_df = pd.DataFrame(table_data)

    # Sort by best metric
    sort_col = (
        "l2_loss_normalized"
        if "l2_loss_normalized" in table_df.columns
        else "l2_loss"
    )
    table_df["_sort"] = table_df[sort_col].apply(
        lambda x: float(x.split(" ±")[0]) if x != "—" else float("inf")
    )
    table_df = table_df.sort_values("_sort").drop("_sort", axis=1)

    # Rename columns
    rename_map = {
        "l2_loss": "MSE",
        "l2_loss_normalized": "Norm. MSE",
        "l0_loss": "L0",
        "variance_explained": "Var. Expl.",
        "num_dead_features": "Dead Features",
        "perc_dead_features": "% Dead",
        "n_seeds": "Seeds",
    }
    for param in param_cols:
        rename_map[param] = get_clean_param_name(param)

    table_df = table_df.rename(columns=rename_map)

    # Reorder columns
    param_cols_renamed = [rename_map[p] for p in param_cols]
    metric_cols = [
        "MSE",
        "Norm. MSE",
        "L0",
        "Var. Expl.",
        "Dead Features",
        "% Dead",
        "Seeds",
    ]
    ordered_cols = param_cols_renamed + [
        c for c in metric_cols if c in table_df.columns
    ]
    table_df = table_df[ordered_cols]

    # Save in multiple formats
    table_df.to_latex(
        output_dir / "paper_table.tex", index=False, escape=False
    )

    with open(output_dir / "paper_table.md", "w") as f:
        f.write(table_df.to_markdown(index=False))

    logger.info(f"\nSaved paper table to {output_dir}/paper_table.{{tex,md}}")


if __name__ == "__main__":
    main()
