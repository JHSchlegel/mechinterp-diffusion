"""
This module provides functions to plot causal circuits previously discovered
using circuit_discovery.py.
"""

# =========================================================================== #
#                             Packages and Presets                            #
# =========================================================================== #
import io
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from activation_utils import SparseAct
from circuit_utils import (
    get_topk_component_indices,
)
from graphviz import Digraph

logger = logging.getLogger(__name__)


# =========================================================================== #
#                        Causal Circuit Plotting                             #
# =========================================================================== #


def plot_causal_circuit(
    nodes: Dict[Union[int, str], Union[SparseAct, float]],
    edges: Dict[Tuple[int, Union[int, str]], torch.Tensor],
    timesteps: List[int],
    save_path: str,
    top_k_nodes_per_ts: int = 10,
    top_k_edges: int = 50,
    num_inference_steps: int = 25,
) -> None:
    """
    Filters and plots the causal circuit from aggregated node and edge data.


    Args:
        nodes (Dict[Union[int, str], Union[SparseAct, float]]): A dictionary
            mapping timesteps (int) or "y" to SparseAct objects (for timesteps
            with associated activations) or floats
            (for "y" with a scalar value).
        edges (Dict[Tuple[int, Union[int, str]], torch.Tensor]): A dictionary
            mapping (t_up, t_down) tuples to edge tensors. Edge tensors can be
            either sparse COO tensors (for JVPs between features) or dense
            tensors (for edges to the probe).
        timesteps (List[int]): List of timesteps to consider.
        save_path (str): Path to save the plot.
        top_k_nodes_per_ts (int, optional): Number of top nodes to keep per
            timestep. Defaults to 10.
        top_k_edges (int, optional): Number of top edges to keep overall.
            Defaults to 50.
        num_inference_steps (int, optional): Number of diffusion timesteps.
            Defaults to 25.
    """
    # Filter to get the top-k most important nodes per timestep
    filtered_nodes = defaultdict(list)
    node_effects = {}
    all_timesteps = sorted([t for t in timesteps if isinstance(t, int)])

    for t_idx in all_timesteps:
        if t_idx not in nodes or not isinstance(nodes[t_idx], SparseAct):
            continue
        sparse_act = nodes[t_idx]

        top_indices = get_topk_component_indices(
            sparse_act, top_k_nodes_per_ts
        )

        # Get the effects tensor for storing magnitudes
        effects_tensor = sparse_act.to_tensor().mean(
            dim=tuple(range(sparse_act.act.ndim - 1))
        )
        num_features = sparse_act.act.shape[-1]
        for idx in top_indices:
            feature_id = "res" if idx >= num_features else idx
            filtered_nodes[t_idx].append(feature_id)
            node_effects[(t_idx, feature_id)] = effects_tensor[idx].item()

    if "y" in nodes:
        filtered_nodes["y"] = ["probe"]
        node_effects[("y", "probe")] = nodes["y"]

    # Find all edges that connect the filtered nodes
    all_valid_edges = []
    num_features_per_ts = {
        t: nodes[t].act.shape[-1]
        for t in all_timesteps
        if t in nodes and hasattr(nodes[t], "act")
    }

    for (t_up, t_down), edge_tensor in edges.items():
        # Handle sparse JVP tensors (feature x feature)
        if (
            isinstance(edge_tensor, torch.Tensor)
            and edge_tensor.is_sparse
            and edge_tensor._nnz() > 0
        ):
            coalesced = edge_tensor.coalesce()
            down_indices, up_indices = coalesced.indices()
            values = coalesced.values()

            if (
                t_up not in num_features_per_ts
                or t_down not in num_features_per_ts
            ):
                continue
            f_up_count = num_features_per_ts[t_up]
            f_down_count = num_features_per_ts[t_down]

            for i in range(len(values)):
                up_feat_idx, down_feat_idx = (
                    up_indices[i].item(),
                    down_indices[i].item(),
                )
                up_feat = "res" if up_feat_idx >= f_up_count else up_feat_idx
                down_feat = (
                    "res" if down_feat_idx >= f_down_count else down_feat_idx
                )

                if up_feat in filtered_nodes.get(
                    t_up, []
                ) and down_feat in filtered_nodes.get(t_down, []):
                    all_valid_edges.append(
                        (
                            (t_up, up_feat),
                            (t_down, down_feat),
                            values[i].item(),
                        )
                    )

        # Handle dense tensors (edges to probe)
        elif (
            isinstance(edge_tensor, torch.Tensor) and not edge_tensor.is_sparse
        ):
            if t_down == "y":
                if t_up not in num_features_per_ts:
                    continue
                f_up_count = num_features_per_ts[t_up]

                for up_idx, weight in enumerate(edge_tensor.tolist()):
                    up_feat = "res" if up_idx >= f_up_count else up_idx
                    if up_feat in filtered_nodes.get(t_up, []):
                        all_valid_edges.append(
                            ((t_up, up_feat), ("y", "probe"), weight)
                        )

    # Apply a single, global top-k filter to the valid edges
    all_valid_edges.sort(key=lambda x: abs(x[2]), reverse=True)
    filtered_edges = all_valid_edges[:top_k_edges]

    _plot_hierarchical_circuit(
        filtered_nodes,
        node_effects,
        filtered_edges,
        save_path,
        num_inference_steps,
    )


def _plot_hierarchical_circuit(
    filtered_nodes: Dict[Union[int, str], List[Union[int, str]]],
    node_effects: Dict[Tuple[Union[int, str], Union[int, str]], float],
    filtered_edges: List[Tuple[Tuple, Tuple, float]],
    save_path: str,
    num_inference_steps: int = 25,
) -> None:
    """
    Plots the hierarchical circuit.

    Args:
        filtered_nodes (Dict[Union[int, str], List[Union[int, str]]]): Nodes to
            plot.
        node_effects (Dict[Tuple[Union[int, str], Union[int, str]], float]):
            Node effect magnitudes.
        filtered_edges (List[Tuple[Tuple, Tuple, float]]): Edges to plot.
        save_path (str): Path to save the plot.
        num_inference_steps (int, optional): Number of diffusion timesteps.
            Defaults to 25.
    """
    dot = Digraph(name="Causal Circuit")
    dot.attr(
        rankdir="LR",
        ranksep="4.0",
        nodesep="0.5",
        splines="ortho",  # "curved",
        compound="true",
    )
    dot.node_attr.update(
        shape="circle",
        style="filled",
        fixedsize="true",
        width="1.2",
        height="1.2",
        fontname="Helvetica-Bold",
        fontsize="20",
        # bold:
    )

    sorted_timesteps = sorted(
        [k for k in filtered_nodes.keys() if isinstance(k, int)]
    )
    all_features_set = set()
    for t_idx in sorted_timesteps:
        for feat in filtered_nodes.get(t_idx, []):
            all_features_set.add(feat)

    sorted_features = sorted(
        list(all_features_set), key=lambda x: (isinstance(x, str), x)
    )
    # node_effect_values = [v for k, v in node_effects.items() if k[0] != "y"]
    # max_abs_effect = (
    #     max(abs(v) for v in node_effect_values) if node_effect_values else
    # 1.0
    # )
    # norm = mcolors.Normalize(vmin=-max_abs_effect, vmax=max_abs_effect)
    node_effect_values = [v for k, v in node_effects.items() if k[0] != "y"]
    if node_effect_values:
        # Calculate the 95th percentile of the absolute effects to cap the
        # color scale
        cap_value = np.percentile([abs(v) for v in node_effect_values], 95)
        # Use the cap_value, ensuring it's not zero to avoid division errors
        max_abs_effect = cap_value if cap_value > 0 else 1.0
    else:
        max_abs_effect = 1.0

    norm = mcolors.Normalize(vmin=-max_abs_effect, vmax=max_abs_effect)
    cmap = plt.get_cmap("RdBu_r")

    for t_idx in sorted_timesteps:
        label = f"t={1.0 - (t_idx / (num_inference_steps - 1)):.2f}"
        with dot.subgraph(name=f"cluster_{t_idx}") as cluster:
            cluster.attr(
                label=label,
                style="dashed",
                color="gray",
                fontsize="24",
                fontname="Helvetica-Bold",
            )
            for feat_idx in sorted_features:
                node_name = f"t{t_idx}_f{feat_idx}"
                if feat_idx in filtered_nodes.get(t_idx, []):
                    effect = node_effects.get((t_idx, feat_idx), 0.0)
                    rgba = cmap(norm(effect))
                    color = mcolors.to_hex(rgba)
                    text_color = (
                        "#000000" if (0.2 < norm(effect) < 0.8) else "#ffffff"
                    )
                    node_label = "Res" if feat_idx == "res" else f"F{feat_idx}"
                    shape = "ellipse" if feat_idx == "res" else "circle"
                    cluster.node(
                        node_name,
                        label=node_label,
                        fillcolor=color,
                        fontcolor=text_color,
                        shape=shape,
                    )
                else:
                    cluster.node(node_name, style="invis")

    for feat_idx in sorted_features:
        for i in range(len(sorted_timesteps) - 1):
            source_node = f"t{sorted_timesteps[i]}_f{feat_idx}"
            dest_node = f"t{sorted_timesteps[i+1]}_f{feat_idx}"
            dot.edge(source_node, dest_node, style="invis", weight="10000")

    if "y" in filtered_nodes:
        with dot.subgraph(name="cluster_y") as cluster:
            cluster.attr(
                label="",  # label,
                style="dashed",
                color="gray",
                fontsize="24",
                fontname="Helvetica-Bold",
            )
            cluster.node(
                "probe_node",
                label="Probe",
                shape="doublecircle",
                fillcolor="lightgrey",
                fontcolor="black",
            )

    if filtered_edges:
        max_edge_weight = max(abs(e[2]) for e in filtered_edges) or 1.0
        for up_node, down_node, weight in filtered_edges:
            source = f"t{up_node[0]}_f{up_node[1]}"
            target = (
                "probe_node"
                if down_node[0] == "y"
                else f"t{down_node[0]}_f{down_node[1]}"
            )
            normalized_weight = abs(weight) / max_edge_weight
            penwidth = str(0.8 + 3.0 * normalized_weight)
            color = "darkblue" if weight < 0 else "firebrick"
            dot.edge(
                source,
                target,
                penwidth=penwidth,
                color=color,
                constraint="false",
            )

    # if sorted_timesteps and "y" in filtered_nodes:
    #     last_node_name = f"t{sorted_timesteps[-1]}_f{sorted_features[0]}"
    #     dot.edge(last_node_name, "probe_node", style="invis", weight="10000")

    if sorted_timesteps and "y" in filtered_nodes:
        # Connect to middle feature for vertical centering
        middle_feat_idx = len(sorted_features) // 2
        last_node_name = (
            f"t{sorted_timesteps[-1]}_f{sorted_features[middle_feat_idx]}"
        )
        dot.edge(last_node_name, "probe_node", style="invis", weight="10000")

    output_dir = os.path.dirname(save_path)
    os.makedirs(output_dir, exist_ok=True)
    png_data = dot.pipe(format="png")
    img = plt.imread(io.BytesIO(png_data))
    fig, ax = plt.subplots(figsize=(20, 28), dpi=300)
    ax.imshow(img)
    ax.axis("off")

    final_save_path = f"{os.path.splitext(save_path)[0]}.png"
    plt.savefig(final_save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Circuit plot saved to {final_save_path}")


# =========================================================================== #
#                           Edge and Node Stats                               #
# =========================================================================== #


def _setup_plot_style() -> None:
    """
    Sets a professional style for Matplotlib plots.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            # "font.family": "serif",
            # "font.serif": ["Times New Roman"],
            "font.size": 20,
            "axes.labelsize": 20,
            "axes.labelweight": "bold",
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "axes.titlesize": 22,
            "axes.titleweight": "bold",
            "figure.dpi": 300,
        }
    )


def _create_single_boxplot(
    ax: plt.Axes,
    data: List[np.ndarray],
    labels: List[str],
    title: str,
    xlabel: str,
    ylabel: str,
    color: str,
) -> None:
    """
    Helper function to create and style a single boxplot on a given axis.

    Args:
        ax (plt.Axes): The axis to plot on.
        data (List[np.ndarray]): List of data arrays for each box.
        labels (List[str]): Labels for each box.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        color (str): Color for the boxes.
    """
    boxprops = dict(
        facecolor=color, alpha=0.8, edgecolor="black", linewidth=1.5
    )
    medianprops = dict(color="black", linewidth=2.5)
    whiskerprops = dict(color="black", linewidth=1.5)
    capprops = dict(color="black", linewidth=1.5)
    flierprops = dict(
        marker="o",
        markerfacecolor="black",
        markersize=4,
        linestyle="none",
        alpha=0.8,
        markeredgecolor="none",
    )

    ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        boxprops=boxprops,
        medianprops=medianprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        flierprops=flierprops,
    )

    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")

    # Only draw the major grid lines (at powers of 10)
    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.7,
        color="grey",
        alpha=0.5,
    )

    ax.spines[["top", "right"]].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def plot_node_edge_distributions(
    nodes: Dict[Any, "SparseAct"],
    edges: Dict[Any, torch.Tensor],
    num_inference_steps: int,
    base_save_path: Path,
    output_modes: List[
        Literal["combined", "nodes_only", "edges_only"]
    ] = [  # noqa:B006
        "combined",
        "nodes_only",
        "edges_only",
    ],
) -> None:
    """
    Creates and saves beautiful, paper-ready boxplots for node and edge
    effect distributions. Generates multiple files as specified.

    Args:
        nodes (Dict[Any, SparseAct]): Dictionary mapping timesteps to
            SparseAct objects.
        edges (Dict[Any, torch.Tensor]): Dictionary mapping (t_up, t_down)
            tuples to edge tensors.
        num_inference_steps (int): Number of diffusion inference steps.
        base_save_path (Path): Base path to save the plots. Different suffixe
            will be added based on the plot type.
        output_modes (List[Literal["combined", "nodes_only", "edges_only"]],
            optional): List of plot types to generate. Defaults to all three.
    """
    _setup_plot_style()
    NODE_COLOR = "#4682B4"  # Steel Blue
    EDGE_COLOR = "#B22222"  # Firebrick

    # Process Node Data
    node_data = {}
    for t, sparse_act in nodes.items():
        if not isinstance(t, int):
            continue
        values = sparse_act.to_tensor().abs().flatten().cpu().numpy()
        values = values[values > 1e-12]
        if len(values) > 0:
            diffusion_time = 1.0 - (t / (num_inference_steps - 1))
            node_data[diffusion_time] = values

    sorted_node_times = sorted(node_data.keys(), reverse=True)
    node_plot_data = [node_data[t] for t in sorted_node_times]
    node_labels = [f"{t:.2f}" for t in sorted_node_times]

    # Process Edge Data
    edge_data = {}
    for (t_up, t_down), tensor in edges.items():
        if not isinstance(t_down, int):
            continue
        values = tensor.coalesce().values().abs().cpu().numpy()
        values = values[values > 1e-9]
        if len(values) > 0:
            dt_up = 1.0 - (t_up / (num_inference_steps - 1))
            dt_down = 1.0 - (t_down / (num_inference_steps - 1))
            label = f"{dt_up:.2f}→{dt_down:.2f}"
            edge_data[label] = values

    # Sort labels
    sorted_edge_labels = sorted(edge_data.keys(), reverse=True)
    edge_plot_data = [edge_data[label] for label in sorted_edge_labels]

    if "combined" in output_modes:
        fig, axes = plt.subplots(1, 2, figsize=(22, 9), sharey=True)
        _create_single_boxplot(
            axes[0],
            node_plot_data,
            node_labels,
            "Node Effect Distribution",
            "Diffusion Time (t)",
            "Absolute Effect Magnitude (log scale)",
            NODE_COLOR,
        )
        _create_single_boxplot(
            axes[1],
            edge_plot_data,
            sorted_edge_labels,
            "Edge Effect Distribution",
            "Diffusion Time Transition ($t \\rightarrow t - \\Delta t$)",
            "",
            EDGE_COLOR,
        )
        fig.tight_layout(pad=2.5)
        save_path = base_save_path.with_name(
            f"{base_save_path.stem}_combined"
        ).with_suffix(base_save_path.suffix)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved combined distribution plot to {save_path}")

    if "nodes_only" in output_modes:
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        _create_single_boxplot(
            ax,
            node_plot_data,
            node_labels,
            "",
            "Diffusion Time (t)",
            "Absolute Effect Magnitude (log scale)",
            NODE_COLOR,
        )
        fig.tight_layout(pad=2.5)
        save_path = base_save_path.with_name(
            f"{base_save_path.stem}_nodes"
        ).with_suffix(base_save_path.suffix)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved nodes-only distribution plot to {save_path}")

    if "edges_only" in output_modes:
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        _create_single_boxplot(
            ax,
            edge_plot_data,
            sorted_edge_labels,
            "",
            "Diffusion Time Transition ($t \\rightarrow t - \\Delta t$)",
            "Absolute Effect Magnitude (log scale)",
            EDGE_COLOR,
        )
        fig.tight_layout(pad=2.5)
        save_path = base_save_path.with_name(
            f"{base_save_path.stem}_edges"
        ).with_suffix(base_save_path.suffix)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved edges-only distribution plot to {save_path}")
