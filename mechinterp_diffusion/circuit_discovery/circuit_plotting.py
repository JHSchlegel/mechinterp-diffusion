"""
Module for visualizing causal circuits in diffusion models.
"""

# =========================================================================== #
#                              Packages and Presets                           #
# =========================================================================== #

import io
import logging
import os
from typing import Dict, List, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch
from activation_utils import SparseAct
from graphviz import Digraph

logger = logging.getLogger(__name__)

# Constants for visualization
NODE_SIZE_MIN = 0.3
NODE_SIZE_MAX = 1.5
EDGE_WIDTH_MIN = 0.5
EDGE_WIDTH_MAX = 3.0
FONT_SIZE = 10
TITLE_FONT_SIZE = 14


# TODO: Add metric as final node


# =========================================================================== #
#                             Main Plotting Function                          #
# =========================================================================== #
def plot_causal_circuit(
    nodes: Dict[int, SparseAct],
    edges: Dict[Tuple[int, int], Dict],
    timesteps: List[int],
    save_path: str,
    node_threshold: float = 1e-3,
    edge_threshold: float = 1e-4,
    top_k_nodes: int = 20,
    top_k_edges: int = 50,
    num_inference_steps: int = 25,
) -> None:
    """
    Plot the causal circuit showing nodes and edges (temporal dependencies).

    Args:
        nodes (Dict[int, SparseAct]): Dictionary mapping timestep ->
            SparseAct of feature effects
        edges (Dict[Tuple[int, int], Dict]): Dictionary mapping
            (t_early, t_late) -> edge information
        timesteps (List[int]): List of timestep indices
        save_path (str): Path to save the plot
        node_threshold (float, optional): Minimum effect for including a node.
            Defaults to 1e-3.
        edge_threshold (float, optional): Minimum weight for including an edge.
            Defaults to 1e-4.
        top_k_nodes (int, optional): Maximum nodes per timestep to show.
            Defaults to 20.
        top_k_edges (int, optional): Maximum total edges to show. Defaults
            to 50.
        num_inference_steps (int, optional): Number of diffusion timesteps.
            Defaults to 25.
    """

    # ilter and select important nodes
    filtered_nodes = {}
    node_effects = {}  # Store effect magnitudes for sizing

    for t_idx in timesteps:
        if t_idx not in nodes:
            continue

        sparse_act = nodes[t_idx]
        feature_effects = sparse_act.act

        # Handle different tensor shapes
        if feature_effects.ndim > 1:
            feature_effects = feature_effects.mean(
                dim=tuple(range(feature_effects.ndim - 1))
            )

        abs_effects = feature_effects.abs()
        signed_effects = feature_effects

        # Get top-k features above threshold
        mask = abs_effects > node_threshold
        active_indices = torch.where(mask)[0]

        if len(active_indices) > top_k_nodes:
            top_vals, top_indices = torch.topk(abs_effects, top_k_nodes)
            active_indices = top_indices[top_vals > node_threshold]

        filtered_nodes[t_idx] = []
        for feat_idx in active_indices.tolist():
            effect = signed_effects[feat_idx].item()
            filtered_nodes[t_idx].append(feat_idx)
            node_effects[(t_idx, feat_idx)] = effect

    # Step 2: Filter edges based on importance
    filtered_edges = []

    for (t_early, t_late), edge_info in edges.items():
        weight_matrix = edge_info["weight_matrix"]
        early_features = edge_info["early_features"]
        late_features = edge_info["late_features"]

        # Find significant edges
        for i, early_feat in enumerate(early_features.tolist()):
            if early_feat not in filtered_nodes.get(t_early, []):
                continue

            for j, late_feat in enumerate(late_features.tolist()):
                if late_feat not in filtered_nodes.get(t_late, []):
                    continue

                weight = abs(weight_matrix[j, i].item())
                if weight > edge_threshold:
                    filtered_edges.append(
                        ((t_early, early_feat), (t_late, late_feat), weight)
                    )

    # Sort and limit edges
    filtered_edges.sort(key=lambda x: x[2], reverse=True)
    if len(filtered_edges) > top_k_edges:
        logger.info(
            f"Limiting edges from {len(filtered_edges)} to {top_k_edges}"
        )
        filtered_edges = filtered_edges[:top_k_edges]

    # Create the graph
    _plot_hierarchical_circuit(
        filtered_nodes,
        node_effects,
        filtered_edges,
        save_path,
        num_inference_steps,
    )


# =========================================================================== #
#                              Helper Functions                               #
# =========================================================================== #
def _plot_hierarchical_circuit(
    filtered_nodes: Dict[int, List[int]],
    node_effects: Dict[Tuple[int, int], float],
    filtered_edges: List[Tuple[Tuple[int, int], Tuple[int, int], float]],
    save_path: str,
    num_inference_steps: int = 25,
) -> None:
    """Plot the hierarchical causal circuit.

    Args:
        filtered_nodes (Dict[int, List[int]]): Nodes exceeding the threshold.
        node_effects (Dict[Tuple[int, int], float]): Node effect magnitudes.
        filtered_edges (List[Tuple[Tuple[int, int], Tuple[int, int], float]]):
            Edges exceeding the threshold.
        save_path (str): Path to save the plot.
        num_inference_steps (int, optional): Number idffuions timestesp.
            Defaults to 25.
    """

    def convert_timestep_to_diffusion_time(t_idx):
        return 1.0 - (t_idx / (num_inference_steps - 1))

    dot = Digraph(name="Causal Circuit")
    dot.attr(
        rankdir="LR",
        ranksep="4.0",
        nodesep="0.5",
        splines="curved",
        compound="true",
        concentrate="true",
    )
    dot.node_attr.update(
        shape="circle",
        style="filled",
        fixedsize="true",
        width="1.2",
        height="1.2",
        fontname="Helvetica",
        fontsize="11",
    )

    sorted_timesteps = sorted(list(filtered_nodes.keys()))

    all_features_set = set(
        feat for t in sorted_timesteps for feat in filtered_nodes.get(t, [])
    )
    sorted_features = sorted(list(all_features_set))

    # Use absolute max for a symmetrical color scale around zero
    max_abs_effect = (
        max(abs(v) for v in node_effects.values()) if node_effects else 1.0
    )
    norm = mcolors.Normalize(vmin=-max_abs_effect, vmax=max_abs_effect)
    cmap = plt.get_cmap("RdBu_r")  # Red-White-Blue colormap

    # Full grid of nodes (including invisible placeholders)
    for t_idx in sorted_timesteps:
        for feat_idx in sorted_features:
            node_name = f"t{t_idx}_f{feat_idx}"
            if feat_idx in filtered_nodes.get(t_idx, []):
                effect = node_effects.get((t_idx, feat_idx), 0.0)
                rgba = cmap(norm(effect))
                color = mcolors.to_hex(rgba)
                text_color = (
                    "#000000" if (0.2 < norm(effect) < 0.8) else "#ffffff"
                )
                dot.node(
                    node_name,
                    label=f"F{feat_idx}",
                    fillcolor=color,
                    fontcolor=text_color,
                )
            else:
                dot.node(node_name, style="invis")

    # Enforce horizontal feature alignment for temporal consistency
    for feat_idx in sorted_features:
        for i in range(len(sorted_timesteps) - 1):
            source_node = f"t{sorted_timesteps[i]}_f{feat_idx}"
            dest_node = f"t{sorted_timesteps[i+1]}_f{feat_idx}"
            dot.edge(source_node, dest_node, style="invis", weight="1000")

    # Draw visible data edges
    max_edge_weight = (
        max(e[2] for e in filtered_edges) if filtered_edges else 1.0
    )
    min_edge_weight = (
        min(e[2] for e in filtered_edges) if filtered_edges else 0.0
    )
    for (t_early, feat_early), (t_late, feat_late), weight in filtered_edges:
        source = f"t{t_early}_f{feat_early}"
        target = f"t{t_late}_f{feat_late}"
        normalized = (
            (weight - min_edge_weight) / (max_edge_weight - min_edge_weight)
            if max_edge_weight > min_edge_weight
            else 0.5
        )
        edge_width = 0.5 + 2.5 * normalized
        dot.edge(
            source,
            target,
            penwidth=str(edge_width),
            color="darkblue",
            constraint="false",
        )

    # Add decorative cluster boxes
    for t_idx in sorted_timesteps:
        diffusion_time = convert_timestep_to_diffusion_time(t_idx)
        time_label = f"t={diffusion_time:.3f}"
        with dot.subgraph(name=f"cluster_t{t_idx}") as cluster:
            cluster.attr(
                label=time_label, style="dashed", color="gray", fontsize="16"
            )
            for feat_idx in sorted_features:
                cluster.node(f"t{t_idx}_f{feat_idx}")

    # Render graph and add colorbar
    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    png_data = dot.pipe(format="png")
    img = plt.imread(io.BytesIO(png_data))

    fig, ax = plt.subplots(figsize=(16, 10), dpi=150)
    ax.imshow(img)
    ax.axis("off")

    cax = fig.add_axes(
        [
            ax.get_position().x1 + 0.01,
            ax.get_position().y0,
            0.02,
            ax.get_position().height,
        ]
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Feature Effect Magnitude", fontsize=12)

    final_save_path = f"{os.path.splitext(save_path)[0]}.png"
    plt.savefig(final_save_path, bbox_inches="tight")
    plt.savefig(final_save_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Circuit plot saved to {final_save_path}")
