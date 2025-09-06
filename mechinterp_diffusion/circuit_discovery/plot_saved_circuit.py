"""
Script to load a pre-computed and aggregated circuit from a file and plot it
using a configuration-driven approach.
"""

# =========================================================================== #
#                             Packages and Presets                            #
# =========================================================================== #
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from simple_parsing import Serializable, parse

sys.path.append(str(Path(__file__).resolve().parent.parent))

from circuit_plotting import plot_causal_circuit, plot_node_edge_distributions

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =========================================================================== #
#                              Config Definition                              #
# =========================================================================== #


@dataclass
class PlottingConfig(Serializable):
    """Configuration for plotting a saved circuit."""

    circuit_path: str = (
        "../../results/circuits/birds_vs_cats/20250905_055102/circuit_p50_s2.pt"
    )
    """Path to the saved circuit .pt file."""

    output_dir: str = "plots"
    """Directory to save the circuit plot."""

    top_k_nodes: int = 7
    """Number of top nodes to display per timestep."""
    top_k_edges_per_ts: int = 7
    """Total number of top edges to display in the plot."""

    num_inference_steps: int = 25
    """Number of diffusion inference steps (used for distribution plots)."""

    def __post_init__(self):
        Path(self.output_dir).mkdir(exist_ok=True)


# =========================================================================== #
#                                 Main Execution                              #
# =========================================================================== #

if __name__ == "__main__":
    config = parse(PlottingConfig)
    logger.info(f"Running with configuration:\n{config.dumps_yaml()}")

    circuit_name = Path(config.circuit_path).stem
    circuit_save_path = os.path.join(
        config.output_dir, f"{circuit_name}_plot.png"
    )

    if not os.path.exists(config.circuit_path):
        logger.error(f"Circuit file not found at: {config.circuit_path}")
        sys.exit(1)

    logger.info(f"Loading circuit from {config.circuit_path}")
    circuit_data = torch.load(
        config.circuit_path, map_location="cpu", weights_only=False
    )
    nodes = circuit_data["nodes"]
    edges = circuit_data["edges"]
    metadata_config = circuit_data.get("config", {})

    logger.info("Circuit data loaded successfully.")

    # Extract timesteps from metadata for plotting
    timesteps_analyzed = metadata_config.get("timesteps", [])
    probe_timestep = metadata_config.get("probe_timestep", None)
    if probe_timestep and probe_timestep not in timesteps_analyzed:
        timesteps_analyzed.append(probe_timestep)

    if not timesteps_analyzed:
        logger.warning(
            "Could not determine timesteps from metadata. "
            "Inferring from node keys."
        )
        timesteps_analyzed = sorted(
            [k for k in nodes.keys() if isinstance(k, int)]
        )

    plot_causal_circuit(
        nodes=nodes,
        edges=edges,
        timesteps=timesteps_analyzed,
        save_path=circuit_save_path,
        top_k_nodes_per_ts=config.top_k_nodes,
        top_k_edges_per_ts=config.top_k_edges_per_ts,
        num_inference_steps=metadata_config.get("num_inference_steps", "N/A"),
    )

    logger.info("Plot saved successfully.")

    dist_plot_path = Path(config.output_dir) / "circuit_distributions.png"
    plot_node_edge_distributions(
        nodes=nodes,
        edges=edges,
        num_inference_steps=config.num_inference_steps,
        base_save_path=dist_plot_path,
    )
