"""
Main script for sparse feature circuit discovery in diffusion models.
"""

# =========================================================================== #
#                             Packages and Presets                            #
# =========================================================================== #


import datetime
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal

import torch
from datasets import load_from_disk
from safetensors.torch import load_file
from simple_parsing import Serializable, parse
from tqdm import tqdm

# Ensure project root is in path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from circuit_plotting import plot_causal_circuit, plot_node_edge_distributions
from circuit_utils import clear_memory_cache, get_topk_component_indices
from config import TopKSAEConfig
from core.diffusion.hooked_sd_pipeline import HookedStableDiffusionPipeline
from core.sae.topk_sae import TopKSAE
from core.utils.reproducibility import set_all_seeds
from probe import LatentProbe
from temporal_attribution import (
    compute_edges_to_probe,
    compute_node_effects,
    temporal_jvp_aggregated,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =========================================================================== #
#                              Config Definition                              #
# =========================================================================== #


@dataclass
class CircuitDiscoveryConfig(Serializable):
    """Configuration for large-scale circuit discovery."""

    # -------------------------------------------------------------------------
    # Path Configuration
    # -------------------------------------------------------------------------
    sae_path: str = (
        "../../checkpoints/sae/down_blocks.2.attentions.0/TopKSAE_dsae-5120_timesteps-all_20250816_083716/step_488282"  # noqa: E501
    )
    """Path to the SAE checkpoint."""

    probe_path: str = (
        "../../checkpoints/probe/cnn_birds_vs_cats_20250620_165530.safetensors"
    )
    """Path to the probe checkpoint."""

    dataset_path: str = (
        "../../data/prompts/circuit_discovery/birds_vs_cats/comparative_pairs/"
    )
    """Path to the dataset of prompts."""

    output_dir: (
        str
    ) = "../../results/circuits/birds_vs_cats/" + datetime.datetime.now().strftime(  # noqa: E501
        "%Y%m%d_%H%M%S"
    )
    """Directory to save results to."""

    # -------------------------------------------------------------------------
    # Discovery Configuration
    # -------------------------------------------------------------------------
    hook_name: str = "unet.down_blocks.2.attentions.0"
    """Name of cross attention layer to hook for feature activations."""

    timesteps: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    """Timesteps to analyze for circuit discovery."""

    probe_timestep: int = 4
    """Timestep at which to apply the probe."""

    num_prompts: int = 4
    """Number of prompts to use for circuit discovery."""

    num_seeds: int = 1
    """Number of random seeds to use for circuit discovery."""

    top_k_jvp_nodes: int = 10
    """Number of top K JVP nodes to consider."""

    node_method: Literal["grad", "ig"] = "ig"
    """
    Method to use for node attribution (gradient or integrated gradients).
    """

    ig_steps: int = 5
    """Number of steps to use for integrated gradients."""

    # -------------------------------------------------------------------------
    # Model & Diffusion Configuration
    # -------------------------------------------------------------------------
    model_id: str = "stabilityai/stable-diffusion-2-1"
    """HuggingFace model ID for Stable Diffusion."""

    num_inference_steps: int = 25
    """Number of inference steps to use for image generation."""

    guidance_scale: float = 9.0
    """Scale for classifier-free guidance."""

    height: int = 512
    """Height of the generated images."""

    width: int = 512
    """Width of the generated images."""

    torch_dtype: str = "float32"
    """Torch dtype to use (e.g., 'float32', 'float16')."""

    device: str = "cpu"
    """Device to run the computations on (e.g., 'cpu', 'cuda')."""

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    object_a: str = "bird"
    """Concept A in comparative prompts."""

    object_b: str = "cat"
    """Concept B in comparative prompts."""

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# =========================================================================== #
#                              Core Discovery Logic                           #
# =========================================================================== #


def get_aggregated_circuit(
    model: HookedStableDiffusionPipeline,
    sae: TopKSAE,
    probe: LatentProbe,
    prompts: list,
    config: CircuitDiscoveryConfig,
) -> tuple[dict, dict]:
    """
    Runs circuit discovery over all prompts and seeds, and aggregates the
    results.

    Args:
        model (HookedStableDiffusionPipeline): The diffusion model pipeline.
        sae (TopKSAE): The sparse autoencoder for reconstruction.
        probe (LatentProbe): The probe for classifying.
        prompts (list): List of prompts to analyze.
        config (CircuitDiscoveryConfig): Configuration for circuit discovery.

    Returns:
        tuple[dict, dict]: Aggregated node and edge data.
    """
    running_nodes = defaultdict(lambda: 0.0)
    running_edges = defaultdict(lambda: 0.0)
    total_runs = 0
    pbar = tqdm(
        total=len(prompts) * config.num_seeds, desc="Aggregating Circuits"
    )

    for example in prompts:
        clean_prompt = example["clean_prompt"]
        patch_prompt = example["patch_prompt"]

        patch_answer = example["patch_answer"]

        if patch_answer == config.object_b:  # Target is "cat"
            # We want to maximize the "cat" score
            current_metric_fn = lambda x: probe(x)  # noqa: E731
        elif patch_answer == config.object_a:  # Target is "bird"
            # We want to maximize the "bird" score
            current_metric_fn = lambda x: -probe(x)  # noqa: E731
        else:
            raise ValueError(f"Unknown patch_answer: {patch_answer}")

        for seed in range(config.num_seeds):
            total_runs += 1
            set_all_seeds(seed)
            logger.info(
                f"\n--- Running for prompt: '{clean_prompt[:50]}...' "
                f"with seed: {seed} ---"
            )

            node_results = compute_node_effects(
                model=model,
                sae=sae,
                hook_name=config.hook_name,
                timesteps_to_analyze=config.timesteps
                + [config.probe_timestep],
                clean_prompt=clean_prompt,
                dest_prompt=patch_prompt,
                method=config.node_method,
                probe_timestep=config.probe_timestep,
                metric_fn=current_metric_fn,
                device=config.device,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                height=config.height,
                width=config.width,
                include_residuals=True,
                ig_steps=config.ig_steps,
            )
            edges_to_probe = compute_edges_to_probe(
                node_results.grads, node_results.deltas, config.probe_timestep
            )
            current_run_edges = {}
            sorted_timesteps = sorted(config.timesteps)

            for i in range(len(sorted_timesteps) - 1):
                t_up, t_down = sorted_timesteps[i], sorted_timesteps[i + 1]
                downstream_feature_indices = get_topk_component_indices(
                    node_results.nodes[t_down], k=config.top_k_jvp_nodes
                )
                if not downstream_feature_indices:
                    logger.warning(
                        f"No important nodes for edge {t_up}->{t_down}. Skip."
                    )
                    continue
                aggregated_edge_tensor = temporal_jvp_aggregated(
                    model=model,
                    sae=sae,
                    hook_name=config.hook_name,
                    t_upstream=t_up,
                    t_downstream=t_down,
                    downstream_feature_indices=downstream_feature_indices,
                    left_vec=node_results.grads[t_down],
                    right_vec=node_results.deltas[t_up],
                    prompt=clean_prompt,
                    device=config.device,
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale,
                    height=config.height,
                    width=config.width,
                )
                current_run_edges[(t_up, t_down)] = aggregated_edge_tensor

            # Aggregate nodes
            for t, node_val in node_results.nodes.items():
                running_nodes[t] = running_nodes[t] + node_val.to("cpu")
            running_nodes["y"] = running_nodes["y"] + torch.tensor(
                node_results.total_effect or 0.0, device="cpu"
            )

            # Aggregate inter-timestep edges
            for edge, edge_val in current_run_edges.items():
                # handle sparse and dense tensors correctly
                current_val_cpu = edge_val.to("cpu")
                if isinstance(
                    running_edges[edge], float
                ):  # It's the initial 0.0
                    running_edges[edge] = current_val_cpu
                elif current_val_cpu.is_sparse:
                    running_edges[edge] = running_edges[edge] + current_val_cpu
                else:  # The accumulator is sparse but the new value is dense
                    running_edges[edge] = current_val_cpu + running_edges[edge]

            # Aggregate edge to probe (this is always dense)
            edge_to_probe_agg = edges_to_probe.to_tensor().mean(dim=(0, 1, 2))
            running_edges[(config.probe_timestep, "y")] = running_edges[
                (config.probe_timestep, "y")
            ] + edge_to_probe_agg.to("cpu")

            pbar.update(1)
            del node_results, edges_to_probe, current_run_edges
            clear_memory_cache(config.device)

    pbar.close()

    logger.info(f"Normalizing aggregated results over {total_runs} runs...")
    final_nodes = {t: val / total_runs for t, val in running_nodes.items()}
    final_edges = {
        edge: val / total_runs for edge, val in running_edges.items()
    }

    return final_nodes, final_edges


# =========================================================================== #
#                                 Main Execution                              #
# =========================================================================== #

if __name__ == "__main__":
    config = parse(CircuitDiscoveryConfig)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    # save config:
    config.save_json(os.path.join(config.output_dir, "config.json"), indent=4)
    logger.info(f"Running with configuration:\n{config.dumps_yaml()}")

    torch_dtype = getattr(torch, config.torch_dtype)

    logger.info("Loading models...")
    pipe = HookedStableDiffusionPipeline.from_pretrained(
        config.model_id, torch_dtype=torch_dtype, use_safetensors=True
    ).to(device=config.device, torch_dtype=torch_dtype)
    pipe.set_progress_bar_config(disable=True)

    sae = TopKSAE.load_from_disk(
        config.sae_path, config_class=TopKSAEConfig, device=config.device
    )
    sae = sae.to(dtype=torch_dtype).eval()

    probe = LatentProbe(
        n_latent_channels=sae.cfg.d_in,
        probe_type="cnn",
        spatial_resolution=(16, 16),
    )
    probe.load_state_dict(load_file(config.probe_path, device=config.device))
    probe = probe.to(device=config.device, dtype=torch_dtype).eval()
    logger.info("Models loaded successfully.")

    logger.info(f"Loading dataset from {config.dataset_path}")
    dataset = load_from_disk(config.dataset_path)["train"]
    prompts_to_run = list(
        dataset.shuffle(seed=42).select(range(config.num_prompts))
    )

    logger.info(
        f"Starting circuit discovery for {config.num_prompts} prompts "
        f"and {config.num_seeds} seeds..."
    )
    final_nodes, final_edges = get_aggregated_circuit(
        pipe, sae, probe, prompts_to_run, config
    )

    circuit_filename = f"circuit_p{config.num_prompts}_s{config.num_seeds}.pt"
    save_path = os.path.join(config.output_dir, circuit_filename)
    logger.info(f"Saving aggregated circuit to {save_path}")

    data_to_save = {
        "nodes": {k: v for k, v in final_nodes.items()},
        "edges": {k: v for k, v in final_edges.items()},
        "config": config.to_dict(),
    }
    torch.save(data_to_save, save_path)
    logger.info("Circuit discovery and saving complete.")

    plot_causal_circuit(
        nodes=final_nodes,
        edges=final_edges,
        timesteps=config.timesteps + [config.probe_timestep],
        save_path=Path(config.output_dir)
        / f"circuit_p{config.num_prompts}_s{config.num_seeds}.png",
        top_k_nodes_per_ts=10,
        top_k_edges=50,
        num_inference_steps=config.num_inference_steps,
    )

    dist_plot_path = (
        Path(config.output_dir)
        / f"circuit_dist_p{config.num_prompts}_s{config.num_seeds}.png"
    )
    plot_node_edge_distributions(
        nodes=final_nodes,
        edges=final_edges,
        num_inference_steps=config.num_inference_steps,
        base_save_path=dist_plot_path,
    )
    logger.info("All plotting complete.")
