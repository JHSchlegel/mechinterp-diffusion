"""
This module contains utility functions for analyzing and plotting
Sparse Autoencoder features and diffusion model data.
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #

from pathlib import Path

# =========================================================================== #
#                            Classes and Functions                            #
# =========================================================================== #


def convert_timestep_to_diffusion_time(
    timestep: int, max_timestep: int = 961
) -> float:
    """
    Convert a discrete timestep to a normalized diffusion time in [0, 1].

    Args:
        timestep (int): The current discrete timestep
        max_timestep (int, optional): The maximum timestep value. Defaults to
            961.

    Returns:
        float: Normalized diffusion time in [0, 1]
    """
    return (timestep - 1) / (max_timestep - 1)


def get_block_label(sae_path: Path) -> str:
    """
    Generates a concise block label from an SAE path.

    Args:
        sae_path (Path): Path to the SAE model directory

    Returns:
        str: Concise block label (e.g., "down.2.0")

    Example:
        ../../checkpoints/sae/down_blocks.2.attentions.0/... -> down.2.0
    """
    block_name = sae_path.parent.parent.name
    parts = block_name.split(".")

    block_type = parts[0].split("_")[0]
    block_num = parts[1]
    attn_num = parts[3]
    return f"{block_type}.{block_num}.{attn_num}"
