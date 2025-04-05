"""
Configuration file for extracting latent activations from SDXL multi-step
diffusion model and training sparse autoencoders on them.

Source/ adapted from:
    https://github.com/JHSchlegel/SAeUron/blob/main/SAE/config.py
Adaptations made:
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Union

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import torch
from simple_parsing import Serializable

# TODO: implement train and test split logic


@dataclass
class VanillaSAEConfig(Serializable):
    pass


@dataclass
class LatentsExtractionConfig(Serializable):
    hook_names: Union[List[str], None] = "unet.up_blocks.0.attentions.1"
    """List of model layers from which to extract the activations."""

    extracted_latents_path: Union[str, None] = None
    """Where to save the extracted latent activations."""

    dataset_name: str = "laion"
    """
    Name of huggingface prompt dataset to use for extracting the latent
    activations. Must be one of ['laion', 'flickr30k']. For 'laion', the
    guangyil/laion-coco-aestheticlaion-coco-aesthetic dataset is used.
    For 'flickr30k', the flickr30k dataset is used.
    """

    dataset_split: str = "train"
    """
    Split of the dataset to use for extracting the latent activations. One
    of ['train', 'test']
    """

    dataset_size: Optional[int] = 50_000
    """
    Number of prompts to use for extracting the latent activations. If None,
    the entire dataset will be used.
    """

    column_name: str = "caption"
    """Name of column in the dataset that includes the prompts."""

    model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    """
    Name of huggingface model to use for extracting the latent activations.
    """

    device: Union[torch.device, str] = "cuda"
    """Device to use for extracting the latent activations."""

    dtype: torch.dtype = torch.float16
    """Data type to use for extracting the latent activations."""

    num_inference_steps: int = 50
    """Number of diffusion inference steps during latents extraction."""

    seed: int = 42
    """Random seed for reproducibility."""

    batch_size: int = 64
    """Number of prompts to process in parallel during latents extraction."""

    extract_every_n_timesteps: int = 1
    """
    How frequently to save the extracted latent activations during diffusion
    process.
    """

    guidance_scale: float = 0.0
    """Scale for classifier-free guidance during diffusion process."""

    output_or_diff: str = "diff"
    """
    Whether to also save the input or just the output latent representation
    """

    def __post_init__(self):
        if isinstance(self.hook_names, str):
            self.hook_names = [self.hook_names]

        if self.extracted_latents_path is None:
            self.extracted_latents_path = os.path.join(
                "activations",
                self.model_name.split("/")[-1],
                self.dataset_name.split("/")[-1],
                self.dataset_split,
                str(self.dataset_size),
            )
