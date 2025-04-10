"""
Configuration file for extracting latent activations from SDXL multi-step
diffusion model and training sparse autoencoders on them.

Source/ adapted from:
    https://github.com/JHSchlegel/SAeUron/blob/main/SAE/config.py
Adaptations made:
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Union

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
from simple_parsing import Serializable

# TODO: implement train and test split logic


# =========================================================================== #
#                           SAE Configuration                                 #
# =========================================================================== #
@dataclass
class VanillaSAEConfig(Serializable):
    pass


# =========================================================================== #
#                          Latents Extraction Configuration                   #
# =========================================================================== #
@dataclass
class LatentsExtractionConfig(Serializable):
    hook_names: Union[List[str], str, None] = field(
        default_factory=lambda: ["unet.down_blocks.2.attentions.0"]
    )
    """List of model layers from which to extract the activations."""

    extracted_latents_path: Union[str, None] = None
    """Where to save the extracted latent activations."""

    dataset_name: str = "flickr30k"
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

    model_name: str = "stabilityai/stable-diffusion-2-1"
    """
    Name of huggingface model to use for extracting the latent activations.
    """

    dtype: str = "float16"
    """Data type to use for extracting the latent activations. One of
    ['float16', 'float32']
    """

    num_inference_steps: int = 25
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

    guidance_scale: float = 9.0
    """Scale for classifier-free guidance during diffusion process."""

    height: int = 512
    """Height of the generated images."""

    width: int = 512
    """Width of the generated images."""

    output_or_diff: str = "diff"
    """
    Whether to also save the input or just the output latent representation
    """

    unconditional: bool = False
    """
    Whether to extract unconditional latents or conditional latents whenever
    guidance_scale > 1.0
    """

    def __post_init__(self):
        if isinstance(self.hook_names, str):
            self.hook_names = [self.hook_names]

        if self.extracted_latents_path is None:
            self.extracted_latents_path = os.path.join(
                "../../../activations",
                self.model_name.split("/")[-1],
                self.dataset_name.split("/")[-1],
                self.dataset_split,
                f"subset_size-{str(self.dataset_size)}",
                f"{self.num_inference_steps}-inference-steps",
                f"every-{self.extract_every_n_timesteps}-steps",
            )
