"""
This module implements extraction and visualization of top examples for each
SAE feature.

Example usage:
    python extract_top_examples.py \
            --sae_path /path/to/sae/checkpoint \
            --output_dir /path/to/output/dir
"""

# =========================================================================== #
#                           Packages and Presets                              #
# =========================================================================== #


import heapq
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset, load_from_disk
from matplotlib.colors import ListedColormap
from PIL import Image
from simple_parsing import Serializable, parse
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import TopKSAEConfig
from core.diffusion.hooked_sd_pipeline import HookedStableDiffusionPipeline
from core.sae.topk_sae import TopKSAE
from core.utils.reproducibility import set_all_seeds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",  # noqa: E501
)
logger = logging.getLogger(__name__)

DTYPE_MAP = {"float16": torch.float16, "float32": torch.float32}


# =========================================================================== #
#                              Main Function                                  #
# =========================================================================== #
def main() -> None:
    """Main function to extract and visualize top examples for SAE features."""
    args = parse(TopExamplesConfig)
    extractor = TopExamplesExtractor(args)
    extractor.run()


# =========================================================================== #
#                            Configuration Class                              #
# =========================================================================== #
@dataclass
class TopExamplesConfig(Serializable):
    """Configuration for extracting top examples for SAE features."""

    target_modules: List[str] = field(default_factory=list)
    """List of target modules to hook into. Can be specified multiple times"""

    sae_paths: List[str] = field(default_factory=list)
    """List of paths to SAE models. Must match the order of --target_modules"""

    model_id: str = "stabilityai/stable-diffusion-2-1"
    """Hugging Face model ID for the diffusion model."""

    target_module: str = "unet.down_blocks.2.attentions.0"
    """Target module to extract activations from."""

    output_dir: str = "../../results/top_examples"
    """Directory to save the results."""

    dataset_path: str = "../../data/prompts/laion-coco_captions"
    """Path to the dataset containing prompts."""

    prompt_column: str = "caption"
    """Column name containing prompts in the dataset."""

    num_prompts: int = 2_000
    """Number of prompts to process."""

    num_examples_per_feature: int = 9
    """Number of top examples to save per feature."""

    number_of_seeds: int = 5
    """How many different seeds to use for generation."""

    output_or_diff: Literal["diff", "output"] = "diff"
    """Whether to save the output or the difference from the input."""

    torch_dtype: str = "float16"
    """Torch data type: 'float16' or 'float32'."""

    guidance_scale: float = 9.0
    """Guidance scale for classifier-free guidance."""

    num_inference_steps: int = 25
    """Number of inference steps for the diffusion process."""

    height: int = 512
    """Height of the generated images."""

    width: int = 512
    """Width of the generated images."""

    save_individual_examples: bool = True
    """Whether to save individual examples along with grid visualizations."""

    features_to_process: Optional[List[int]] = None
    """Specific feature indices to process (None for all features)."""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """Device to use for computation."""

    batch_size: int = 1
    """Batch size for processing prompts."""

    temporal_aggregation: str = "max"
    """Temporal aggregation method: 'max' or 'mean'."""

    def __post_init__(self) -> None:
        """Post-initialization checks and setup."""
        if len(self.sae_paths) != len(self.target_modules):
            raise ValueError(
                "Number of SAE paths must match number of target modules."
            )

        self.sae_config = dict(
            zip(self.target_modules, self.sae_paths, strict=False)
        )


# =========================================================================== #
#                       Top Examples Extraction Class                         #
# =========================================================================== #
class TopExamplesExtractor:
    """Extract and visualize top examples for each SAE feature."""

    def __init__(self, config: TopExamplesConfig) -> None:
        """Initialize the top examples extractor.

        Args:
            config (TopExamplesConfig): Configuration for extraction.
        """
        self.config = config
        self.device = torch.device(config.device)
        self.torch_dtype = DTYPE_MAP[config.torch_dtype]

        self._setup_output_dirs()

        # ---------------------------------------------------------------------
        # Initialize SAE and Diffusion Pipeline
        # ---------------------------------------------------------------------
        self.saes: Dict[str, TopKSAE] = {}
        for target_module, sae_path in self.config.sae_config.items():
            logger.info(
                f"Loading SAE for module '{target_module}' from {sae_path}"
            )
            sae = TopKSAE.load_from_disk(
                path_str=sae_path,
                config_class=TopKSAEConfig,
                device=config.device,
            ).to(dtype=self.torch_dtype)
            sae.eval()
            self.saes[target_module] = sae

        self.pipe = HookedStableDiffusionPipeline.from_pretrained(
            config.model_id,
            torch_dtype=self.torch_dtype,
            safety_checker=None,
        ).to(self.device)

        self.pipe.scheduler.set_timesteps(
            self.config.num_inference_steps, device="cpu"
        )
        self.scheduler_timesteps = self.pipe.scheduler.timesteps.to(
            self.device
        )

        # ---------------------------------------------------------------------
        # Initialize feature tracking
        # ---------------------------------------------------------------------
        self.features_to_process: Dict[str, List[int]] = {}
        self.feature_examples: Dict[str, Dict[int, List]] = {}
        for target_module, sae in self.saes.items():
            self.features_to_process[target_module] = (
                self._get_features_to_process(sae)
            )
            self.feature_examples[target_module] = {
                feature_idx: []
                for feature_idx in self.features_to_process[target_module]
            }
        self.example_counter = 0  # for heap tie breakign

    def _setup_output_dirs(self) -> None:
        """Create output directories for results."""
        now = time.strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(
            self.config.output_dir,
            f"top_examples_{now}",
        )

        os.makedirs(self.save_dir, exist_ok=True)

    def _get_features_to_process(self, sae: TopKSAE) -> List[int]:
        """Determine which features to process based on configuration.

        Args:
            sae (TopKSAE): The SAE instance.

        Returns:
            List[int]: List of feature indices to process.
        """
        if self.config.features_to_process:
            return self.config.features_to_process
        return list(range(sae.d_sae))

    def _load_dataset(self) -> Dataset:
        """Load the prompt dataset.

        Returns:
            Dataset: The loaded dataset.
        """
        dataset = load_from_disk(self.config.dataset_path)
        dataset = dataset["test"]

        # Make sure we have the prompt column
        assert (
            self.config.prompt_column in dataset.column_names
        ), f"Dataset does not contain column: {self.config.prompt_column}"

        if self.config.num_prompts < len(dataset):
            dataset: Dataset = dataset.select(range(self.config.num_prompts))

        logger.info(f"Loaded dataset with {len(dataset)} prompts")
        return dataset

    def _compute_feature_activation_strength(
        self,
        feature_maps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute feature activation strength across timesteps.

        Args:
            feature_maps (torch.Tensor): Feature maps of shape
                [bs, timesteps, spatial, d_sae].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Max activation strength per feature across timesteps
                - Timestep indices with max activation
        """
        # Get spatial mean for each timestep
        # [bs, timesteps, spatial, d_sae] -> [bs, timesteps, d_sae]
        spatial_means = feature_maps.mean(dim=2)

        if self.config.temporal_aggregation == "max":
            # max across timesteps for each feature
            # [bs, timesteps, d_sae] -> [bs, d_sae]
            values, max_timesteps = spatial_means.max(dim=1)
        else:  # Mean aggregation
            # Get mean across timesteps for each feature
            # [bs, timesteps, d_sae] -> [bs, d_sae]
            values = spatial_means.mean(dim=1)
            _, max_timesteps = spatial_means.max(dim=1)

        return values, max_timesteps

    def _convert_timestep_to_diffusion_time(
        self, timestep: int | torch.Tensor
    ) -> float | torch.Tensor:
        """
        Convert discrete timestep to normalized diffusion time for plotting.

        Args:
            timestep (int): The current timestep (1-indexed).

        Returns:
            float: Normalized diffusion time in the range [0, 1].
        """
        return (timestep - 1) / (max(self.scheduler_timesteps) - 1)

    def extract_top_examples(self) -> None:
        """Extract top examples for each feature."""
        dataset = self._load_dataset()
        batch_size = self.config.batch_size
        num_batches = (len(dataset) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="Processing prompts"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            batch_examples = dataset[start_idx:end_idx]

            prompts = batch_examples[self.config.prompt_column]
            prompt_indices = list(range(start_idx, end_idx))
            self._process_prompts_batch(prompts, prompt_indices)

        self._filter_and_save_top_examples()

    @torch.no_grad()
    def _process_prompts_batch(
        self, prompts: List[str], prompt_indices: List[int]
    ) -> None:
        """Process a batch of prompts to extract feature activations.

        Args:
            prompts (List[str]): List of text prompts.
            prompt_indices (List[int]): Indices of the prompts in the dataset.
        """
        meta_prompts = [
            p for p in prompts for _ in range(self.config.number_of_seeds)
        ]
        base_seeds = [42 + i for i in range(self.config.number_of_seeds)]
        meta_seeds = [s for _ in prompts for s in base_seeds]
        generators = [
            torch.Generator(device=self.device).manual_seed(seed)
            for seed in meta_seeds
        ]

        # Cache activations from ALL modules at once for massive speedup
        target_modules = list(self.config.sae_config.keys())

        # ---------------------------------------------------------------------
        # Cache Activations from the target module
        # ---------------------------------------------------------------------
        images, cache = self.pipe.run_with_cache(
            prompt=meta_prompts,
            positions_to_cache=list(target_modules),
            num_inference_steps=self.config.num_inference_steps,
            generator=generators,
            guidance_scale=self.config.guidance_scale,
            save_input=True,
            save_output=True,
            height=self.config.height,
            width=self.config.width,
        )

        for target_module, sae in self.saes.items():
            if self.config.output_or_diff == "diff":
                output = (
                    cache["output"][target_module]
                    - cache["input"][target_module]
                )
            else:
                output = cache["output"][target_module]

            meta_bs, timesteps, spatial_dim, _ = output.shape
            # don't use vectorized processing when using batch topk
            # this way we don't have leakage
            if sae.cfg.use_batch_topk:
                all_feature_activations = []
                for b in range(meta_bs):
                    ex = output[b]
                    timestep_feature_maps = []
                    for t in range(timesteps):
                        timestep_output = ex[t]
                        sae_input, _ = sae.preprocess_input(timestep_output)
                        feature_acts, _ = sae._get_topk(
                            torch.relu(sae.encode(sae_input.to(self.device)))
                        )
                        timestep_feature_maps.append(feature_acts)
                    stacked_maps = torch.stack(timestep_feature_maps)
                    all_feature_activations.append(stacked_maps)
                batch_feature_maps = torch.stack(all_feature_activations)
            # if not using batch topk, we can vectorize
            else:
                output_reshaped = output.view(
                    meta_bs * timesteps, spatial_dim, sae.d_in
                )
                sae_input, _ = sae.preprocess_input(output_reshaped)
                feature_acts, _ = sae._get_topk(
                    torch.relu(sae.encode(sae_input.to(self.device)))
                )
                batch_feature_maps = feature_acts.view(
                    meta_bs, timesteps, spatial_dim, sae.d_sae
                )

            batch_agg_activations, batch_max_timesteps = (
                self._compute_feature_activation_strength(batch_feature_maps)
            )

            for i in range(meta_bs):
                original_prompt_idx = i // self.config.number_of_seeds
                self._update_top_examples(
                    target_module=target_module,
                    prompt=prompts[original_prompt_idx],
                    prompt_idx=prompt_indices[original_prompt_idx],
                    image=images[i],
                    agg_activations=batch_agg_activations[i],
                    max_timesteps=batch_max_timesteps[i],
                    feature_maps=batch_feature_maps[i],
                    seed=meta_seeds[i],
                )

    def _update_top_examples(
        self,
        target_module: str,
        prompt: str,
        prompt_idx: int,
        image: Image.Image,
        agg_activations: torch.Tensor,
        max_timesteps: torch.Tensor,
        feature_maps: torch.Tensor,
        seed: int,
    ) -> None:
        """Update the tracking of top examples for each feature.

        Args:
            target_module (str): Target module name.
            prompt (str): Text prompt used.
            prompt_idx (int): Index of the prompt.
            image (Image.Image): Generated image.
            max_activations (torch.Tensor): Max activation value per feature.
            max_timesteps (torch.Tensor): Timestep with max activation per
                feature.
            feature_maps (torch.Tensor): Feature activation maps.
            generator_seed (int): Seed used for generation.
        """
        for feature_idx in self.features_to_process[target_module]:
            activation = agg_activations[feature_idx].item()
            heap = self.feature_examples[target_module][feature_idx]

            max_t_idx = max_timesteps[feature_idx].item()
            feature_map_at_max_t = (
                feature_maps[max_t_idx, :, feature_idx].detach().cpu()
            )

            # Convert step index to actual scheduler timestep
            max_t_val = self.scheduler_timesteps[max_t_idx].item()
            max_diffusion_t = self._convert_timestep_to_diffusion_time(
                max_t_val
            )

            example_info = {
                "prompt": prompt,
                "prompt_idx": prompt_idx,
                "activation": activation,
                "timestep": max_diffusion_t,
                "image": image,
                "feature_map": feature_map_at_max_t,
                "seed": seed,
            }

            if len(heap) < self.config.num_examples_per_feature:
                heapq.heappush(
                    heap, (activation, self.example_counter, example_info)
                )
            else:
                heapq.heappushpop(
                    heap, (activation, self.example_counter, example_info)
                )

            self.example_counter += 1

    def _filter_and_save_top_examples(self) -> None:
        """Sort final top examples and and save visualizations."""
        logger.info("Filtering top examples and creating visualizations...")

        for target_module in self.config.sae_config.keys():
            logger.info(f"--- Processing module: {target_module} ---")

            module_features = self.features_to_process[target_module]

            for feature_idx in tqdm(
                module_features,
                desc=f"Saving visualizations for {target_module}",
            ):
                heap = self.feature_examples[target_module][feature_idx]
                top_examples = sorted(
                    [item[2] for item in heap],
                    key=lambda x: x["activation"],
                    reverse=True,
                )

                # -------------------------------------------------------------
                # Visualize and save top examples
                # -------------------------------------------------------------
                self._create_feature_grid(
                    target_module, feature_idx, top_examples
                )
                if self.config.save_individual_examples:
                    self._save_individual_examples(
                        target_module, feature_idx, top_examples
                    )
                self._save_feature_metadata(
                    target_module, feature_idx, top_examples
                )

    def _get_and_create_module_dir(
        self, target_module: str, subfolder: str
    ) -> str:
        """Helper to create and return path for a module's output subfolder.

        Args:
            target_module (str): Target module name.
            subfolder (str): Subfolder name (e.g., "grids", "individual
                "metadata").

        Returns:
            str: Path to the module's output subfolder.
        """
        sanitized_module_name = target_module.replace(".", "_")
        module_dir = os.path.join(
            self.save_dir, sanitized_module_name, subfolder
        )
        os.makedirs(module_dir, exist_ok=True)
        return module_dir

    def _create_feature_grid(
        self,
        target_module: str,
        feature_idx: int,
        examples: List[Dict[str, Any]],
    ) -> None:
        """Create grid visualization of top examples for a feature.

        Args:
            target_module (str): Target module name.
            feature_idx (int): Feature index.
            examples (List[Dict[str, Any]]): List of example information.
        """
        num_examples = len(examples)

        image_size_inches = 3
        fig_width = num_examples * image_size_inches + 1
        fig_height = image_size_inches

        fig, axes = plt.subplots(
            1,
            num_examples,
            figsize=(fig_width, fig_height),
            squeeze=False,
        )
        axes = axes.flatten()

        axes[0].set_ylabel(
            f"Feature {feature_idx}",
            fontsize=14,
            fontweight="bold",
            rotation=0,
            labelpad=40,
            verticalalignment="center",
            horizontalalignment="right",
        )

        for i, example in enumerate(examples):
            ax = axes[i]

            overlay = self._create_heatmap_overlay(
                example["image"], example["feature_map"]
            )
            ax.imshow(np.array(overlay.convert("RGB")))

            ax.set_title(
                f"$t_{{peak}}=${example['timestep']:.3f}", fontsize=10
            )

            if i == 0:
                # For the first plot, keep the ylabel visible but hide
                # ticks/spines
                ax.spines[["top", "right", "bottom", "left"]].set_visible(
                    False
                )
                ax.tick_params(
                    left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False,
                )
            else:
                # For all other plots, turn the axis off completely
                ax.axis("off")

        fig.subplots_adjust(wspace=0.05, hspace=0)

        grids_dir = self._get_and_create_module_dir(target_module, "grids")
        grid_path = os.path.join(grids_dir, f"feature_{feature_idx}.jpg")
        plt.savefig(
            grid_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close(fig)

    def _save_individual_examples(
        self,
        target_module: str,
        feature_idx: int,
        examples: List[Dict[str, Any]],
    ) -> None:
        """Save individual examples for a feature.

        Args:
            target_module (str): Target module name.
            feature_idx (int): Feature index.
            examples (List[Dict[str, Any]]): List of example information.
        """
        individual_dir = self._get_and_create_module_dir(
            target_module, "individual_examples"
        )
        feature_dir = os.path.join(individual_dir, f"feature_{feature_idx}")
        os.makedirs(feature_dir, exist_ok=True)

        for i, example in enumerate(examples):
            image_path = os.path.join(feature_dir, f"example_{i}_image.jpg")
            example["image"].save(image_path)

            overlay = self._create_heatmap_overlay(
                example["image"], example["feature_map"]
            )
            overlay_path = os.path.join(
                feature_dir, f"example_{i}_overlay.jpg"
            )
            overlay.convert("RGB").save(overlay_path)

            prompt_path = os.path.join(feature_dir, f"example_{i}_prompt.txt")
            with open(prompt_path, "w") as f:
                f.write(example["prompt"])

    def _create_heatmap_overlay(
        self,
        image: Image.Image,
        feature_map: torch.Tensor,
    ) -> Image.Image:
        """Create a heatmap overlay on an image.

        Args:
            image (Image.Image): Original image.
            feature_map (torch.Tensor): Feature activation map.

        Returns:
            Image.Image: Image with heatmap overlay.
        """
        # Convert feature map to numpy and normalize
        if isinstance(feature_map, torch.Tensor):
            feature_map = feature_map.numpy()

        feature_map_norm = (feature_map - feature_map.min()) / (
            feature_map.max() - feature_map.min() + 1e-8
        )

        # Calculate spatial dimensions
        spatial_dim = int(math.sqrt(feature_map.shape[0]))
        feature_map_2d = feature_map_norm.reshape(spatial_dim, spatial_dim)

        # Upsample feature map to image size
        h_ratio = image.height // spatial_dim
        w_ratio = image.width // spatial_dim
        upsampled_map = np.kron(feature_map_2d, np.ones((h_ratio, w_ratio)))

        # Create colormap
        jet = plt.cm.jet
        cmap = jet(np.arange(jet.N))
        cmap[:1, -1] = 0  # Make lowest values transparent
        cmap[1:, -1] = 0.3  # Set alpha for rest of heatmap
        custom_cmap = ListedColormap(cmap)

        heatmap_colored = custom_cmap(upsampled_map)
        heatmap_img = Image.fromarray((heatmap_colored * 255).astype(np.uint8))

        # Create overlay
        img_rgba = image.convert("RGBA")
        overlay = Image.alpha_composite(img_rgba, heatmap_img)

        return overlay

    def _save_feature_metadata(
        self,
        target_module: str,
        feature_idx: int,
        examples: List[Dict[str, Any]],
    ) -> None:
        """Save metadata for a feature's top examples.

        Args:
            target_module (str): Target module name.
            feature_idx (int): Feature index.
            examples (List[Dict[str, Any]]): List of example information.
        """
        metadata = {
            "feature_idx": feature_idx,
            "examples": [
                {
                    "prompt": ex["prompt"],
                    "prompt_idx": ex["prompt_idx"],
                    "activation": ex["activation"],
                    "timestep": ex["timestep"].item(),
                    "seed": ex["seed"],
                }
                for ex in examples
            ],
        }

        metadata_dir = self._get_and_create_module_dir(
            target_module, "metadata"
        )
        metadata_path = os.path.join(
            metadata_dir, f"feature_{feature_idx}.json"
        )
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    def run(self) -> None:
        """Run the top examples extraction process."""
        set_all_seeds(42)

        logger.info(
            f"Extracting top examples for {len(self.features_to_process)}"
            " features"
        )
        logger.info(f"Results will be saved to {self.save_dir}")
        self.extract_top_examples()
        logger.info(f"Extraction complete. Results saved to {self.save_dir}")


if __name__ == "__main__":
    main()
