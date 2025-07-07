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
from dataclasses import dataclass
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

    sae_path: str
    """Path to the SAE model checkpoint."""

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
        logger.info(f"Loading SAE model from {config.sae_path}")
        self.sae = TopKSAE.load_from_disk(
            path_str=config.sae_path,
            config_class=TopKSAEConfig,
            device=config.device,
        ).to(dtype=self.torch_dtype)
        self.sae.eval()

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
        self.features_to_process = self._get_features_to_process()
        self.feature_examples = {
            feature_idx: [] for feature_idx in self.features_to_process
        }
        self.example_counter = 0  # for heap tie breakign

    def _setup_output_dirs(self) -> None:
        """Create output directories for results."""
        now = time.strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(
            self.config.output_dir,
            f"top_examples_{now}",
        )

        self.grids_dir = os.path.join(self.save_dir, "grids")
        self.individual_dir = os.path.join(
            self.save_dir, "individual_examples"
        )
        self.metadata_dir = os.path.join(self.save_dir, "metadata")

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.grids_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

        if self.config.save_individual_examples:
            os.makedirs(self.individual_dir, exist_ok=True)

    def _get_features_to_process(self) -> List[int]:
        """Determine which features to process based on configuration.

        Returns:
            List[int]: List of feature indices to process.
        """
        if self.config.features_to_process:
            return self.config.features_to_process
        return list(range(self.sae.d_sae))

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

        Implements the formula:
        A_t^r = (1/wh) * sum(i=1 to h)sum(j=1 to w) S_ij,t^r
        A^r = max_t(A_t^r)

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

        # ---------------------------------------------------------------------
        # Cache Activations from the target module
        # ---------------------------------------------------------------------
        images, cache = self.pipe.run_with_cache(
            prompt=meta_prompts,
            positions_to_cache=[self.config.target_module],
            num_inference_steps=self.config.num_inference_steps,
            generator=generators,
            guidance_scale=self.config.guidance_scale,
            save_input=True,
            save_output=True,
            height=self.config.height,
            width=self.config.width,
        )

        if self.config.output_or_diff == "diff":
            output = (
                cache["output"][self.config.target_module]
                - cache["input"][self.config.target_module]
            )
        else:
            output = cache["output"][self.config.target_module]

        meta_bs, timesteps, spatial_dim, _ = output.shape

        # don't use vectorized processing when using batch topk
        # this way we don't have leakage
        if self.sae.cfg.use_batch_topk:
            all_feature_activations = []
            for b in range(meta_bs):
                # Extract activations for this example
                ex = output[b]  # [timesteps, spatial, channels]

                timestep_feature_maps = []
                # For each timestep, process through SAE
                for t in range(timesteps):
                    timestep_output = ex[t]  # [spatial, channels]

                    sae_input, _ = self.sae.preprocess_input(timestep_output)
                    feature_acts, _ = self.sae._get_topk(
                        torch.relu(self.sae.encode(sae_input.to(self.device)))
                    )

                    timestep_feature_maps.append(feature_acts)

                # [timesteps, spatial, d_sae]
                stacked_maps = torch.stack(timestep_feature_maps)
                all_feature_activations.append(stacked_maps)

            # [meta_bs, timesteps, spatial, d_sae]
            batch_feature_maps = torch.stack(all_feature_activations)

        # use vectorized processing
        else:
            output = output.view(
                meta_bs * timesteps, spatial_dim, self.sae.d_in
            )
            sae_input, _ = self.sae.preprocess_input(output)
            feature_acts, _ = self.sae._get_topk(
                torch.relu(self.sae.encode(sae_input.to(self.device)))
            )

            # reshape to  [meta_bs, timesteps, spatial, d_sae]
            batch_feature_maps = feature_acts.view(
                meta_bs, timesteps, spatial_dim, self.sae.d_sae
            )

        batch_agg_activations, batch_max_timesteps = (
            self._compute_feature_activation_strength(batch_feature_maps)
        )

        # Update top examples for each example in batch
        for i in range(meta_bs):
            original_prompt_idx_in_batch = i // self.config.number_of_seeds
            self._update_top_examples(
                prompt=prompts[original_prompt_idx_in_batch],
                prompt_idx=prompt_indices[original_prompt_idx_in_batch],
                image=images[i],
                agg_activations=batch_agg_activations[i],
                max_timesteps=batch_max_timesteps[i],
                feature_maps=batch_feature_maps[i],
                seed=meta_seeds[i],
            )

    def _update_top_examples(
        self,
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
            prompt (str): Text prompt used.
            prompt_idx (int): Index of the prompt.
            image (Image.Image): Generated image.
            max_activations (torch.Tensor): Max activation value per feature.
            max_timesteps (torch.Tensor): Timestep with max activation per
                feature.
            feature_maps (torch.Tensor): Feature activation maps.
            generator_seed (int): Seed used for generation.
        """
        for feature_idx in self.features_to_process:
            activation = agg_activations[feature_idx].item()

            heap = self.feature_examples[feature_idx]

            max_t = max_timesteps[feature_idx].item()
            feature_map_at_max_t = (
                feature_maps[max_t, :, feature_idx].detach().cpu()
            )

            max_t = self.scheduler_timesteps[max_t]
            max_diffusion_t = self._convert_timestep_to_diffusion_time(max_t)

            example_info = {
                "prompt": prompt,
                "prompt_idx": prompt_idx,
                "activation": activation,
                "timestep": max_diffusion_t,
                "image": image,
                "feature_map": feature_map_at_max_t,
                "seed": seed,
            }

            # Heaps for efficient top-k management
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

        for feature_idx in tqdm(
            self.features_to_process, desc="Saving feature visualizations"
        ):
            heap = self.feature_examples[feature_idx]
            top_examples = sorted(
                [item[2] for item in heap],
                key=lambda x: x["activation"],
                reverse=True,
            )

            # -----------------------------------------------------------------
            # Visualization and saving
            # -----------------------------------------------------------------
            self._create_feature_grid(feature_idx, top_examples)
            if self.config.save_individual_examples:
                self._save_individual_examples(feature_idx, top_examples)

            self._save_feature_metadata(feature_idx, top_examples)

    def _create_feature_grid(
        self,
        feature_idx: int,
        examples: List[Dict[str, Any]],
    ) -> None:
        """Create grid visualization of top examples for a feature.

        Args:
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
            f"Feature\n{feature_idx}",
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

            ax.set_title(f"t={example['timestep']}", fontsize=10)

            ax.axis("off")

        fig.subplots_adjust(wspace=0.05, hspace=0)

        grid_path = os.path.join(self.grids_dir, f"feature_{feature_idx}.jpg")
        plt.savefig(
            grid_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close(fig)

    def _save_individual_examples(
        self,
        feature_idx: int,
        examples: List[Dict[str, Any]],
    ) -> None:
        """Save individual examples for a feature.

        Args:
            feature_idx (int): Feature index.
            examples (List[Dict[str, Any]]): List of example information.
        """
        feature_dir = os.path.join(
            self.individual_dir, f"feature_{feature_idx}"
        )
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
        cmap[1:, -1] = 0.4  # Set alpha for rest of heatmap
        custom_cmap = ListedColormap(cmap)

        heatmap_colored = custom_cmap(upsampled_map)
        heatmap_img = Image.fromarray((heatmap_colored * 255).astype(np.uint8))

        # Create overlay
        img_rgba = image.convert("RGBA")
        overlay = Image.alpha_composite(img_rgba, heatmap_img)

        return overlay

    def _save_feature_metadata(
        self,
        feature_idx: int,
        examples: List[Dict[str, Any]],
    ) -> None:
        """Save metadata for a feature's top examples.

        Args:
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
                    "timestep": ex["timestep"],
                    "seed": ex["seed"],
                }
                for ex in examples
            ],
        }

        metadata_path = os.path.join(
            self.metadata_dir, f"feature_{feature_idx}.json"
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
