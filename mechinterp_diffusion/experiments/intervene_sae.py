"""
This module implements SAE-based interventions for diffusion models. It allows
for different types of interventions (add, scale, reconstruct) on specific
features of the SAE model during specific timesteps of the diffusion process.


Example usage:
python intervene_sae.py --hook_type add --feature_idx 1674 --value 5.0
python intervene_sae.py --hoook_type scale --feature_idx 1674 --value 5.0
"""

# =========================================================================== #
#                           Packages and Presets                              #
# =========================================================================== #

import dataclasses
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from matplotlib.colors import ListedColormap
from PIL import Image
from simple_parsing import parse
from torch import Tensor

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import SAEInterventionConfig, TopKSAEConfig
from core.diffusion.hooked_sd_pipeline import HookedStableDiffusionPipeline
from core.sae.topk_sae import TopKSAE
from core.utils.hooks import (
    EnhancedTimedHook,
    TimedHook,
    add_feature_hook,
    reconstruct_sae_hook,
    scale_feature_hook,
)
from core.utils.reproducibility import set_all_seeds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",  # noqa: E501
)
logger = logging.getLogger(__name__)

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
}


# =========================================================================== #
#                              Main Functionality                             #
# =========================================================================== #
def main() -> None:
    """Main function for interventions."""
    config = parse(SAEInterventionConfig)
    manager = SAEInterventionManager(config)
    manager.run()


# =========================================================================== #
#                        SAE Interventions Class                              #
# =========================================================================== #
class SAEInterventionManager:
    """
    Manager for SAE-based feature interventions and reconstructions.
    """

    def __init__(
        self,
        config: SAEInterventionConfig,
    ) -> None:
        """Initialize the SAE intervention manager.

        Args:
            config: Configuration for the intervention process
        """
        self.config = config
        assert self.config.hook_type in ["add", "scale", "reconstruct"], (
            f"Invalid hook type: {self.config.hook_type}. Must be one of "
            "'add', 'scale', or 'reconstruct'."
        )

        assert self.config.intervention_mode in [
            "grid",
            "trajectory",
            "topk_trace",
        ], (
            f"Invalid intervention mode: {self.config.intervention_mode}. "
            "Must be one of 'grid', 'trajectory', or 'topk_trace'."
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = DTYPE_MAP.get(config.torch_dtype, torch.float32)

        self.sae_model = TopKSAE.load_from_disk(
            self.config.sae_path,
            config_class=TopKSAEConfig,
            device=self.device,
        ).to(dtype=self.torch_dtype)

        self.pipe = HookedStableDiffusionPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=self.torch_dtype,
            safety_checker=None,
        ).to(self.device, self.torch_dtype)

        self.pipe.scheduler.set_timesteps(
            self.config.num_inference_steps, device="cpu"
        )

        self.scheduler_timesteps = self.pipe.scheduler.timesteps.to("cpu")

        self.timesteps = self.config.timesteps
        self.timestep_values = self.config.timestep_values

        self._setup_output_dirs()
        self._save_config()

    def run(self) -> None:
        """Run the SAE intervention experiment based on configuration."""
        set_all_seeds(self.config.seed)
        prompts = self._load_prompts()

        match self.config.intervention_mode:
            case "grid":
                self.run_grid_mode(prompts)
            case "trajectory":
                self.run_trajectory_mode(prompts)
            case "topk_trace":
                self.run_topk_trace_mode(prompts)

        logger.info(f"Results saved to {self.save_dir}")

    def _setup_output_dirs(self) -> None:
        """Create output directories based on current configuration."""
        now = time.strftime("%Y%m%d_%H%M%S")

        intervention_times = (
            "specific"
            if self.config.timesteps
            else "varying" if self.config.timestep_values else "fixed"
        )

        match self.config.intervention_mode:
            case "grid":
                prefix = "grid"
            case "trajectory":
                prefix = "trajectory"
            case "topk_trace":
                prefix = "topk_trace"

        self.save_dir = os.path.join(
            self.config.output_dir,
            f"{prefix}_hook-{self.config.hook_type}_type-{intervention_times}_{now}",  # noqa: E501
            self.config.target_module,
        )

        os.makedirs(self.save_dir, exist_ok=True)

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

    def _save_config(self) -> None:
        """Save configuration to a JSON file for reproducibility."""
        config_dict = dataclasses.asdict(self.config)
        with open(os.path.join(self.save_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=4)

    def _load_prompts(self) -> List[str]:
        """Load prompts from specified dataset or use provided prompt.

        Returns:
            List[str]: List of prompts to use for image generation
        """
        prompts = []
        if not self.config.prompts:
            dataset = load_from_disk(self.config.dataset_path)
            dataset_split = dataset[self.config.dataset_split]
            logger.info(f"Loaded dataset with {len(dataset_split)} examples")
            for i in range(min(self.config.num_prompts, len(dataset_split))):
                prompt_entry = dataset_split[i]
                prompts.append(prompt_entry[self.config.prompt_column])
        else:
            for prompt in self.config.prompts:
                prompts.append(prompt)

        return prompts

    def _locate_module(self, position: str) -> nn.Module:
        """Locate a module in the pipeline by its position string.

        Args:
            position (str): Position string (dot-separated path)

        Returns:
            nn.Module: The module at the specified position
        """
        block = self.pipe
        for step in position.split("."):
            if step.isdigit():
                block = block[int(step)]
            else:
                block = getattr(block, step)
        return block

    def _register_sae_hook(
        self,
        target_module: str,
        feature_idx: int,
        value: float,
        timesteps: Optional[List[int]] = None,
        timestep_values: Optional[Dict[int, float]] = None,
        hook_type: str = "add",
    ) -> Any:
        """Register an SAE-based hook on a specific module.

        Args:
            target_module (str): Module name to hook.
            feature_idx (int): Feature index to intervene with.
            value (float): Strength of the intervention.
            timesteps (optional, Optional[Dict[int, float]]): Timesteps to
                apply equal-valued intervention. Defaults to None.
            timestep_values (optional, Optional[Dict[int, float]]): Dict
                mapping timesteps to values. Defaults to None.
            hook_type (str): Type of hook to register
                ("add", "scale", "reconstruct").

        Returns:
            Any: Hook handle that can be removed later
        """
        module = self._locate_module(target_module)

        match hook_type:
            case "add":
                core_operation = (  # noqa: E731
                    lambda m, i, o: add_feature_hook(
                        self.sae_model, feature_idx, value, m, i, o
                    )
                )
            case "scale":
                core_operation = (  # noqa: E731
                    lambda m, i, o: scale_feature_hook(
                        self.sae_model, feature_idx, value, m, i, o
                    )
                )
            case "reconstruct":
                core_operation = (  # noqa: E731
                    lambda m, i, o: reconstruct_sae_hook(
                        self.sae_model, m, i, o
                    )
                )

        if timesteps:
            hook_fn = TimedHook(
                core_operation,
                self.pipe.scheduler.num_inference_steps,
                timesteps,
            )
        elif timestep_values:
            hook_fn = EnhancedTimedHook(
                lambda m, i, o: None,
                self.pipe.scheduler.num_inference_steps,
                timestep_values,
                feature_idx,
                self.sae_model,
                hook_type,
            )
        else:
            hook_fn = core_operation

        handle = module.register_forward_hook(hook_fn)
        logger.info(
            f"Registered {hook_type} hook on {target_module} "
            f"for feature {feature_idx}"
        )
        return handle

    def _generate_image(
        self,
        prompt: str,
        generator: torch.Generator,
        hook_handle: Optional[Any] = None,
    ) -> Tuple[Image.Image, List[Image.Image], List[int]]:
        """Generate an image with the current pipeline configuration.

        Args:
            prompt (str): Text prompt for generation
            generator (torch.Generator): Random generator
                for reproducibility.
            hook_handle (optional, Optional[Any]): Optional registered sae
                hook handle for intervention. Defaults to None.

        Returns:
            Tuple[Image.Image, List[Image.Image], List[int]]:
                Final image, intermediate steps, and capture step indices.
        """
        with torch.no_grad():
            output = self.pipe(
                prompt=prompt,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                generator=generator,
                output_type="pil",
                height=self.config.height,
                width=self.config.width,
            )

            final_image = output.images[0]

            # Reset the generator to get the same result
            generator = torch.Generator(self.device).manual_seed(
                generator.initial_seed()
            )

            capture_steps = list(
                range(
                    0,
                    self.config.num_inference_steps,
                    self.config.capture_interval,
                )
            )

            if self.config.num_inference_steps - 1 not in capture_steps:
                capture_steps.append(self.config.num_inference_steps - 1)

            _, intermediates, _ = self.pipe.run_with_cache_intermediate(
                prompt=prompt,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                generator=generator,
                positions_to_cache=[self.config.target_module],
                save_output=True,
                output_type="pil",
                height=self.config.height,
                width=self.config.width,
            )

            intermediate_images = [intermediates[i] for i in capture_steps]

        return final_image, intermediate_images, capture_steps

    def _capture_feature_map(
        self,
        target_module: str,
        feature_idx: int,
        prompt: str,
        capture_timesteps: List[int],
    ) -> Tuple[Image.Image, List[torch.Tensor]]:
        """
        Capture the activation map of a specific SAE feature during diffusion.

        Args:
            target_module (str): Module name to hook.
            feature_idx (int): Feature index to visualize.
            prompt (str): Prompt for image generation.
            capture_timesteps (List[int]): Specific timesteps to capture.

        Returns:
            Tuple[Image.Image, List[torch.Tensor]]: Generated image and list
                of feature map tensors
        """
        module = self._locate_module(target_module)
        feature_maps = {}
        current_step = [0]  # Use a list to allow mutation in the hook

        # Hook to capture feature activations
        def feature_hook(module, input, output):
            if current_step[0] in capture_timesteps:
                diff = (
                    (output[0] - input[0])
                    .permute((0, 2, 3, 1))
                    .to(self.sae_model.device)
                )
                diff, _ = self.sae_model.preprocess_input(diff)
                activations = F.relu(self.sae_model.encode(diff))
                top_acts, _ = self.sae_model._get_topk(
                    activations, k=self.sae_model.cfg.k
                )

                # Extract activations for the target feature
                feature_activations = top_acts[..., feature_idx]

                feature_maps[current_step[0]] = (
                    feature_activations.detach().cpu()
                )

            current_step[0] += 1
            return output

        handle = module.register_forward_hook(feature_hook)

        with torch.no_grad():
            image = self.pipe(
                prompt=prompt,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                output_type="pil",
                height=self.config.height,
                width=self.config.width,
            ).images[0]

        handle.remove()

        feature_map_list = [
            feature_maps[t] for t in sorted(feature_maps.keys())
        ]

        return image, feature_map_list

    def run_grid_mode(self, prompts: List[str]) -> None:
        """
        Run intervention in grid mode, comparing different feature/value
        combinations.

        Args:
            prompts (List[str]): List of prompts to generate images for
        """
        logger.info("Running in grid mode")

        for prompt_idx, prompt in enumerate(prompts):
            prompt_cleaned = (
                prompt[:80]
                .replace(" ", "_")
                .replace(",", "_")
                .replace(".", "_")
            )
            prompt_dir = os.path.join(
                self.save_dir, f"{prompt_idx}_{prompt_cleaned}"
            )
            os.makedirs(prompt_dir, exist_ok=True)

            with open(os.path.join(prompt_dir, "prompt.txt"), "w") as f:
                f.write(prompt)

            # -----------------------------------------------------------------
            # Original Image
            # -----------------------------------------------------------------
            logger.info("Generating original image for grid comparison...")
            generator = torch.Generator(self.device).manual_seed(
                self.config.seed
            )
            original_image, _, _ = self._generate_image(prompt, generator)

            original_path = os.path.join(prompt_dir, "original.png")
            original_image.save(original_path)
            logger.info(f"Saved original image to {original_path}")

            # -----------------------------------------------------------------
            # Intervention Grid
            # -----------------------------------------------------------------
            grid_images = []

            for feature_idx in self.config.features:
                feature_row = []
                feature_dir = os.path.join(
                    prompt_dir, f"feature_{feature_idx}"
                )
                os.makedirs(feature_dir, exist_ok=True)

                feature_row.append(original_image)

                for value in self.config.intervention_values:
                    logger.info(
                        f"Generating grid image for feature {feature_idx}, "
                        f"value {value}"
                    )

                    hook_handle = self._register_sae_hook(
                        target_module=self.config.target_module,
                        feature_idx=feature_idx,
                        value=value,
                        hook_type=self.config.hook_type,
                        timesteps=self.timesteps,
                        timestep_values=self.timestep_values,
                    )

                    # identical seed for reproducibility
                    generator = torch.Generator(self.device).manual_seed(
                        self.config.seed
                    )
                    intervened_image, _, _ = self._generate_image(
                        prompt, generator, hook_handle
                    )

                    hook_handle.remove()

                    compare_path = os.path.join(
                        feature_dir, f"compare_value_{value}.png"
                    )
                    self._visualize_results(
                        original_images=[original_image],
                        intervened_images=[intervened_image],
                        save_path=compare_path,
                    )
                    logger.info(
                        f"Saved comparison for feature {feature_idx}, "
                        f"value {value}"
                    )

                    feature_row.append(intervened_image)

                grid_images.append(feature_row)

            grid_path = os.path.join(prompt_dir, "grid_visualization.png")
            column_labels = ["Original"] + self.config.intervention_values

            self._visualize_grid_results(
                grid_images=grid_images,
                save_path=grid_path,
                intervention_values=column_labels,
                feature_indices=self.config.features,
                prompt=prompt,
            )
            logger.info(f"Grid visualization saved to {grid_path}")

    def run_trajectory_mode(self, prompts: List[str]) -> None:
        """
        Run intervention in trajectory mode, analyzing the diffusion process
        over time.

        Args:
            prompts (List[str]): List of prompts to generate images for.
        """
        for prompt_idx, prompt in enumerate(prompts):
            for _, feature in enumerate(self.config.features):
                prompt_name = (
                    prompt[:80]
                    .replace(" ", "_")
                    .replace(",", "_")
                    .replace(".", "_")
                )
                prompt_dir = os.path.join(
                    self.save_dir,
                    f"{prompt_idx}_{prompt_name}",
                    f"feature_{feature}",
                )
                os.makedirs(prompt_dir, exist_ok=True)
                logger.info(f"Results will be saved to: {prompt_dir}")

                with open(
                    os.path.join(
                        self.save_dir,
                        f"{prompt_idx}_{prompt_name}",
                        "prompt.txt",
                    ),
                    "w",
                ) as f:
                    f.write(prompt)

                # -------------------------------------------------------------
                # Original Image
                # -------------------------------------------------------------
                logger.info(
                    "Generating original images with intermediate steps..."
                )

                generator = torch.Generator(self.device).manual_seed(
                    self.config.seed
                )
                original_final, original_intermediates, capture_steps = (
                    self._generate_image(prompt, generator)
                )
                original_path = os.path.join(prompt_dir, "original.png")
                original_final.save(original_path)
                logger.info(f"Saved original image to {original_path}")

                if self.config.save_activation_heatmap:
                    self._generate_feature_maps(
                        prompt,
                        prompt_dir,
                        original_intermediates,
                        capture_steps,
                    )

                # -------------------------------------------------------------
                # Intervened Image
                # -------------------------------------------------------------
                logger.info(
                    f"Generating images with intervention for feature "
                    f"{feature}..."
                )
                hook_handle = self._register_sae_hook(
                    target_module=self.config.target_module,
                    feature_idx=feature,
                    value=self.config.intervention_values[0],
                    timesteps=self.timesteps,
                    timestep_values=self.timestep_values,
                    hook_type=self.config.hook_type,
                )

                generator = torch.Generator(self.device).manual_seed(
                    self.config.seed
                )
                intervened_final, intervened_intermediates, _ = (
                    self._generate_image(prompt, generator, hook_handle)
                )
                hook_handle.remove()

                intervened_path = os.path.join(prompt_dir, "intervened.png")
                intervened_final.save(intervened_path)
                logger.info(f"Saved intervened image to {intervened_path}")

                # -------------------------------------------------------------
                # Comparison
                # -------------------------------------------------------------
                self._visualize_results(
                    original_images=[original_final],
                    intervened_images=[intervened_final],
                    save_path=os.path.join(prompt_dir, "comparison.png"),
                )
                logger.info(
                    f"Saved final comparison to "
                    f"{os.path.join(prompt_dir, 'comparison.png')}"
                )

                self._visualize_intermediate_results(
                    original_images=original_intermediates,
                    intervened_images=intervened_intermediates,
                    save_path=os.path.join(
                        prompt_dir, "intermediate_comparison.png"
                    ),
                    prompts=[prompt],
                    timesteps=capture_steps,
                )

    def _generate_feature_maps(
        self,
        prompt: str,
        output_dir: str,
        original_images: List[Image.Image],
        capture_steps: List[int],
    ) -> None:
        """Generate and save feature maps.

        Args:
            prompt (str): Text prompt for generation
            output_dir (str): Directory to save feature maps to
            original_images (List[Image.Image]): List of original images
            capture_steps (List[int]): List of timesteps to capture
        """
        logger.info(
            f"Generating feature maps for feature {self.config.features[0]}..."
        )

        _, feature_maps = self._capture_feature_map(
            target_module=self.config.target_module,
            feature_idx=self.config.features[0],
            prompt=prompt,
            capture_timesteps=capture_steps,
        )

        self._visualize_feature_maps(
            feature_maps=feature_maps,
            images=original_images,
            save_path=os.path.join(output_dir, "feature_maps_over_time.png"),
            timesteps=capture_steps,
        )

    def _visualize_results(
        self,
        original_images: List[Image.Image],
        intervened_images: List[Image.Image],
        save_path: str,
    ) -> None:
        """
        Visualize a comparison, precisely styled to match the grid plot.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        title_font_properties = {
            "fontsize": 14,
            "fontweight": "bold",
            "pad": 10,
        }

        # Plot Original Image
        axes[0].imshow(np.array(original_images[0]))
        axes[0].set_title("Original", **title_font_properties)
        axes[0].axis("off")

        # Plot Intervened/Reconstructed Image
        title = (
            "Reconstruction"
            if self.config.hook_type == "reconstruct"
            else "Intervened"
        )
        interv_array = np.array(intervened_images[0])
        axes[1].imshow(interv_array)
        axes[1].set_title(title, **title_font_properties)
        axes[1].axis("off")

        # Plot Difference Map
        orig_array = np.array(original_images[0]).astype(np.float32)
        interv_array = interv_array.astype(np.float32)
        diff = np.abs(orig_array - interv_array)
        diff_max = diff.max()
        if diff_max > 0:
            diff /= diff_max
        axes[2].imshow(diff, cmap="viridis")
        axes[2].set_title("Difference", **title_font_properties)
        axes[2].axis("off")

        plt.subplots_adjust(
            left=0.05, right=0.95, top=0.9, bottom=0.05, wspace=0.15
        )

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved styled comparison to {save_path}")

    def _visualize_intermediate_results(
        self,
        original_images: List[Image.Image],
        intervened_images: List[Image.Image],
        save_path: str,
        prompts: List[str],
        timesteps: Optional[List[int]] = None,
    ) -> None:
        """Visualize original and intervened images at various timesteps.

        Args:
            original_images (List[Image.Image]): List of original images at
                different timesteps.
            intervened_images (List[Image.Image]): List of intervened images
                at different timesteps.
            save_path (str): Path to save visualization.
            prompts List[str]: Prompts used for generation.
            timesteps (Optional[List[int]]): Timesteps corresponding to images.
        """
        n_steps = len(original_images)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig, axes = plt.subplots(3, n_steps, figsize=(n_steps * 4, 12))
        axes[0, 0].set_ylabel(
            "Original",
            fontsize=14,
            fontweight="bold",
            rotation=0,
            labelpad=60,
            verticalalignment="center",
        )
        if hasattr(self, "config") and self.config.hook_type == "reconstruct":
            axes[1, 0].set_ylabel(
                "Reconstruction",
                fontsize=14,
                fontweight="bold",
                rotation=0,
                labelpad=60,
                verticalalignment="center",
            )
        else:
            axes[1, 0].set_ylabel(
                "Intervened",
                fontsize=14,
                fontweight="bold",
                rotation=0,
                labelpad=60,
                verticalalignment="center",
            )
        axes[2, 0].set_ylabel(
            "Difference",
            fontsize=14,
            fontweight="bold",
            rotation=0,
            labelpad=60,
            verticalalignment="center",
        )

        for i in range(n_steps):

            diffusion_time = self._convert_timestep_to_diffusion_time(
                self.scheduler_timesteps[timesteps[i]]
            )
            timestep_label = f"t={diffusion_time:.3f}"
            # Original image:
            orig_img = original_images[i]
            if isinstance(orig_img, list) and len(orig_img) > 0:
                orig_img = orig_img[0]
            orig_array = np.array(orig_img)

            axes[0, i].imshow(orig_array)
            axes[0, i].set_title(timestep_label)

            # Intervened image:
            interv_img = intervened_images[i]
            if isinstance(interv_img, list) and len(interv_img) > 0:
                interv_img = interv_img[0]
            interv_array = np.array(interv_img)

            axes[1, i].imshow(interv_array)

            # Calculate and plot difference
            orig_array = orig_array.astype(np.float32)
            interv_array = interv_array.astype(np.float32)

            diff = np.abs(orig_array - interv_array)
            diff_max = diff.max()
            if diff_max > 0:
                diff = diff / diff_max

            axes[2, i].imshow(diff, cmap="viridis")

            # ylabel; ensure no conflict with turning off axes
            for ax in axes[:, i]:
                if i == 0:
                    ax.spines[["top", "right", "bottom"]].set_visible(False)
                    ax.spines["left"].set_visible(False)

                    ax.tick_params(
                        left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False,
                    )
                else:
                    ax.axis("off")

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.05, hspace=0)
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close(fig)

    def _visualize_feature_maps(
        self,
        feature_maps: List[torch.Tensor],
        images: List[Image.Image],
        save_path: str,
        timesteps: Optional[List[int]] = None,
    ) -> None:
        """Visualize feature activation maps across different timesteps.

        Args:
            feature_maps (List[torch.Tensor]): List of feature maps at
                different timesteps
            images (List[Image.Image]): List of generated images at different
                timesteps
            save_path (str): Path to save visualization
            feature_idx (int): Feature index being visualized
            timesteps (optional, Optional[List[int]]): Timesteps corresponding
                to images. Defaults to None.
        """
        n_steps = len(feature_maps)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 4, 5))

        for i in range(n_steps):
            diffusion_time = self._convert_timestep_to_diffusion_time(
                self.scheduler_timesteps[timesteps[i]]
            )
            timestep_label = f"t={diffusion_time:.3f}"

            img = images[i]
            if isinstance(img, list) and len(img) > 0:
                img = img[0]
            img_array = np.array(img)

            img_heatmap_overlay = self._plot_heatmap_overlay(
                original_image=img_array,
                heatmap=feature_maps[i],
            )

            axes[i].imshow(img_heatmap_overlay)
            axes[i].set_title(f"{timestep_label}")
            axes[i].axis("off")

        plt.subplots_adjust(top=0.9)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_heatmap_overlay(
        self,
        original_image: Union[Tensor, np.ndarray],
        heatmap: Union[Tensor, np.ndarray],
    ) -> Image.Image:
        """
        Create a heatmap overlay on the original image to visualize feature
        activations. Based on the implementation in sdxl-unbox, see:
        https://github.com/surkovv/sdxl-unbox/blob/
        d5e383fea440aed59d533062f3d8f8435c9a3737/app.py

        Args:
            original_image (Union[Tensor, np.ndarray]): Original image to
                overlay the heatmap on.
            heatmap (Union[Tensor, np.ndarray]): Heatmap to overlay on the
                original image.

        Returns:
            Image.Image: Image with heatmap overlay.
        """
        if isinstance(original_image, Tensor):
            original_image = original_image.cpu().numpy()
        if isinstance(heatmap, Tensor):
            heatmap = heatmap.cpu().numpy()

        if heatmap.shape[0] == 2:
            heatmap = heatmap.mean(axis=0)

        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)

        image = Image.fromarray(original_image).convert("RGBA")

        # Upsample heatmap:
        heatmap = np.kron(heatmap, np.ones((32, 32)))

        jet = plt.cm.jet
        cmap = jet(np.arange(jet.N))
        cmap[:1, -1] = 0
        cmap[1:, -1] = 0.6
        cmap = ListedColormap(cmap)
        heatmap = (heatmap - np.min(heatmap)) / (
            np.max(heatmap) - np.min(heatmap)
        )
        heatmap_rgba = cmap(heatmap)
        heatmap_image = Image.fromarray((heatmap_rgba * 255).astype(np.uint8))
        heatmap_with_transparency = Image.alpha_composite(image, heatmap_image)
        return heatmap_with_transparency

    def _visualize_grid_results(
        self,
        grid_images: List[List[Image.Image]],
        save_path: str,
        intervention_values: List[Union[float, str]],
        feature_indices: List[int],
        prompt: str,
    ) -> None:
        """Create and save grid visualizations of feature intervention results.

        Args:
            grid_images (List[List[Image.Image]]): 2D list of images by
                feature and value.
            save_path (str): Path where to save the visualization.
            intervention_values (List[Union[float, str]]): Values for each
                column in the grid.
            feature_indices (List[int]): Indices of features for each row.
            prompt (str): Text prompt used to generate the images.
        """
        n_rows = len(feature_indices)
        n_cols = len(intervention_values)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4.5), squeeze=False
        )

        # Column titles
        for j, val in enumerate(intervention_values):
            title_text = str(val) if isinstance(val, str) else f"ξ={val:.2f}"
            axes[0, j].set_title(
                title_text, pad=10, fontsize=14, fontweight="bold"
            )

        # Feature/row labels
        for i, feat in enumerate(feature_indices):
            axes[i, 0].set_ylabel(
                f"Feature {feat}", fontsize=14, fontweight="bold", labelpad=20
            )

        for i in range(n_rows):
            for j in range(n_cols):
                if (
                    i < len(grid_images)
                    and j < len(grid_images[i])
                    and grid_images[i][j] is not None
                ):
                    axes[i, j].imshow(np.array(grid_images[i][j]))
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])

        plt.subplots_adjust(
            left=0.15,
            right=0.98,
            top=0.92,
            bottom=0.05,
            wspace=0.15,
            hspace=0.15,
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        self._create_difference_grid(
            grid_images,
            save_path.replace(".png", "_differences.png"),
            intervention_values,
            feature_indices,
            prompt,
        )

    def _create_difference_grid(
        self,
        grid_images: List[List[Image.Image]],
        save_path: str,
        intervention_values: List[Union[float, str]],
        feature_indices: List[int],
        prompt: str,
    ) -> None:
        """Create and save a grid visualization showing differences
        between original and modified images.

        Args:
            grid_images: 2D list of images by feature and value
            save_path: Path where to save the visualization
            intervention_values: Values for each column in the grid
            feature_indices: Indices of features for each row
            prompt: Text prompt used to generate the images
        """
        n_rows = len(feature_indices)
        n_cols = len(intervention_values)

        fig_diff, axes_diff = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4.5), squeeze=False
        )

        # Column titles
        for j, val in enumerate(intervention_values):
            title_text = (
                str(val) if isinstance(val, str) else f"Value: {val:.2f}"
            )
            axes_diff[0, j].set_title(
                title_text, pad=10, fontsize=14, fontweight="bold"
            )

        # Feature/row labels
        for i, feat in enumerate(feature_indices):
            axes_diff[i, 0].set_ylabel(
                f"Feature {feat}", labelpad=20, fontsize=14, fontweight="bold"
            )

        for i in range(n_rows):
            if i < len(grid_images) and len(grid_images[i]) > 0:
                # First column shows original image
                if grid_images[i][0] is not None:
                    axes_diff[i, 0].imshow(np.array(grid_images[i][0]))
                axes_diff[i, 0].set_xticks([])
                axes_diff[i, 0].set_yticks([])

                for j in range(1, n_cols):
                    if (
                        j < len(grid_images[i])
                        and grid_images[i][j] is not None
                    ):
                        orig = np.array(grid_images[i][0]).astype(np.float32)
                        comp = np.array(grid_images[i][j]).astype(np.float32)
                        diff = np.abs(orig - comp)
                        if diff.max() > 0:
                            diff = diff / diff.max()
                        axes_diff[i, j].imshow(diff, cmap="viridis")
                    axes_diff[i, j].set_xticks([])
                    axes_diff[i, j].set_yticks([])

        plt.subplots_adjust(
            left=0.15,
            right=0.98,
            top=0.92,
            bottom=0.05,
            wspace=0.15,
            hspace=0.15,
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig_diff)

    def run_topk_trace_mode(self, prompts: List[str]) -> None:
        """
        Run top-k feature tracing mode to analyze which features are most
        active at each timestep, and see what happens to features when we
        intervene on the top feature at a specific timestep.

        Args:
            prompts (List[str]): List of prompts to analyze
        """
        knockout_timestep = self.config.timesteps[0]

        logger.info(
            f"Running top-k trace (knockout cascade analysis) for "
            f"feature at t={knockout_timestep}."
        )

        for prompt_idx, prompt in enumerate(prompts):
            prompt_name = (
                prompt[:80]
                .replace(" ", "_")
                .replace(",", "_")
                .replace(".", "_")
            )
            prompt_dir = os.path.join(
                self.save_dir,
                f"{prompt_idx}_{prompt_name}_knockout_t_{knockout_timestep}",
            )
            os.makedirs(prompt_dir, exist_ok=True)
            with open(os.path.join(prompt_dir, "prompt.txt"), "w") as f:
                f.write(prompt)

            logger.info("Performing baseline top-k trace:")
            generator = torch.Generator(self.device).manual_seed(
                self.config.seed
            )
            img_orig, data_orig = self._capture_topk_over_time(
                prompt=prompt,
                k=self.config.topk_trace_k,
                generator=generator,
            )
            img_orig.save(os.path.join(prompt_dir, "original_image.png"))

            activations_at_knockout = data_orig["activations"][
                knockout_timestep
            ]
            feat_to_knockout = np.argmax(activations_at_knockout).item()
            logger.info(
                f"Top feature at t={knockout_timestep} is #{feat_to_knockout}"
                f"Preparing knockout run."
            )

            final_prompt_dir = f"{prompt_dir}_feat_{feat_to_knockout}"
            os.rename(prompt_dir, final_prompt_dir)
            prompt_dir = final_prompt_dir

            logger.info(
                f"Performing intervention run with feature {feat_to_knockout} "
                f"knocked out at t={knockout_timestep}."
            )
            hook_handle = self._register_sae_hook(
                target_module=self.config.target_module,
                feature_idx=feat_to_knockout,
                value=self.config.intervention_values[0],
                hook_type="scale",
                timesteps=[knockout_timestep],
            )

            generator = torch.Generator(self.device).manual_seed(
                self.config.seed
            )
            img_int, data_int = self._capture_topk_over_time(
                prompt=prompt,
                k=self.config.topk_trace_k,
                generator=generator,
            )
            hook_handle.remove()
            img_int.save(os.path.join(prompt_dir, "intervened_image.png"))

            # -----------------------------------------------------------------
            # Final Image Comparison
            # -----------------------------------------------------------------
            self._visualize_results(
                original_images=[img_orig],
                intervened_images=[img_int],
                save_path=os.path.join(prompt_dir, "comparison.png"),
            )

            self._visualize_cascade_effect(
                data_orig=data_orig,
                data_int=data_int,
                img_orig=img_orig,
                img_int=img_int,
                prompt=prompt,
                feat_to_knockout=feat_to_knockout,
                knockout_timestep=knockout_timestep,
                output_dir=prompt_dir,
            )

    def _capture_topk_over_time(
        self,
        prompt: str,
        k: int,
        generator: torch.Generator,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Capture the activations at each timestep during a single generation
        run.

        Args:
            prompt (str): Prompt for image generation.
            k (int): Number of top features to track.
            generator (torch.Generator): Random generator for reproducibility.

        Returns:
            Tuple[Image.Image, Dict[str, Any]]: A tuple containing:
                - The final generated image.
                - A dictionary with feature data over time.
        """
        module = self._locate_module(self.config.target_module)

        topk_features_by_timestep = []
        activations_by_timestep = []
        current_step = [0]

        def capture_hook(module, input, output):
            diff = (
                (output[0] - input[0])
                .permute((0, 2, 3, 1))
                .to(self.sae_model.device)
            )

            # ! only use conditional term to quantify importance of feature
            if self.config.guidance_scale > 1.0 and diff.shape[0] == 2:
                diff = diff[1]

            h, w, channels = diff.shape
            diff, _ = self.sae_model.preprocess_input(diff.view(-1, channels))
            activations = F.relu(self.sae_model.encode(diff))
            top_acts, _ = self.sae_model._get_topk(
                activations, k=self.sae_model.cfg.k
            )

            spatial_mean_activations = top_acts.mean(dim=0)

            _, topk_indices = torch.topk(spatial_mean_activations, k, dim=0)

            topk_features_by_timestep.append(
                topk_indices.detach().cpu().numpy()
            )
            activations_by_timestep.append(
                spatial_mean_activations.detach().cpu().numpy()
            )
            current_step[0] += 1
            return output

        handle = module.register_forward_hook(capture_hook)

        with torch.no_grad():
            output = self.pipe(
                prompt=prompt,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                generator=generator,
                output_type="pil",
                height=self.config.height,
                width=self.config.width,
            )
            image = output.images[0]

        handle.remove()

        num_steps = self.config.num_inference_steps
        data = {
            "timesteps": np.arange(num_steps),
            "activations": activations_by_timestep,
        }
        return image, data

    def _visualize_cascade_effect(
        self,
        data_orig: Dict[str, Any],
        data_int: Dict[str, Any],
        img_orig: Image.Image,
        img_int: Image.Image,
        prompt: str,
        feat_to_knockout: int,
        knockout_timestep: int,
        output_dir: str,
    ) -> None:
        """
        Visualize the cascade effect of a feature knockout from multiple angles

        Args:
            data_orig (Dict[str, Any]): Data from the original run.
            data_int (Dict[str, Any]): Data from the intervened run.
            img_orig (Image.Image): Original image.
            img_int (Image.Image): Intervened image.
            prompt (str): The prompt used for generation.
            feat_to_knockout (int): Feature index that was knocked out.
            knockout_timestep (int): Timestep at which the feature was knocked
                out.
            output_dir (str): Directory to save the visualizations.
        """
        logger.info(
            f"Generating cascade effect visualizations in {output_dir}"
        )

        info_path = os.path.join(output_dir, "knockout_info.txt")
        with open(info_path, "w") as f:
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Knockout Timestep (raw): {knockout_timestep}\n")
            diffusion_time = self._convert_timestep_to_diffusion_time(
                self.scheduler_timesteps[knockout_timestep]
            )
            f.write(
                f"Knockout Timestep (diffusion time): {diffusion_time:.4f}\n"
            )
            f.write(f"Knocked-out Feature ID: {feat_to_knockout}\n")
        logger.info(f"Knockout details saved to {info_path}")

        num_steps = len(data_orig["timesteps"])
        activations_orig = np.stack(data_orig["activations"])
        activations_int = np.stack(data_int["activations"])
        analysis_range = range(knockout_timestep + 1, num_steps)
        analysis_diffusion_times = [
            self._convert_timestep_to_diffusion_time(
                self.scheduler_timesteps[t]
            )
            for t in analysis_range
        ]

        cosine_sims, l2_distances = [], []
        for t in analysis_range:
            vec_orig = activations_orig[t]
            vec_int = activations_int[t]
            # Cosine similarity
            dot_product = np.dot(vec_orig, vec_int)
            norm_prod = np.linalg.norm(vec_orig) * np.linalg.norm(vec_int)
            cosine_sims.append(dot_product / norm_prod if norm_prod > 0 else 0)
            # L2 distance
            l2_distances.append(np.linalg.norm(vec_orig - vec_int))

        fig2, axes2 = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        axes2[0].plot(
            analysis_diffusion_times,
            cosine_sims,
            "o-",
            label="Cosine Similarity",
        )
        axes2[0].set_title(
            "Activation Vector Similarity After Knockout", fontsize=16
        )
        axes2[0].set_ylabel("Cosine Similarity", fontsize=12)
        axes2[0].grid(True, linestyle="--", alpha=0.6)
        axes2[0].legend()
        axes2[0].set_ylim(-1.05, 1.05)

        axes2[1].plot(
            analysis_diffusion_times,
            l2_distances,
            "s-",
            color="red",
            label="L2 Distance",
        )
        axes2[1].set_title(
            "Activation Vector Distance After Knockout", fontsize=16
        )
        axes2[1].set_xlabel("Diffusion Time", fontsize=12)
        axes2[1].set_ylabel("L2 Distance", fontsize=12)
        axes2[1].grid(True, linestyle="--", alpha=0.6)
        axes2[1].legend()

        fig2.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "2_cascade_metrics.png"),
            bbox_inches="tight",
        )
        plt.close(fig2)

        # --- Plot 3: Statistical Analysis of Activation Changes ---
        delta_activations = (
            activations_int[list(analysis_range)]
            - activations_orig[list(analysis_range)]
        ).flatten()

        fig3, axes3 = plt.subplots(1, 2, figsize=(10, 7))

        axes3[0].hist(delta_activations, bins=50, color="purple", alpha=0.7)
        axes3[0].set_title("Distribution of Activation Changes", fontsize=16)
        axes3[0].set_xlabel(
            "Change in Activation (Intervened - Original)", fontsize=12
        )
        axes3[0].set_ylabel("Frequency", fontsize=12)
        axes3[0].axvline(0, color="k", linestyle="--", lw=2)
        axes3[0].grid(True, linestyle="--", alpha=0.5)

        axes3[1].boxplot(delta_activations, vert=False, patch_artist=True)
        axes3[1].set_title("Summary of Activation Changes", fontsize=16)
        axes3[1].set_xlabel("Change in Activation", fontsize=12)
        axes3[1].grid(True, linestyle="--", alpha=0.5)
        axes3[1].axvline(0, color="k", linestyle="--", lw=2)

        fig3.suptitle(
            f"Statistical Impact on All Feature Activations Post-Knockout (t > {knockout_timestep})",  # noqa: E501
            fontsize=18,
            y=1.02,
        )
        fig3.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "3_activation_statistics.png"),
            bbox_inches="tight",
        )
        plt.close(fig3)

        delta_activations_by_time = []
        timestep_labels = []

        for i, t in enumerate(analysis_range):
            delta_t = activations_int[t] - activations_orig[t]
            delta_activations_by_time.append(delta_t)
            timestep_labels.append(f"t={analysis_diffusion_times[i]:.3f}")

        # Reverse order so early times (smaller t values) are at top
        delta_activations_by_time = delta_activations_by_time[::-1]
        timestep_labels = timestep_labels[::-1]
        analysis_diffusion_times_rev = analysis_diffusion_times[::-1]

        # Create histogram ridges plot
        fig3, ax3 = plt.subplots(1, 1, figsize=(16, 12))

        # Parameters for ridge plot
        ridge_height = 0.8  # Height of each ridge
        ridge_spacing = 1.0  # Spacing between ridges
        colors = plt.cm.viridis(
            np.linspace(0, 1, len(delta_activations_by_time))
        )

        # Find global min/max for consistent x-axis
        all_deltas = np.concatenate(delta_activations_by_time)
        x_min, x_max = np.percentile(
            all_deltas, [1, 99]
        )  # Use percentiles to avoid outliers

        # Use a reasonable number of bins
        n_bins = min(50, max(10, int(np.sqrt(len(all_deltas)))))
        bin_edges = np.linspace(x_min, x_max, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]

        y_positions = []

        for i, (delta_t, label, color) in enumerate(
            zip(
                delta_activations_by_time,
                timestep_labels,
                colors,
                strict=False,
            )
        ):
            # Calculate histogram
            hist, _ = np.histogram(delta_t, bins=bin_edges, density=True)

            # Normalize histogram height for consistent ridge heights
            if hist.max() > 0:
                hist = hist / hist.max() * ridge_height

            # Calculate y position for this ridge (early times at top)
            y_base = i * ridge_spacing
            y_positions.append(y_base)

            # Create ridge using step plot for histogram
            for _, (bin_center, height) in enumerate(
                zip(bin_centers, hist, strict=False)
            ):  # noqa: E501
                ax3.bar(
                    bin_center,
                    height,
                    bottom=y_base,
                    width=bin_width * 0.8,
                    color=color,
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.1,
                )

            # Add baseline
            ax3.axhline(y=y_base, color="black", alpha=0.2, linewidth=0.5)

            # Add text label on the left
            ax3.text(
                x_min - (x_max - x_min) * 0.05,
                y_base + ridge_height / 2,
                label,
                ha="right",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        # Add vertical line at x=0
        ax3.axvline(
            0,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label="No Change",
            zorder=10,
        )

        # Customize the plot
        ax3.set_xlabel(
            "Change in Activation (Intervened - Original)", fontsize=14
        )
        ax3.set_ylabel("Diffusion Time (Early → Late)", fontsize=14)
        ax3.set_title(
            "Histogram Ridges: Activation Changes Over Time After Knockout",
            fontsize=16,
            pad=20,
        )

        # Remove y-axis ticks since we have labels on the side
        ax3.set_yticks([])

        # Extend x-axis slightly for labels
        x_range_extended = x_max - x_min
        ax3.set_xlim(
            x_min - x_range_extended * 0.15, x_max + x_range_extended * 0.05
        )
        ax3.set_ylim(
            -ridge_spacing * 0.5,
            (len(delta_activations_by_time)) * ridge_spacing,
        )

        # Add grid
        ax3.grid(True, linestyle="--", alpha=0.3, axis="x")

        # Add colorbar to show time progression
        sm = plt.cm.ScalarMappable(
            cmap="viridis",
            norm=plt.Normalize(
                vmin=min(analysis_diffusion_times_rev),
                vmax=max(analysis_diffusion_times_rev),
            ),
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax3, shrink=0.8, pad=0.02)
        cbar.set_label("Diffusion Time", fontsize=12)

        fig3.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "3_activation_density_ridges.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig3)

        ridge_data = []
        inactive_counts = []
        total_counts = []

        selected_indices = list(range(0, len(analysis_range), 4))
        if len(analysis_range) - 1 not in selected_indices:
            selected_indices.append(
                len(analysis_range) - 1
            )  # Always include the last one

        for idx in selected_indices:
            i = idx
            t = list(analysis_range)[i]
            orig_t = activations_orig[t]
            int_t = activations_int[t]
            delta_t = int_t - orig_t

            # Filter out features that were zero both before and after
            active_mask = ~((np.abs(orig_t) < 1e-8) & (np.abs(int_t) < 1e-8))
            delta_t_active = delta_t[active_mask]

            # Count inactive features
            n_total = len(delta_t)
            n_inactive = n_total - np.sum(active_mask)
            inactive_counts.append(n_inactive)
            total_counts.append(n_total)

            time_label = f"t={analysis_diffusion_times[i]:.3f}"
            for val in delta_t_active:
                ridge_data.append(
                    {
                        "activation_change": val,
                        "timestep": time_label,
                        "diffusion_time": analysis_diffusion_times[i],
                    }
                )

        # Save inactive feature statistics for selected timepoints
        inactive_stats_path = os.path.join(
            output_dir, "inactive_feature_stats.txt"
        )
        with open(inactive_stats_path, "w") as f:
            f.write(
                "Inactive Feature Statistics (zero before AND after intervention)\n"  # noqa: E501
            )
            f.write("=" * 60 + "\n")
            f.write(
                f"{'Timestep':<12} {'Total':<8} {'Inactive':<10} {'Active':<8} {'Inactive %':<12}\n"  # noqa: E501
            )
            f.write("-" * 60 + "\n")
            for idx, (total, inactive) in enumerate(
                zip(total_counts, inactive_counts, strict=False)
            ):
                active = total - inactive
                inactive_pct = (inactive / total * 100) if total > 0 else 0
                i = selected_indices[idx]
                time_label = f"t={analysis_diffusion_times[i]:.3f}"
                f.write(
                    f"{time_label:<12} {total:<8} {inactive:<10} {active:<8} {inactive_pct:<12.1f}\n"  # noqa: E501
                )

        # Create ridge plot
        df = pd.DataFrame(ridge_data)
        df = df.sort_values("diffusion_time")

        # get nubmer of activation that are approx 0
        num_approx_zero = np.sum(np.abs(df["activation_change"]) < 1e-8)

        logger.info(
            f"Number of approx. zero activations: {num_approx_zero}/{len(df)} "
            f"min: {df['activation_change'].min()}, "
            f"max: {df['activation_change'].max()})"
        )

        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        n_times = len(df["timestep"].unique())
        pal = sns.cubehelix_palette(n_times, rot=-0.25, light=0.7)

        g = sns.FacetGrid(
            df,
            row="timestep",
            hue="timestep",
            aspect=5,
            height=1.5,
            palette=pal,
        )
        g.map(
            sns.kdeplot,
            "activation_change",
            bw_adjust=0.5,
            clip_on=False,
            fill=True,
            alpha=1,
            linewidth=1.5,
        )
        g.map(
            sns.kdeplot,
            "activation_change",
            clip_on=False,
            color="w",
            lw=2,
            bw_adjust=0.5,
        )

        # Add labels
        def label(x, color, label):
            ax = plt.gca()
            ax.text(
                0,
                0.2,
                label,
                fontweight="bold",
                color=color,
                ha="left",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )

        g.map(label, "activation_change")

        xlim_lower = df["activation_change"].quantile(0.01)
        xlim_upper = df["activation_change"].quantile(0.99)
        g.set(xlim=(xlim_lower, xlim_upper))

        g.figure.subplots_adjust(hspace=-0.25)
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)

        g.set_xlabels("Change in Activation", fontsize=10)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "activation_ridges.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()


if __name__ == "__main__":
    main()
