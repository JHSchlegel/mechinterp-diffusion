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

        assert self.config.hook_type != "reconstruct" or not (
            self.timesteps or self.timestep_values
        ), (
            "Reconstruction hooks do not support timesteps or timestep values."
            " Please set `timesteps` and `timestep_values` to None."
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
        self.sae_type = config.sae_path.split("_")[0].split("/")[-1]
        logging.info(f"SAE type: {self.sae_type}")

        self.sae_model = TopKSAE.load_from_disk(
            self.config.sae_path,
            config_class=TopKSAEConfig,
            device=self.device,
        ).to(dtype=self.torch_dtype)

        self.pipe = HookedStableDiffusionPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=self.torch_dtype,
            safety_checker=None,
        ).to(self.device)

        self.pipe.scheduler.set_timesteps(
            self.config.num_inference_steps, device="cpu"
        )
        self.scheduler_timesteps = self.pipe.scheduler.timesteps.to(
            self.device
        )

        self.scheduler_timesteps = self.pipe.scheduler.timesteps.to(
            self.device
        )

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
        current_step = [0]

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
                        prompts=[prompt],
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
                    prompts=[prompt],
                    timesteps=self.timesteps,
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
            feature_idx=self.config.features[0],
            timesteps=capture_steps,
        )

    def _visualize_results(
        self,
        original_images: List[Image.Image],
        intervened_images: List[Image.Image],
        save_path: str,
        prompts: List[str],
        timesteps: Optional[List[int]] = None,
    ) -> None:
        """Visualize the results of intervention compared to original images.

        Args:
            original_images (List[Images.Image]): Original generated images.
            intervened_images (List[Image.Image]): Images generated with
                intervention
            save_path (str): Path to save visualization to.
            prompts (List[str]): Prompts used to generate images.
            timesteps (Optional[List[int]]): Timesteps at which intervention
                was applied.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes = axes.reshape(1, -1)

        ## Original image:
        orig_img = original_images[0]
        orig_array = np.array(orig_img)

        axes[0, 0].imshow(orig_array)
        axes[0, 0].set_title("Original")
        axes[0, 0].axis("off")

        ## Intervened image:
        interv_img = intervened_images[0]
        interv_array = np.array(interv_img)

        axes[0, 1].imshow(interv_array)
        axes[0, 1].set_title("Intervened")
        axes[0, 1].axis("off")

        # Calculate and plot absolute difference
        orig_array = orig_array.astype(np.float32)
        interv_array = interv_array.astype(np.float32)

        diff = np.abs(orig_array - interv_array)
        diff_max = diff.max()
        if diff_max > 0:
            diff = diff / diff_max

        axes[0, 2].imshow(diff, cmap="viridis")
        axes[0, 2].set_title("Difference")
        axes[0, 2].axis("off")

        plt.subplots_adjust(top=0.95)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

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
                ffontsize=14,
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
        feature_idx: int,
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
        https://github.com/surkovv/sdxl-unbox/blob/d5e383fea440aed59d533062f3d8f8435c9a3737/app.py

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
            title_text = str(val) if isinstance(val, str) else f"β={val:.2f}"
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
        active at each timestep.

        Args:
            prompts (List[str]): List of prompts to analyze
        """
        pass


if __name__ == "__main__":
    main()
