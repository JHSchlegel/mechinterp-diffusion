"""
SAE Performance Assessment Script

Example usage:
    - Reconstruction: python evaluate_sae.py --mode reconstruction
    - Feature Removal: python evaluate_sae.py --mode feature_removal
"""

# =========================================================================== #
#                           Packages and Presets                              #
# =========================================================================== #

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import lpips
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from umap import UMAP

matplotlib.use("Agg")
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from icecream import ic
from simple_parsing import Serializable, parse
from torch import Tensor
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import TopKSAEConfig
from core.diffusion.hooked_sd_pipeline import HookedStableDiffusionPipeline
from core.sae.metrics import explained_variance
from core.sae.topk_sae import TopKSAE
from core.utils.analysis_utils import get_block_label
from core.utils.hooks import TimedHook
from core.utils.reproducibility import set_all_seeds

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
}

# =========================================================================== #
#                           Configuration                                     #
# =========================================================================== #


@dataclass
class SAEAssessmentConfig(Serializable):
    """Configuration for SAE performance assessment."""

    # SAE and model configuration
    sae_paths: List[str] = field(
        default_factory=lambda: [
            "../../checkpoints/sae/down_blocks.2.attentions.0/TopKSAE_dsae-5120_timesteps-all_20250816_083716/step_488282",  # noqa: E501
            "../../checkpoints/sae/up_blocks.1.attentions.1/TopKSAE_dsae-5120_timesteps-all_20250815_224124/step_488282",  # noqa: E501
        ]
    )
    """Paths to SAE model checkpoints"""

    target_modules: List[str] = field(
        default_factory=lambda: [
            "unet.down_blocks.2.attentions.0",
            "unet.up_blocks.1.attentions.1",
        ]
    )
    """Target modules to evaluate (e.g., 'unet.down_blocks.2.attentions.0')"""

    mode: Literal["reconstruction", "feature_removal"] = "reconstruction"
    """Evaluation mode: 'reconstruction' or 'feature_removal'."""

    # Dataset configuration
    dataset_path: str = "../../data/prompts/laion-coco_captions"
    """Path to HuggingFace prompt dataset"""

    dataset_split: str = "test"
    """Dataset split to use"""

    prompt_column: str = "caption"
    """Column name containing prompts"""

    num_prompts: int = 100
    """Number of prompts to evaluate"""

    # Evaluation configuration
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1337])
    """Random seeds for evaluation"""

    timestep_intervals: List[str] = field(
        default_factory=lambda: ["0-4", "10-14", "20-24"]
    )
    """Timestep intervals for feature removal (format: 'start-end')"""

    # Model configuration
    model_id: str = "stabilityai/stable-diffusion-2-1"
    """Hugging Face model ID"""

    torch_dtype: str = "float16"
    """Torch data type (float16 or float32)"""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """Device to run evaluation on"""

    num_inference_steps: int = 25
    """Number of diffusion steps"""

    guidance_scale: float = 9.0
    """Classifier-free guidance scale"""

    height: int = 512
    """Height of the generated images"""

    width: int = 512
    """Width of the generated images"""

    # Output configuration
    output_base_dir: str = "../../results/sae_evaluation"
    """Base output directory for results"""

    save_images: bool = False
    """Whether to save generated images"""

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.torch_dtype not in ["float16", "float32"]:
            raise ValueError(f"Invalid torch_dtype: {self.torch_dtype}")

        if not self.sae_paths:
            raise ValueError("At least one SAE path must be provided")

        if not self.target_modules:
            raise ValueError("At least one target module must be provided")


# =========================================================================== #
#                              Helper Functions                               #
# =========================================================================== #
def compute_manhattan_distance(
    img1: Tensor, img2: Tensor
) -> Tuple[float, float]:
    """compute pixelwise manhattan distance between two images.

    Args:
        img1 (Tensor): First input image.
        img2 (Tensor): Second input image.

    Returns:
        Tuple[float, float]: Mean and median Manhattan distance.
    """
    diff = torch.abs(img1 - img2)
    mean_dist = diff.mean().item()
    median_dist = diff.median().item()
    return mean_dist, median_dist


def load_prompts(
    dataset_path: str, dataset_split: str, prompt_column: str, num_prompts: int
) -> List[str]:
    """Load prompts from dataset.

    Args:
        dataset_path (str): Path to the dataset
        dataset_split (str): split to use i.e. 'test'
        prompt_column (str): Column name for prompts
        num_prompts (int): Number of prompts to load

    Returns:
        List[str]: List of loaded prompts
    """
    dataset = load_from_disk(dataset_path)
    split_data = dataset[dataset_split]
    logger.info(f"Loaded dataset with {len(split_data)} examples")

    prompts = []
    for i in range(min(num_prompts, len(split_data))):
        prompt_entry = split_data[i]
        prompts.append(prompt_entry[prompt_column])

    return prompts


def create_output_directory(config: SAEAssessmentConfig) -> str:
    """Create output directory for results.

    Args:
        config (SAEAssessmentConfig): Configuration object

    Returns:
        str: Path to the created output directory
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_dir = (
        Path(config.output_base_dir) / config.mode / f"assessment_{timestamp}"
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir)


# =========================================================================== #
#                               Main Function                                 #
# =========================================================================== #


def main():
    config = parse(SAEAssessmentConfig)

    # Load prompts
    logger.info(f"Loading prompts from {config.dataset_path}")
    prompts = load_prompts(
        config.dataset_path,
        config.dataset_split,
        config.prompt_column,
        config.num_prompts,
    )
    logger.info(f"Loaded {len(prompts)} prompts")

    # Initialize assessment
    assessment = SAEPerformanceAssessment(config)

    # Run assessment
    results_df = assessment.run_full_assessment(prompts)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SAE PERFORMANCE ASSESSMENT COMPLETE")
    logger.info("=" * 60)

    # Per-module summary
    logger.info("\nPER-MODULE SUMMARY:")
    for module in config.target_modules:
        module_data = results_df[results_df["target_module"] == module]
        logger.info(f"\n  {module}:")
        logger.info(
            f"    LPIPS: {module_data['reconstruction_lpips'].mean():.4f} "
            f"± {module_data['reconstruction_lpips'].std():.4f}"
        )
        logger.info(
            f"    R²: {module_data['reconstruction_r2'].mean():.4f} "
            f"± {module_data['reconstruction_r2'].std():.4f}"
        )
        logger.info(
            f"    Dead Features: {module_data['dead_features_pct'].mean():.2%}"
        )


# =========================================================================== #
#                        Feature Removal Hook                                 #
# =========================================================================== #


class FeatureKnockoutHook:
    """
    Hook that removes all SAE features during specified timestep intervals.
    """

    def __init__(
        self, sae: TopKSAE, total_steps: int, removal_interval: Tuple[int, int]
    ):
        """
        Initialize feature removal hook.

        Args:
            sae (TopKSAE): Sparse autoencoder model
            total_steps (int): Total number of diffusion steps
            removal_interval Tuple[int, int]: (start_step, end_step) interval
                for feature removal
        """
        self.sae = sae
        self.total_steps = total_steps
        self.removal_interval = removal_interval
        self.current_step = 0

    def __call__(self, module, input, output):
        """Apply feature removal if within specified interval."""
        if (
            self.removal_interval[0]
            <= self.current_step
            <= self.removal_interval[1]
        ):
            diff = (
                (output[0] - input[0])
                .permute((0, 2, 3, 1))
                .to(self.sae.device)
            )
            _, diff_cond = diff.chunk(2)
            bs, h, w, c = diff_cond.shape
            diff_cond = diff_cond.reshape(bs * h * w, c)
            sae_input, info = self.sae.preprocess_input(diff_cond)
            ic(sae_input.shape)
            activations: Tensor = F.relu(self.sae.encode(sae_input))
            top_acts, _ = self.sae._get_topk(activations, k=self.sae.cfg.k)
            ic(top_acts.squeeze(-1).shape)
            ic(self.sae.W_dec.weight.shape)
            to_add: Tensor = self.sae.postprocess_output(
                top_acts.squeeze(-1) @ self.sae.W_dec.weight.T, info
            )

            output_tensor = output[0]

            to_add_permuted = to_add.permute(0, 3, 1, 2).to(
                output_tensor.device
            )

            uncond = -torch.ones(
                1, device=output_tensor.device, dtype=output_tensor.dtype
            )
            cond = torch.ones(
                1, device=output_tensor.device, dtype=output_tensor.dtype
            )
            multiplier = torch.cat([uncond, cond]).view(-1, 1, 1, 1)
            ic(to_add.shape)
            ic(top_acts.shape)

            # Subtract for knockout
            modified_output = output_tensor - to_add_permuted * multiplier
            return (modified_output,)

        else:
            # Normal pass-through
            result = output

        # Increment step counter
        self.current_step = (self.current_step + 1) % self.total_steps
        return result


# =========================================================================== #
#                           SAE Assessment Class                              #
# =========================================================================== #


class SAEPerformanceAssessment:

    def __init__(self, config: SAEAssessmentConfig):
        """
        Initialize SAE performance assessment.

        Args:
            config (SAEAssessmentConfig): Assessment configuration
        """
        self.config = config
        self.device = config.device
        self.torch_dtype = DTYPE_MAP.get(config.torch_dtype, torch.float32)

        # Initialize LPIPS loss
        self.lpips_loss = lpips.LPIPS(net="alex").to(self.device)

        self.pipeline = HookedStableDiffusionPipeline.from_pretrained(
            config.model_id, torch_dtype=self.torch_dtype, safety_checker=None
        ).to(self.device)

        self.pipeline.scheduler.set_timesteps(
            config.num_inference_steps, device="cpu"
        )

        self.results = []

    def _locate_module(self, position: str) -> nn.Module:
        """Locate a module in the pipeline by its position string."""
        block = self.pipeline
        for step in position.split("."):
            if step.isdigit():
                block = block[int(step)]
            else:
                block = getattr(block, step)
        return block

    @torch.no_grad()
    def evaluate_reconstruction(
        self, sae_path: str, target_module: str, prompts: List[str], seed: int
    ) -> Tuple[Dict[str, Any], TopKSAE]:
        """
        Evaluate reconstruction performance of an SAE without generation bias.

        Args:
            sae_path (str): Path to SAE checkpoint
            target_module (str): Module to apply SAE to
            prompts (List[str]): List of prompts for generation
            seed (int): Random seed

        Returns:
            - Dictionary containing evaluation metrics
            - Loaded TopKSAE
        """
        logger.info(
            f"Evaluating reconstruction - SAE: {Path(sae_path).name}, "
            f"Module: {target_module}, Seed: {seed}"
        )

        # Load SAE with proper device and dtype handling
        sae = TopKSAE.load_from_disk(
            sae_path, config_class=TopKSAEConfig, device="cpu"
        )
        sae = sae.to(device=self.device, dtype=self.torch_dtype)
        sae.eval()

        # Get dead features percentage
        feature_is_inactive = (
            sae.num_tokens_inactive >= sae.cfg.num_tokens_dead_threshold
        )
        dead_features_pct = feature_is_inactive.float().mean().item()

        # Get target module
        module = self._locate_module(target_module)

        results = {
            "sae_path": sae_path,
            "target_module": target_module,
            "seed": seed,
            "dead_features_pct": dead_features_pct,
            "lpips_scores": [],
            "mean_manhattan": [],
            "median_manhattan": [],
            "r2_scores": [],
            "r2_scores_compound": [],  # For compound error evaluation
        }

        for prompt in tqdm(prompts, desc=f"Seed {seed}"):
            set_all_seeds(seed)

            activation_collector = []
            current_timestep = [0]

            def collection_hook(m, i, o):
                # Calculate and store the diff
                original_diff = (o[0] - i[0]).permute((0, 2, 3, 1))
                original_diff = original_diff.to(
                    device=self.device, dtype=self.torch_dtype
                )

                # Only use conditional part if cfg > 1
                if original_diff.shape[0] == 2:
                    _, original_diff = original_diff.chunk(2, dim=0)

                # Store activation data
                activation_collector.append(  # noqa: B023
                    {
                        "input": i[0].detach().cpu(),
                        "output": o[0].detach().cpu(),
                        "diff": original_diff.detach().cpu(),
                        "timestep": current_timestep[0],  # noqa: B023
                    }
                )

                current_timestep[0] = (  # noqa: B023
                    current_timestep[0] + 1  # noqa: B023
                ) % self.config.num_inference_steps

                return o  # Return UNCHANGED output

            handle = module.register_forward_hook(collection_hook)

            # Generate original image and collect activations
            original_image = self.pipeline(
                prompt=prompt,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                height=self.config.height,
                width=self.config.width,
                generator=torch.Generator(device=self.device).manual_seed(
                    seed
                ),
            ).images[0]

            handle.remove()

            all_original = []
            all_reconstructed = []
            timestep_r2 = {}

            for activation_data in activation_collector:
                original_diff = activation_data["diff"].to(
                    device=self.device, dtype=self.torch_dtype
                )
                _, H, W, C = original_diff.shape
                original_diff_flat = original_diff.view(H * W, C)

                sae_input, info = sae.preprocess_input(original_diff_flat)
                activations = torch.relu(sae.encode(sae_input))
                top_acts, _ = sae._get_topk(activations, k=sae.cfg.k)
                reconstructed = sae.decode(top_acts)

                all_original.append(sae_input.detach().cpu())
                all_reconstructed.append(reconstructed.detach().cpu())

                ts = activation_data["timestep"]
                timestep_r2[ts] = explained_variance(
                    x=sae_input.float(), x_reconstructed=reconstructed.float()
                )

            all_original = torch.cat(all_original).float()
            all_reconstructed = torch.cat(all_reconstructed).float()
            r2_score = explained_variance(
                x=all_original, x_reconstructed=all_reconstructed
            )

            current_timestep[0] = 0

            def reconstruction_hook(m, i, o):
                # Calculate diff
                original_diff = (o[0] - i[0]).permute((0, 2, 3, 1))
                original_diff = original_diff.to(
                    device=self.device, dtype=self.torch_dtype
                )

                if original_diff.shape[0] == 2:
                    _, original_diff = original_diff.chunk(2, dim=0)

                _, H, W, C = original_diff.shape
                original_diff_flat = original_diff.view(H * W, C)

                # Process through SAE
                sae_input, info = sae.preprocess_input(original_diff_flat)
                activations = torch.relu(sae.encode(sae_input))
                top_acts, _ = sae._get_topk(activations, k=sae.cfg.k)
                reconstructed = sae.decode(top_acts)
                sae_output = sae.postprocess_output(reconstructed, info)

                # Reshape and create reconstructed output
                reconstructed_diff = sae_output.view(1, H, W, C).permute(
                    0, 3, 1, 2
                )
                reconstructed_output = i[0] + reconstructed_diff.to(
                    device=i[0].device, dtype=i[0].dtype
                )

                current_timestep[0] = (  # noqa: B023
                    current_timestep[0] + 1  # noqa: B023
                ) % self.config.num_inference_steps

                return (reconstructed_output,)

            # Create timed hook for reconstruction
            timed_hook = TimedHook(
                reconstruction_hook,
                self.config.num_inference_steps,
                apply_at_steps=list(range(self.config.num_inference_steps)),
            )

            handle = module.register_forward_hook(timed_hook)

            set_all_seeds(seed)
            reconstructed_image = self.pipeline(
                prompt=prompt,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                height=self.config.height,
                width=self.config.width,
                generator=torch.Generator(device=self.device).manual_seed(
                    seed
                ),
            ).images[0]

            handle.remove()

            # Convert images to tensors
            original_tensor = (
                torch.from_numpy(np.array(original_image))
                .float()
                .permute(2, 0, 1)
                .unsqueeze(0)
                / 255.0
            )
            reconstructed_tensor = (
                torch.from_numpy(np.array(reconstructed_image))
                .float()
                .permute(2, 0, 1)
                .unsqueeze(0)
                / 255.0
            )

            original_tensor = original_tensor.to(self.device)
            reconstructed_tensor = reconstructed_tensor.to(self.device)

            # Compute metrics; LPIPS requires input in [-1, 1] range
            lpips_score = self.lpips_loss(
                original_tensor * 2 - 1, reconstructed_tensor * 2 - 1
            ).item()
            mean_manhattan, median_manhattan = compute_manhattan_distance(
                original_tensor, reconstructed_tensor
            )

            # Store results
            results["lpips_scores"].append(lpips_score)
            results["mean_manhattan"].append(mean_manhattan)
            results["median_manhattan"].append(median_manhattan)
            results["r2_scores"].append(r2_score)

            # Store per-timestep R²
            if "timestep_r2" not in results:
                results["timestep_r2"] = []
            results["timestep_r2"].append(timestep_r2)

        # Compute averages
        results["avg_lpips"] = np.mean(results["lpips_scores"])
        results["avg_mean_manhattan"] = np.mean(results["mean_manhattan"])
        results["avg_median_manhattan"] = np.mean(results["median_manhattan"])
        results["avg_r2"] = np.mean(results["r2_scores"])

        # Compute average per-timestep R²
        if "timestep_r2" in results:
            avg_timestep_r2 = {}
            for ts in range(self.config.num_inference_steps):
                ts_values = [
                    prompt_r2.get(ts, 0)
                    for prompt_r2 in results["timestep_r2"]
                ]
                avg_timestep_r2[ts] = np.mean(ts_values) if ts_values else 0
            results["avg_timestep_r2"] = avg_timestep_r2

        logger.info(
            f"Results - Dead features: {dead_features_pct:.2%}, "
            f"R² (unbiased): {results['avg_r2']:.4f}, "
            f"LPIPS (compound): {results['avg_lpips']:.4f}"
        )

        return results, sae

    @torch.no_grad()
    def evaluate_feature_removal(
        self,
        sae_path: str,
        target_module: str,
        prompts: List[str],
        seed: int,
        timestep_intervals: List[Tuple[int, int]],
    ) -> Tuple[Dict[str, Any], TopKSAE]:
        """
        Evaluate impact of removing all features during timestep intervals.

        Args:
            sae_path (str): Path to SAE checkpoint
            target_module (str): Module to apply SAE to
            prompts (List[str]): List of prompts for generation
            seed (int): Random seed
            timestep_intervals (List[Tuple[int, int]]): List of (start, end)
                timestep intervals

        Returns:
            - Dictionary containing evaluation metrics
            - Loaded TopKSAE
        """
        logger.info(
            f"Evaluating feature removal - Module: {target_module}, "
            f"Seed: {seed}"
        )

        # Load SAE with proper device and dtype handling
        sae = TopKSAE.load_from_disk(
            sae_path, config_class=TopKSAEConfig, device="cpu"
        )
        sae = sae.to(device=self.device, dtype=self.torch_dtype)
        sae.eval()

        # Get target module
        module = self._locate_module(target_module)

        results = {
            "sae_path": sae_path,
            "target_module": target_module,
            "seed": seed,
            "timestep_intervals": {},
        }

        for interval in timestep_intervals:
            interval_key = f"{interval[0]}-{interval[1]}"
            results["timestep_intervals"][interval_key] = {
                "lpips_scores": [],
                "mean_manhattan": [],
                "median_manhattan": [],
            }

            # Setup feature removal hook
            removal_hook = FeatureKnockoutHook(
                sae, self.config.num_inference_steps, interval
            )

            for prompt in tqdm(prompts, desc=f"Interval {interval_key}"):
                set_all_seeds(seed)

                # Generate original image
                original_image = self.pipeline(
                    prompt=prompt,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                    height=self.config.height,
                    width=self.config.width,
                    generator=torch.Generator(device=self.device).manual_seed(
                        seed
                    ),
                ).images[0]

                # Generate image with feature removal
                handle = module.register_forward_hook(removal_hook)
                removal_hook.current_step = 0  # Reset step counter

                set_all_seeds(seed)
                modified_image = self.pipeline(
                    prompt=prompt,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                    height=self.config.height,
                    width=self.config.width,
                    generator=torch.Generator(device=self.device).manual_seed(
                        seed
                    ),
                ).images[0]

                handle.remove()

                # Convert to tensors and compute metrics
                original_tensor = (
                    torch.from_numpy(np.array(original_image))
                    .float()
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    / 255.0
                )
                modified_tensor = (
                    torch.from_numpy(np.array(modified_image))
                    .float()
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    / 255.0
                )

                original_tensor = original_tensor.to(self.device)
                modified_tensor = modified_tensor.to(self.device)

                lpips_score = self.lpips_loss(
                    original_tensor * 2 - 1, modified_tensor * 2 - 1
                ).item()
                mean_manhattan, median_manhattan = compute_manhattan_distance(
                    original_tensor, modified_tensor
                )

                results["timestep_intervals"][interval_key][
                    "lpips_scores"
                ].append(lpips_score)
                results["timestep_intervals"][interval_key][
                    "mean_manhattan"
                ].append(mean_manhattan)
                results["timestep_intervals"][interval_key][
                    "median_manhattan"
                ].append(median_manhattan)

            # Compute averages for this interval
            interval_results = results["timestep_intervals"][interval_key]
            interval_results["avg_lpips"] = np.mean(
                interval_results["lpips_scores"]
            )
            interval_results["avg_mean_manhattan"] = np.mean(
                interval_results["mean_manhattan"]
            )
            interval_results["avg_median_manhattan"] = np.mean(
                interval_results["median_manhattan"]
            )

        return results, sae

    def run_full_assessment(self, prompts: List[str]) -> pd.DataFrame:
        """
        Run complete assessment across all SAEs, modules, and seeds.

        Args:
            prompts (List[str]): List of prompts for generation

        Returns:
            pd.DataFrame: DataFrame containing all results
        """
        logger.info("Starting comprehensive SAE performance assessment...")
        logger.info(
            f"SAEs: {len(self.config.sae_paths)}, "
            f"Modules: {len(self.config.target_modules)}, "
            f"Seeds: {len(self.config.seeds)}, Prompts: {len(prompts)}"
        )

        # Parse timestep intervals
        timestep_intervals = []
        for interval_str in self.config.timestep_intervals:
            start, end = map(int, interval_str.split("-"))
            timestep_intervals.append((start, end))

        # Create base output directory
        base_output_dir = create_output_directory(self.config)

        all_results = []

        for sae_path, target_module in zip(
            self.config.sae_paths, self.config.target_modules, strict=True
        ):
            sae_name = Path(sae_path).name

            # Create module-specific directory
            module_name = target_module.replace(".", "_")
            module_dir = Path(base_output_dir) / module_name / sae_name
            module_dir.mkdir(parents=True, exist_ok=True)

            # Save configuration for this module
            config_path = module_dir / "config.json"
            self.config.save(str(config_path))

            module_results = []

            for seed in self.config.seeds:
                combined_result = {
                    "sae_name": sae_name,
                    "sae_path": sae_path,
                    "target_module": target_module,
                    "seed": seed,
                }
                if self.config.mode == "reconstruction":
                    recon_results, sae = self.evaluate_reconstruction(
                        sae_path, target_module, prompts, seed
                    )

                    combined_result.update(
                        {
                            "dead_features_pct": recon_results[
                                "dead_features_pct"
                            ],
                            "reconstruction_lpips": recon_results["avg_lpips"],
                            "reconstruction_mean_manhattan": recon_results[
                                "avg_mean_manhattan"
                            ],
                            "reconstruction_median_manhattan": recon_results[
                                "avg_median_manhattan"
                            ],
                            "reconstruction_r2": recon_results["avg_r2"],
                            "avg_timestep_r2": recon_results.get(
                                "avg_timestep_r2", {}
                            ),
                        }
                    )
                else:
                    removal_results, sae = self.evaluate_feature_removal(
                        sae_path,
                        target_module,
                        prompts,
                        seed,
                        timestep_intervals,
                    )

                    for interval_key, interval_data in removal_results[
                        "timestep_intervals"
                    ].items():
                        combined_result[f"removal_{interval_key}_lpips"] = (
                            interval_data["avg_lpips"]
                        )
                        combined_result[
                            f"removal_{interval_key}_mean_manhattan"
                        ] = interval_data["avg_mean_manhattan"]
                        combined_result[
                            f"removal_{interval_key}_median_manhattan"
                        ] = interval_data["avg_median_manhattan"]

                module_results.append(combined_result)
                all_results.append(combined_result)

            # Save module-specific results
            module_df = pd.DataFrame(module_results)
            module_df.to_csv(module_dir / "results.csv", index=False)

            # Save detailed metrics for each seed
            for seed_result in module_results:
                seed_dir = module_dir / f"seed_{seed_result['seed']}"
                seed_dir.mkdir(exist_ok=True)

                with open(seed_dir / "metrics.json", "w") as f:
                    json.dump(seed_result, f, indent=2)

            self._plot_umap(sae, module_dir)

        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        self.results_df = results_df
        self.base_output_dir = str(base_output_dir)

        # Save overall results
        results_df.to_csv(
            Path(base_output_dir) / "all_results.csv", index=False
        )

        if self.config.mode == "reconstruction":
            # Create and save per-timestep R² plots
            self._plot_timestep_metrics(
                all_results, Path(self.base_output_dir)
            )

        # Save prompts used
        prompts_path = Path(base_output_dir) / "prompts.json"
        with open(prompts_path, "w") as f:
            json.dump({"prompts": prompts}, f, indent=2)

        return results_df

    def _plot_umap(self, sae: TopKSAE, output_dir: Path) -> None:
        """Plot UMAP embeddings of feature columns for the given SAE model.

        Args:
            sae (TopKSAE): The SAE model to visualize.
            output_dir (Path): The directory to save the UMAP plot.
        """

        # [1280, 5120] -> [1280, 2]
        W_dec = sae.W_dec.weight.detach().cpu().numpy()

        logger.info(
            f"Fitting UMAP for Decoder with weight shape: {W_dec.shape}"
        )
        reducer = UMAP(n_components=2, metric="cosine", random_state=42)

        embedding = reducer.fit_transform(W_dec)

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.scatter(
            embedding[:, 0], embedding[:, 1], c="#191970", s=10, alpha=0.6
        )

        # Labels
        ax.set_xlabel("UMAP 1", fontsize=12)
        ax.set_ylabel("UMAP 2", fontsize=12)
        ax.set_title("UMAP of SAE Features", fontsize=14)

        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)

        sns.despine()

        plt.tight_layout()

        output_path = Path(output_dir) / "decoder_umap.pdf"
        fig.savefig(output_path, dpi=300)
        plt.close()

        logger.info(f"Saved decoder UMAP to {output_path}")

    def _plot_timestep_metrics(
        self, results: List[Dict], output_dir: Path
    ) -> None:
        """Plot per-timestep R² metrics with nice styling.

        Args:
            results (List[Dict]): List of result dictionaries containing
                R² metrics.
            output_dir (Path): Directory to save the plots.
        """

        # Define nice colors (AAAS journal style)
        PAPER_COLORS = {
            "aaas": [
                "#3B4992",
                "#EE0000",
                "#008B45",
                "#631879",
                "#008280",
                "#BB0021",
            ]
        }
        colors = PAPER_COLORS["aaas"]

        # Collect timestep R² data with proper block labels
        timestep_data = {}
        block_labels = {}

        for result in results:
            if "avg_timestep_r2" in result and result["avg_timestep_r2"]:
                module = result["target_module"]
                sae_path = result["sae_path"]

                # Get proper block label using YOUR function
                block_label = get_block_label(Path(sae_path))
                block_labels[module] = block_label

                if module not in timestep_data:
                    timestep_data[module] = {}
                for ts, r2 in result["avg_timestep_r2"].items():
                    # Convert timestep to diffusion time
                    # (1.0 = noisy, 0.0 = clean)
                    diffusion_time = 1.0 - (
                        int(ts) / (self.config.num_inference_steps - 1)
                    )
                    if diffusion_time not in timestep_data[module]:
                        timestep_data[module][diffusion_time] = []
                    timestep_data[module][diffusion_time].append(r2)

        # Create plot with larger figure size
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.tick_params(
            axis="both", which="major", labelsize=18, width=2, length=8
        )

        # Make tick labels bold
        for label in ax.get_xticklabels():
            label.set_fontsize(18)
            label.set_fontweight("bold")

        for label in ax.get_yticklabels():
            label.set_fontsize(18)
            label.set_fontweight("bold")

        # Set font properties globally for this plot
        plt.rcParams.update(
            {
                "font.size": 24,
                "font.weight": "bold",
                "axes.labelweight": "bold",
                "axes.titleweight": "bold",
                "legend.fontsize": 24,
                "legend.title_fontsize": 26,
            }
        )

        for idx, (module, time_dict) in enumerate(timestep_data.items()):
            diffusion_times = sorted(
                time_dict.keys(), reverse=True
            )  # Sort from 1.0 to 0.0
            means = [np.mean(time_dict[t]) for t in diffusion_times]
            stds = [np.std(time_dict[t]) for t in diffusion_times]

            # Use block label for legend
            label = block_labels.get(module, module)
            color = colors[idx % len(colors)]

            ax.plot(
                diffusion_times,
                means,
                marker="o",
                markersize=8,
                linewidth=3,
                label=label,
                color=color,
            )

            # Add shaded std region
            ax.fill_between(
                diffusion_times,
                [m - s for m, s in zip(means, stds, strict=False)],
                [m + s for m, s in zip(means, stds, strict=False)],
                alpha=0.2,
                linewidth=0,
                color=color,
            )

        ax.set_xlabel(
            "Diffusion Time",
            fontsize=26,
            fontweight="bold",
        )
        ax.set_ylabel("R² Score", fontsize=26, fontweight="bold")

        ax.set_xlim(1.0, 0.0)  # Reverse x-axis: 1.0 (noisy) to 0.0 (clean)
        ax.set_ylim(0, 1)

        ax.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.4)
        ax.set_axisbelow(True)  # Put grid behind plot elements

        # Legend with nice formatting
        legend = ax.legend(
            loc="best",
            frameon=True,
            fancybox=True,
            shadow=True,
            borderpad=1,
            columnspacing=1.5,
            prop={"weight": "bold", "size": 20},
        )
        legend.get_title().set_fontweight("bold")
        legend.get_title().set_fontsize(22)

        # Make tick labels bold
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

        # Thicker axes lines
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        plt.tight_layout()

        # Save with high DPI
        plt.savefig(
            output_dir / "timestep_r2_plot.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(
            output_dir / "timestep_r2_plot.pdf",
            dpi=300,
            bbox_inches="tight",
            format="pdf",
        )
        plt.close()

        # Reset rcParams to defaults
        plt.rcParams.update(plt.rcParamsDefault)

        # Save timestep data to CSV with block labels and diffusion time
        timestep_df_data = []
        for module, time_dict in timestep_data.items():
            block_label = block_labels.get(module, module)
            for diff_time, values in time_dict.items():
                timestep_df_data.append(
                    {
                        "module": module,
                        "block_label": block_label,
                        "diffusion_time": diff_time,
                        "mean_r2": np.mean(values),
                        "std_r2": np.std(values),
                        "n_samples": len(values),
                    }
                )

        timestep_df = pd.DataFrame(timestep_df_data)
        timestep_df.to_csv(output_dir / "timestep_metrics.csv", index=False)
        logger.info(f"Saved timestep metrics plot and data to {output_dir}")


if __name__ == "__main__":
    main()
