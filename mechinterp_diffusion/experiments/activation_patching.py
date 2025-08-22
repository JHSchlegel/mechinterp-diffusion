"""
Module for automated activation patching experiments on diffusion models.
"""

# =========================================================================== #
#                           Packages and Presets                              #
# =========================================================================== #

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from simple_parsing import Serializable, parse

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import TopKSAEConfig
from core.diffusion.hooked_sd_pipeline import HookedStableDiffusionPipeline
from core.sae.topk_sae import TopKSAE
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
#                          Predefined Prompt Pairs                            #
# =========================================================================== #

PROMPT_PAIRS = [
    # {
    #     "source": "A headshot photo of a woman",
    #     "control": "A headshot photo of a man",
    # },
    # {
    #     "source": "A headshot photo of a man",
    #     "control": "A headshot photo of a woman",
    # },
    # {
    #     "source": "A high-resolution photo of a rose",
    #     "control": "A high-resolution photo of a dog",
    # },
    {
        "source": "A high-resolution photo of a cat.",
        "control": "A high-resolution photo of a bird.",
    },
    {
        "source": "A photo-realistic image of a bird.",
        "control": "A photo-realistic image of a cat.",
    },
    {
        "source": "A high-resolution photo of a dog.",
        "control": "A high-resolution photo of a cat.",
    },
    {
        "source": "A high-resolution photo of a dog.",
        "control": "A high-resolution photo of a man.",
    },
    {
        "source": "A crystal clear mountain lake.",
        "control": "A crystal clear ocean bay.",
    },
    {
        "source": "A gothic cathedral with tall spires.",
        "control": "A modern skyscraper with glass facade.",
    },
    {
        "source": " An ancient, gnarled oak tree in a forest.",
        "control": " A tall, slender palm tree on a tropical beach.",
    },
]

# =========================================================================== #
#                        Activation Patching Configuration                    #
# =========================================================================== #


@dataclass
class ActivationPatchingConfig(Serializable):
    """Configuration for activation patching experiments."""

    # -------------------------------------------------------------------------
    # Model configuration
    # -------------------------------------------------------------------------
    model_id: str = "stabilityai/stable-diffusion-2-1"
    """Hugging Face model identifier for the diffusion model."""

    sae_paths: List[str] = field(
        default_factory=lambda: [
            "../../checkpoints/sae/up_blocks.1.attentions.1/TopKSAE_dsae-5120_timesteps-all_20250815_224124/step_488282",  # noqa: E501
            "../../checkpoints/sae/down_blocks.2.attentions.0/TopKSAE_dsae-5120_timesteps-all_20250816_083716/step_488282",  # noqa: E501
        ]
    )
    """Paths to the trained TopKSAE model directories."""

    hook_names: List[str] = field(
        default_factory=lambda: [
            "unet.up_blocks.1.attentions.1",
            "unet.down_blocks.2.attentions.0",
        ]
    )
    """Names of the model components to hook for activation patching."""

    torch_dtype: str = "float16"
    """Torch data type for models ('float16' or 'float32')."""

    num_inference_steps: int = 25
    """Number of denoising steps for diffusion generation."""

    guidance_scale: float = 9.0
    """Classifier-free guidance scale for generation."""

    seed: int = 42
    """Random seed for reproducible results."""

    output_dir: str = "../../results/patching"
    """Directory to save experiment results and generated images."""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """Device to run computations on."""

    height: int = 768
    """Height of the generated images."""

    width: int = 768
    """Width of the generated images."""

    # -------------------------------------------------------------------------
    # Patching configuration
    # -------------------------------------------------------------------------
    patching_timestep_indices: Optional[List[int]] = None
    """
    Timestep indices to apply patching. If None, defaults to ALL inference
    steps.
    """

    patching_method: Literal["replace", "ablate_and_replace"] = (
        "ablate_and_replace"
    )
    """Patching method: 'replace' or 'ablate_and_replace'."""

    k_values: List[int] = field(default_factory=lambda: [1, 5, 20, 50, 200])
    """Grid of k values (number of top features to patch)."""

    patch_scale: float = 2.0
    """Factor to scale the patch vector by."""

    reconstruct_scale: float = 1.0
    """Scale factor for reconstruction strength. Will be subtracted."""

    # -------------------------------------------------------------------------
    # Experiment settings
    # -------------------------------------------------------------------------
    figure_dpi: int = 300
    """DPI for paper-ready figures."""

    def __post_init__(self):
        if self.patching_timestep_indices is None:
            self.patching_timestep_indices = list(
                range(self.num_inference_steps)
            )


# =========================================================================== #
#                              Main Function                                 #
# =========================================================================== #


def main():
    """
    Main function to run activation patching experiments with config.
    """
    config = parse(ActivationPatchingConfig)

    logger.info(f"Running experiments with k values: {config.k_values}")
    logger.info(f"Patching timesteps: {config.patching_timestep_indices}")

    dtype = DTYPE_MAP.get(config.torch_dtype, torch.float32)

    pipe = HookedStableDiffusionPipeline.from_pretrained(
        config.model_id, torch_dtype=dtype, safety_checker=None
    ).to(config.device)

    # Process each SAE/hook pair separately
    for sae_path, hook_name in zip(
        config.sae_paths, config.hook_names, strict=False
    ):
        logger.info(f"{'='*80}")
        logger.info(f"Processing SAE: {sae_path}")
        logger.info(f"Hook: {hook_name}")
        logger.info(f"{'='*80}")

        sae = TopKSAE.load_from_disk(
            sae_path, config_class=TopKSAEConfig, device=config.device
        ).to(dtype=dtype)
        sae.eval()

        for pair_idx in range(len(PROMPT_PAIRS)):
            prompt_pair = PROMPT_PAIRS[pair_idx]
            source_prompt = prompt_pair["source"]
            control_prompt = prompt_pair["control"]

            logger.info(f"{'-'*80}")
            logger.info(f"Prompt Pair {pair_idx + 1}/{len(PROMPT_PAIRS)}")
            logger.info(f"Source: {source_prompt[:50]}...")
            logger.info(f"Control: {control_prompt[:50]}...")
            logger.info(f"{'-'*80}")

            run_experiment_with_k_grid(
                source_prompt,
                control_prompt,
                config,
                pipe,
                sae,
                hook_name,
                height=config.height,
                width=config.width,
            )

    logger.info("All experiments complete!")


# =========================================================================== #
#                          Feature Analysis Functions                         #
# =========================================================================== #


def find_top_k_features(
    pipe: HookedStableDiffusionPipeline,
    sae: TopKSAE,
    source_prompt: str,
    control_prompt: str,
    timestep_indices: List[int],
    k: int,
    config: ActivationPatchingConfig,
    hook_name: str,
    height: int = 512,
    width: int = 512,
) -> List[int]:
    """
    Find top-k differentially active features between source and control
    prompts.

    Args:
        pipe (HookedStableDiffusionPipeline): The diffusion model pipeline.
        sae (TopKSAE): The sparse autoencoder for feature analysis.
        source_prompt (str): The source prompt to analyze.
        control_prompt (str): The control prompt to compare against.
        timestep_indices (List[int]): Timestep indices to analyze.
        k (int): Number of top features to return.
        config (ActivationPatchingConfig): Configuration object containing
            experiment parameters.
        height (int, optional): Height of the input images. Defaults to 512.
        width (int, optional): Width of the input images. Defaults to 512.

    Returns:
        List[int]: Indices of the top-k most differentially active features.
    """
    logger.info(f"Finding top {k} differentially active features:")

    def get_mean_activations(prompt: str) -> torch.Tensor:
        """
        Get mean SAE feature activations for a given prompt across timesteps.

        Args:
            prompt (str): The prompt to analyze.

        Returns:
            torch.Tensor: Mean activation values across all analyzed timesteps.
        """
        generator = torch.Generator(config.device).manual_seed(config.seed)

        # Run inference and cache activations
        _, cache = pipe.run_with_cache(
            prompt=prompt,
            positions_to_cache=[hook_name],
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            generator=generator,
            save_input=True,
            save_output=True,
            height=height,
            width=width,
        )

        with torch.no_grad():
            all_feature_acts = []
            for t_idx in range(config.num_inference_steps):
                # Extract cached activations
                input_cached = cache["input"][hook_name][:, t_idx]
                output_cached = cache["output"][hook_name][:, t_idx]

                # Compute residual stream
                diff = (output_cached - input_cached).to(config.device)

                # Process through SAE
                sae_in, _ = sae.preprocess_input(diff)
                feature_acts = torch.relu(sae.encode(sae_in))
                top_acts, _ = sae._get_topk(feature_acts, k=sae.cfg.k)
                all_feature_acts.append(top_acts)

            # Average across timesteps and spatial dimensions
            mean_acts = torch.stack(all_feature_acts).mean(dim=(0, 1))

        return mean_acts

    # Get activations for both prompts
    s_src = get_mean_activations(source_prompt)
    s_ctrl = get_mean_activations(control_prompt)

    # Normalize to relative activations
    total_s_src = torch.sum(s_src)
    total_s_ctrl = torch.sum(s_ctrl)
    relative_s_src = s_src / (total_s_src + 1e-9)
    relative_s_ctrl = s_ctrl / (total_s_ctrl + 1e-9)

    # Compute gamma score (differential activation)
    gamma = relative_s_src - relative_s_ctrl
    _, top_k_indices = torch.topk(gamma, k)

    return top_k_indices.tolist()


def extract_source_patch_vectors(
    pipe: HookedStableDiffusionPipeline,
    sae: TopKSAE,
    source_prompt: str,
    causal_feature_indices: List[int],
    timestep_indices: List[int],
    config: ActivationPatchingConfig,
    hook_name: str,
    height: int = 512,
    width: int = 512,
) -> Tuple[Dict[int, torch.Tensor], List[Image.Image]]:
    """
    Extract patch vectors from source prompt for identified causal features.

    Args:
        pipe (HookedStableDiffusionPipeline): The diffusion model pipeline.
        sae (TopKSAE): The sparse autoencoder.
        source_prompt (str): Source prompt to extract features from.
        causal_feature_indices (List[int]): Indices of features to patch.
        timestep_indices (List[int]): Timesteps to extract patches for.
            config: Configuration object.
        height (int, optional): Height of the input images. Defaults to 512.
        width (int, optional): Width of the input images. Defaults to 512.

    Returns:
        Tuple[Dict[int, torch.Tensor], List[Image.Image]]: Patch vectors and
            source image.
    """
    logger.info("Extracting source patch vectors for causal features...")

    generator = torch.Generator(config.device).manual_seed(config.seed)
    source_img, source_cache = pipe.run_with_cache(
        prompt=source_prompt,
        positions_to_cache=[hook_name],
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        generator=generator,
        save_input=True,
        save_output=True,
        height=height,
        width=width,
    )

    source_patch_vectors = {}
    with torch.no_grad():
        for t_idx in timestep_indices:
            source_input_cached = source_cache["input"][hook_name][:, t_idx]
            source_output_cached = source_cache["output"][hook_name][:, t_idx]
            source_diff = (source_output_cached - source_input_cached).to(
                config.device
            )

            sae_in, _ = sae.preprocess_input(source_diff)
            feature_acts = torch.relu(sae.encode(sae_in))
            top_acts, _ = sae._get_topk(feature_acts, k=sae.cfg.k)

            causal_acts = torch.zeros_like(feature_acts)
            for f_idx in causal_feature_indices:
                causal_acts[:, f_idx] = top_acts[:, f_idx]

            reconstructed = sae.decode(causal_acts)
            reconstructed_post = sae.postprocess_output(reconstructed, {})

            B, HW, C = source_diff.shape
            H = W = int(np.sqrt(HW))
            source_patch_vectors[t_idx] = (
                reconstructed_post.reshape(B, HW, C)
                .permute(0, 2, 1)
                .reshape(B, C, H, W)
            )

    logger.info("Successfully extracted patch vectors")
    return source_patch_vectors, source_img


def create_patching_hook(
    source_patch_vectors: Dict[int, torch.Tensor],
    sae: TopKSAE,
    timestep_indices: List[int],
    patching_method: str,
    patch_scale: float,
    reconstruct_scale: float,
    run_type: str,
) -> Callable:
    """
    Create a hook function for activation patching during inference.

    Args:
        source_patch_vectors (Dict[int, torch.Tensor]):
            Precomputed patch vectors.
        sae (TopKSAE): The sparse autoencoder.
        timestep_indices (List[int]): Timesteps to apply patching.
        patching_method (str): Method for patching
            ('replace' or 'ablate_and_replace').
        patch_scale (float): Scale factor for patch strength.
        reconstruct_scale (float): Scale factor for reconstruction strength.
        run_type (str): Type of run ('control' or 'patched').

    Returns:
        Callable: Hook function for use with the pipeline.
    """
    hook_state = {"step_idx": 0}

    def hook_fn(
        module, in_tensor: torch.Tensor, out_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Hook function that applies activation patching during forward pass.

        Args:
            module: The hooked module (not used).
            in_tensor (torch.Tensor): Input tensor to the module.
            out_tensor (torch.Tensor): Output tensor from the module.

        Returns:
            torch.Tensor: Modified output tensor with patching applied.
        """
        t_idx = hook_state["step_idx"]

        if run_type == "patched" and t_idx in timestep_indices:
            out_tensor_val = out_tensor[0]
            source_patch_diff = source_patch_vectors[t_idx][0].to(
                out_tensor_val.dtype
            )
            pure_source_patch = source_patch_diff * patch_scale

            # Apply ablate-and-replace method if specified
            # i.e. subtract SAES reconstruction and add patching
            if patching_method == "ablate_and_replace":
                in_tensor_val = in_tensor[0]
                corrupted_diff = out_tensor_val - in_tensor_val
                B, C, H, W = corrupted_diff.shape

                with torch.no_grad():
                    # Process corrupted signal through SAE
                    corrupted_diff_permuted = corrupted_diff.permute(
                        0, 2, 3, 1
                    )
                    sae_in, _ = sae.preprocess_input(corrupted_diff_permuted)
                    feature_acts = torch.relu(sae.encode(sae_in))
                    top_acts, _ = sae._get_topk(feature_acts, k=sae.cfg.k)

                    # Reconstruct signal
                    reconstructed_corrupted_post = sae.postprocess_output(
                        sae.decode(top_acts), {}
                    )
                    HW = H * W
                    reconstructed_corrupted_diff = (
                        reconstructed_corrupted_post.reshape(B, HW, C)
                        .permute(0, 2, 1)
                        .reshape(B, C, H, W)
                    )

                # Subtract corrupted reconstruction before adding source patch
                out_tensor_val -= (
                    reconstruct_scale * reconstructed_corrupted_diff
                )

            # Apply patch differentially to conditional/unconditional passes
            if out_tensor_val.shape[0] == 2:  # Classifier-free guidance
                out_tensor_val[1] += pure_source_patch  # Conditional
                out_tensor_val[0] -= pure_source_patch  # Unconditional
            else:
                out_tensor_val += pure_source_patch

        hook_state["step_idx"] += 1
        return out_tensor

    return hook_fn


# =========================================================================== #
#                            Experiment Functions                            #
# =========================================================================== #


def run_experiment_with_k_grid(
    source_prompt: str,
    control_prompt: str,
    config: ActivationPatchingConfig,
    pipe: HookedStableDiffusionPipeline,
    sae: TopKSAE,
    hook_name: str,
    height: int = 512,
    width: int = 512,
) -> Path:
    """
    Run activation patching experiment with grid of k values.

    Args:
        source_prompt (str): Prompt containing features to transfer from.
        control_prompt (str): Control prompt to transfer features to.
        config (ActivationPatchingConfig): Configuration object.
        pipe (HookedStableDiffusionPipeline): The diffusion model pipeline.
        sae (TopKSAE): The sparse autoencoder.

    Returns:
        Path: Directory where results were saved.
    """
    set_all_seeds(config.seed)

    block_name = hook_name.replace("unet.", "").replace(".", "_")

    words = (
        source_prompt.lower().replace(",", "").replace(".", "").split()[:10]
    )
    prompt_slug = "_".join(words)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_folder = f"{prompt_slug}"

    # Timestep folder naming
    if len(config.patching_timestep_indices) == config.num_inference_steps:
        ts_folder = "all_timesteps"
    else:
        min_ts = min(config.patching_timestep_indices)
        max_ts = max(config.patching_timestep_indices)
        ts_folder = f"timesteps_{min_ts}-{max_ts}"

    base_save_dir = (
        Path(config.output_dir)
        / block_name
        / prompt_folder
        / ts_folder
        / f"{timestamp}"
    )
    base_save_dir.mkdir(parents=True, exist_ok=True)

    # Save config:
    with open(base_save_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=4)

    # Generate source image once
    generator = torch.Generator(config.device).manual_seed(config.seed)
    source_img = pipe(
        prompt=source_prompt,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        generator=generator,
        height=height,
        width=width,
    ).images
    source_img[0].save(base_save_dir / "source_image.png")

    # Generate control image once
    generator = torch.Generator(config.device).manual_seed(config.seed)
    control_img = pipe(
        prompt=control_prompt,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        generator=generator,
        height=height,
        width=width,
    ).images
    control_img[0].save(base_save_dir / "control_image.png")

    # Store results for each k value
    results = {
        "source_img": source_img[0],
        "control_img": control_img[0],
        "k_results": {},
    }

    # Find features for the highest k value once
    max_k = max(config.k_values)
    logger.info(
        f"Finding top {max_k} features (will subset for smaller k values)..."
    )
    all_causal_features = find_top_k_features(
        pipe,
        sae,
        source_prompt,
        control_prompt,
        config.patching_timestep_indices,
        max_k,
        config,
        hook_name,
        height=height,
        width=width,
    )

    # Run experiments for each k value by subsetting features
    for k in sorted(config.k_values):
        logger.info(f"Running patching with k={k}...")

        causal_feature_indices = all_causal_features[:k]

        # Extract clean patch vectors for these features
        source_patch_vectors, _ = extract_source_patch_vectors(
            pipe,
            sae,
            source_prompt,
            causal_feature_indices,
            config.patching_timestep_indices,
            config,
            hook_name,
            height=height,
            width=width,
        )

        # Run patched generation
        hook_fn = create_patching_hook(
            source_patch_vectors,
            sae,
            config.patching_timestep_indices,
            config.patching_method,
            config.patch_scale,
            config.reconstruct_scale,
            "patched",
        )

        generator = torch.Generator(config.device).manual_seed(config.seed)
        patched_images = pipe.run_with_hooks(
            prompt=control_prompt,
            position_hook_dict={hook_name: hook_fn},
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            generator=generator,
            height=height,
            width=width,
        )

        # Save individual result
        k_dir = base_save_dir / f"{k}_features"
        k_dir.mkdir(exist_ok=True)
        patched_images[0].save(k_dir / "patched_image.png")

        results["k_results"][k] = patched_images[0]

        # Save experiment details
        with open(k_dir / "details.txt", "w") as f:
            f.write(f"Source: {source_prompt}\n")
            f.write(f"Control: {control_prompt}\n")
            f.write(f"K value: {k}\n")
            f.write(f"Features: {causal_feature_indices}\n")

    # Create paper-ready figure if requested
    create_paper_figure(
        results, config.k_values, base_save_dir, config.figure_dpi
    )

    logger.info(f"Results saved to: {base_save_dir}")
    return base_save_dir


def create_paper_figure(
    results: Dict, k_values: List[int], save_dir: Path, dpi: int = 300
) -> None:
    """
    Create paper-ready figure with control on left, source on right,
    and k variations in between (lowest k to highest k).

    Args:
        results (Dict[Any]): Dictionary containing images for each k value.
        k_values (List[int]): List of k values used.
        save_dir (Path): Directory to save the figure.
        dpi (int): DPI for the figure.
    """
    # Sort k values to ensure proper ordering
    sorted_k = sorted(k_values)

    # Create figure with appropriate size
    n_images = len(sorted_k) + 2  # +2 for control and source
    fig_width = 3 * n_images  # 3 inches per image
    fig, axes = plt.subplots(1, n_images, figsize=(fig_width, 3.5))

    # Control image on the left
    axes[0].imshow(results["control_img"])
    axes[0].set_title("Control", fontsize=28, fontweight="bold")
    axes[0].axis("off")

    # K variations in the middle (lowest to highest)
    for i, k in enumerate(sorted_k, 1):
        axes[i].imshow(results["k_results"][k])
        axes[i].set_title(f"{k} Features", fontsize=28, fontweight="bold")
        axes[i].axis("off")

    # Source image on the right
    axes[-1].imshow(results["source_img"])
    axes[-1].set_title("Source", fontsize=28, fontweight="bold")
    axes[-1].axis("off")

    # Adjust layout and save
    plt.tight_layout(pad=0.5)
    plt.savefig(
        save_dir / "paper_figure.pdf",
        dpi=dpi,
        bbox_inches="tight",
        format="pdf",
    )
    plt.savefig(
        save_dir / "paper_figure.png",
        dpi=dpi,
        bbox_inches="tight",
        format="png",
    )
    plt.close()

    logger.info(f"Figure saved to {save_dir / 'paper_figure.pdf'}")


if __name__ == "__main__":
    main()
