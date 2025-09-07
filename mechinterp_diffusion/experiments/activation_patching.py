"""
Module for automated activation patching experiments on diffusion models.

Example Usage:
    # Add top-k features from source prompt to control prompt and subtract
    # top-k features from control prompt:
    python activation_patching.py --patching_mode add_source_subtract_control

    # Add top-k features from source prompt to control prompt only:
    python activation_patching.py --patching_mode add_source

    # First subtract entire SAE reconstruction of control prompt, then add
    # top-k features from source prompt:
    python activation_patching.py --patching_mode blate_and_add_source

Finally, by setting `--experiment_mode bidirectional`, the above patching
methods can be applied in both directions between a source and control prompt.
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
    {
        "source": "A bear in a forest.",
        "control": "A bear in the arctic.",
    },
    {
        "source": "A high-resolution photo of a cat.",
        "control": "A high-resolution photo of a bird.",
    },
    {
        "source": "A high-resolution photo of a dog.",
        "control": "A high-resolution photo of a cat.",
    },
    # {
    #     "source": "A high-resolution photo of a dog.",
    #     "control": "A high-resolution photo of a man.",
    # },
    # {
    #     "source": "A portrait of a man.",
    #     "control": "A portrait of a dog.",
    # },
    {
        "source": "A crystal clear mountain lake.",
        "control": "A crystal clear ocean bay.",
    },
    # {
    #     "source": "A castle on a sunny day.",
    #     "control": "A castle on a stormy day.",
    # },
    {
        "source": "A ship sailing on a calm ocean.",
        "control": "A ship sailing on a rough ocean.",
    },
    {
        "source": "An image of a tall cathedral.",
        "control": "An image of a tall glass skyscraper.",
    },
    # {
    #     "source": "A city street during the day.",
    #     "control": "A city street at night.",
    # },
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
            "../../checkpoints/sae/down_blocks.2.attentions.0/TopKSAE_dsae-5120_timesteps-all_20250816_083716/step_488282",  # noqa: E501
            "../../checkpoints/sae/up_blocks.1.attentions.1/TopKSAE_dsae-5120_timesteps-all_20250815_224124/step_488282",  # noqa: E501
        ]
    )
    """Paths to the trained TopKSAE model directories."""

    hook_names: List[str] = field(
        default_factory=lambda: [
            "unet.down_blocks.2.attentions.0",
            "unet.up_blocks.1.attentions.1",
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

    output_dir: str = "../../results/patching_add_subtract_768"
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

    patching_mode: Literal[
        "add_source",
        "add_source_subtract_control",
        "ablate_and_add_source",
    ] = "add_source_subtract_control"
    """
    Specifies the patching method:
    - 'add_source': Adds the top-k features from the source prompt.
    - 'add_source_subtract_control': Adds top-k source features and subtracts
        top-k control features.
    - 'ablate_and_add_source': Subtracts the SAE's entire reconstruction of the
        destination signal,
      then adds the top-k source features.
    """

    experiment_mode: Literal["standard", "bidirectional"] = "standard"
    """'standard' for one-way patching, 'bidirectional' for two-way."""

    k_values: List[int] = field(default_factory=lambda: [1, 5, 20, 50, 200])
    """Grid of k values (number of top features to patch)."""

    manual_source_indices: Optional[List[int]] = None
    """Manually specify source feature indices, bypassing k-grid search."""

    manual_control_indices: Optional[List[int]] = None
    """
    Manually specify control feature indices for subtraction in
    'add_source_subtract_control' mode.
    """

    add_scale: float = 2.0
    """Scale for adding features."""

    subtract_scale: float = 1.0
    """Scale for subtracting features."""

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

            if config.experiment_mode == "standard":
                run_standard_mode(
                    source_prompt,
                    control_prompt,
                    config,
                    pipe,
                    sae,
                    hook_name,
                    height=config.height,
                    width=config.width,
                )
            else:
                run_bidirectional_mode(
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
    k: int,
    config: ActivationPatchingConfig,
    hook_name: str,
    height: int = 512,
    width: int = 512,
) -> Tuple[List[int], List[int]]:
    """
    Find top-k differentially active features between source and control
    prompts.

    Args:
        pipe (HookedStableDiffusionPipeline): The diffusion model pipeline.
        sae (TopKSAE): The sparse autoencoder for feature analysis.
        source_prompt (str): The source prompt to analyze.
        control_prompt (str): The control prompt to compare against.
        k (int): Number of top features to return.
        config (ActivationPatchingConfig): Configuration object containing
            experiment parameters.
        height (int, optional): Height of the input images. Defaults to 512.
        width (int, optional): Width of the input images. Defaults to 512.

    Returns:
        Tuple[List[int], List[int], List[float], List[float]]:
        Indices and differentials of the top-k most differentially
            active features for source and control prompts.
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
    relative_s_src = s_src / (total_s_src + 1e-12)
    relative_s_ctrl = s_ctrl / (total_s_ctrl + 1e-12)

    # Compute gamma score (differential activation) form surkov et al.
    gamma = relative_s_src - relative_s_ctrl
    _, top_src_indices = torch.topk(gamma, k)
    _, top_ctrl_indices = torch.topk(-gamma, k)

    return top_src_indices.tolist(), top_ctrl_indices.tolist()


def extract_activations(
    pipe: HookedStableDiffusionPipeline,
    sae: TopKSAE,
    prompt: str,
    feature_indices: List[int],
    config: ActivationPatchingConfig,
    hook_name: str,
    height: int = 512,
    width: int = 512,
) -> Tuple[Dict[int, torch.Tensor]]:
    """
    Extract patch vectors from source prompt for identified causal features.

    Args:
        pipe (HookedStableDiffusionPipeline): The diffusion model pipeline.
        sae (TopKSAE): The sparse autoencoder.
        prompt (str): Prompt to extract features from.
        feature_indices (List[int]): Indices of features to cache activations
            for.
        config: Configuration object.
        height (int, optional): Height of the input images. Defaults to 512.
        width (int, optional): Width of the input images. Defaults to 512.

    Returns:
        Tuple[Dict[int, torch.Tensor]]: Patch vectors.
    """

    generator = torch.Generator(config.device).manual_seed(config.seed)
    img, cache = pipe.run_with_cache(
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

    activations = {}
    with torch.no_grad():
        for t_idx in config.patching_timestep_indices:
            input_cached = cache["input"][hook_name][:, t_idx]
            output_cached = cache["output"][hook_name][:, t_idx]
            diff = (output_cached - input_cached).to(config.device)

            sae_in, _ = sae.preprocess_input(diff)
            feature_acts = torch.relu(sae.encode(sae_in))
            top_acts, _ = sae._get_topk(feature_acts, k=sae.cfg.k)

            causal_acts = torch.zeros_like(feature_acts)
            causal_acts[:, feature_indices] = top_acts[:, feature_indices]

            reconstructed = causal_acts @ sae.W_dec.weight.T
            reconstructed_post = sae.postprocess_output(reconstructed, {})

            B, HW, C = diff.shape
            H = W = int(np.sqrt(HW))
            activations[t_idx] = (
                reconstructed_post.reshape(B, HW, C)
                .permute(0, 2, 1)
                .reshape(B, C, H, W)
            )

    return activations


# =========================================================================== #
#                           Hook Creation Functions                           #
# =========================================================================== #


def create_patching_hook(
    add_vectors: Dict[int, torch.Tensor],
    sae: TopKSAE,
    config: ActivationPatchingConfig,
    subtract_vectors: Optional[Dict[int, torch.Tensor]] = None,
) -> Callable:
    """
    Create a hook function for activation patching during inference.

    Args:
        source_patch_vectors (Dict[int, torch.Tensor]):
            Precomputed patch vectors.
        sae (TopKSAE): The sparse autoencoder.
        config (ActivationPatchingConfig): The configuration object.
        subtract_vectors (Optional[Dict[int, torch.Tensor]]): Precomputed patch
            vectors for subtraction.

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
        if t_idx in config.patching_timestep_indices:
            out_val = out_tensor[0]

            if config.patching_mode == "ablate_and_add_source":
                diff = out_val - in_tensor[0]
                B, C, H, W = diff.shape
                with torch.no_grad():
                    sae_in, _ = sae.preprocess_input(diff.permute(0, 2, 3, 1))
                    acts = torch.relu(sae.encode(sae_in))
                    top_acts, _ = sae._get_topk(acts, k=sae.cfg.k)
                    reconstructed = sae.postprocess_output(
                        sae.decode(top_acts), {}
                    )
                    reconstructed_diff = (
                        reconstructed.reshape(B, H * W, C)
                        .permute(0, 2, 1)
                        .reshape(B, C, H, W)
                    )
                out_val -= config.subtract_scale * reconstructed_diff

            add_patch = add_vectors[t_idx][0].to(out_val.dtype)
            sub_patch = 0
            if (
                config.patching_mode == "add_source_subtract_control"
                and subtract_vectors
            ):
                sub_patch = subtract_vectors[t_idx][0].to(out_val.dtype)

            patch_delta = (
                config.add_scale * add_patch
                - config.subtract_scale * sub_patch
            )
            if out_val.shape[0] == 2:  # [uncond, cond]
                out_val[1] += patch_delta
                out_val[0] -= patch_delta
            else:
                out_val += patch_delta

        hook_state["step_idx"] += 1
        return out_tensor

    return hook_fn


# =========================================================================== #
#                            Experiment Functions                            #
# =========================================================================== #


def run_standard_mode(
    source_prompt: str,
    control_prompt: str,
    config: ActivationPatchingConfig,
    pipe: HookedStableDiffusionPipeline,
    sae: TopKSAE,
    hook_name: str,
    height: int = 512,
    width: int = 512,
) -> None:
    """
    Run activation patching experiment with grid of k values.

    Args:
        source_prompt (str): Prompt containing features to transfer from.
        control_prompt (str): Control prompt to transfer features to.
        config (ActivationPatchingConfig): Configuration object.
        pipe (HookedStableDiffusionPipeline): The diffusion model pipeline.
        sae (TopKSAE): The sparse autoencoder.
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

    save_dir = (
        Path(config.output_dir)
        / block_name
        / Path(config.experiment_mode)
        / prompt_folder
        / ts_folder
        / f"{timestamp}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config:
    with open(save_dir / "config.json", "w") as f:
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
    source_img[0].save(save_dir / "source_image.png")

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
    control_img[0].save(save_dir / "control_image.png")

    # Store results for each k value
    results = {
        "source_img": source_img[0],
        "control_img": control_img[0],
        "k_results": {},
    }

    use_circuit_label = config.manual_source_indices is not None
    if use_circuit_label:
        runs = [
            {
                "k": "manual",
                "source_indices": config.manual_source_indices,
                "control_indices": config.manual_control_indices or [],
            }
        ]

    else:
        # Find features for the highest k value once
        max_k = max(config.k_values)
        logger.info(f"Finding top {max_k} features...")
        src_feats, ctrl_feats = find_top_k_features(
            pipe=pipe,
            sae=sae,
            source_prompt=source_prompt,
            control_prompt=control_prompt,
            k=max_k,
            config=config,
            hook_name=hook_name,
            height=height,
            width=width,
        )
        runs = [
            {
                "k": k,
                "source_indices": src_feats[:k],
                "control_indices": ctrl_feats[:k],
            }
            for k in sorted(config.k_values)
        ]

    # Run experiments for each k value by subsetting features
    for run in runs:
        k, src_indices, ctrl_indices = (
            run["k"],
            run["source_indices"],
            run["control_indices"],
        )
        logger.info(f"Running patch for k={k}...")
        add_vecs = extract_activations(
            pipe,
            sae,
            source_prompt,
            src_indices + ctrl_indices,
            config,
            hook_name,
            height=height,
            width=width,
        )
        sub_vecs = (
            extract_activations(
                pipe,
                sae,
                control_prompt,
                ctrl_indices,
                config,
                hook_name,
                height=height,
                width=width,
            )
            if config.patching_mode == "add_source_subtract_control"
            else None
        )

        # Run patched generation
        hook_fn = create_patching_hook(add_vecs, sae, config, sub_vecs)
        generator.manual_seed(config.seed)

        generator = torch.Generator(config.device).manual_seed(config.seed)
        patched_image = pipe.run_with_hooks(
            prompt=control_prompt,
            position_hook_dict={hook_name: hook_fn},
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            generator=generator,
            height=height,
            width=width,
        )[0]

        # Save individual result
        k_id = "circuit" if use_circuit_label else f"{k}_features"
        (save_dir / k_id).mkdir(exist_ok=True)
        patched_image.save(save_dir / k_id / "patched_image.png")
        results["k_results"][k] = patched_image
        with open(save_dir / k_id / "details.txt", "w") as f:
            f.write(
                f"Source: {source_prompt}\nControl: {control_prompt}\n"
                f"Source Features (+): {src_indices}\n"
            )
            if config.patching_mode == "add_source_subtract_control":
                f.write(f"Control Features (-): {ctrl_indices}\n")

    create_paper_figure(
        results,
        [r["k"] for r in runs],
        save_dir,
        config.figure_dpi,
        use_circuit_label,
    )
    logger.info(f"Standard mode results saved to: {save_dir}")


def run_bidirectional_mode(
    source_prompt: str,
    control_prompt: str,
    config: ActivationPatchingConfig,
    pipe: HookedStableDiffusionPipeline,
    sae: TopKSAE,
    hook_name: str,
    height: int = 512,
    width: int = 512,
) -> None:
    """
    Add most relevant features of prompt 1 and subtract most relevant features
    of prompt 2.

    Args:
        source_prompt (str): The prompt for the source image.
        control_prompt (str): The prompt for the control image.
        config (ActivationPatchingConfig): The configuration for activation
            patching.
        pipe (HookedStableDiffusionPipeline): The diffusion pipeline.
        sae (TopKSAE): The top-k spatial attention extractor.
        hook_name (str): The name of the hook to use.
        height (int, optional): The height of the generated images. Defaults
            to 512.
        width (int, optional): The width of the generated images. Defaults
            to 512.
    """
    set_all_seeds(config.seed)

    # Create directory
    block_name = hook_name.replace("unet.", "").replace(".", "_")
    words = (
        source_prompt.lower().replace(",", "").replace(".", "").split()[:10]
    )
    prompt_slug = "_".join(words)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_save_dir = (
        Path(config.output_dir)
        / block_name
        / Path(config.experiment_mode)
        / prompt_slug
        / timestamp
    )
    base_save_dir.mkdir(parents=True, exist_ok=True)

    with open(base_save_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=4)

    s_to_c_dir, c_to_s_dir = (
        base_save_dir / "source_to_control",
        base_save_dir / "control_to_source",
    )
    s_to_c_dir.mkdir(parents=True, exist_ok=True)
    c_to_s_dir.mkdir(parents=True, exist_ok=True)

    # Generate baseline images
    generator = torch.Generator(config.device).manual_seed(config.seed)
    source_img = pipe(
        prompt=source_prompt,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        generator=generator,
        height=height,
        width=width,
    ).images[0]

    generator = torch.Generator(config.device).manual_seed(config.seed)
    control_img = pipe(
        prompt=control_prompt,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        generator=generator,
        height=height,
        width=width,
    ).images[0]

    source_img.save(s_to_c_dir / "source_image.png")
    source_img.save(c_to_s_dir / "control_image.png")
    control_img.save(s_to_c_dir / "control_image.png")
    control_img.save(c_to_s_dir / "source_image.png")

    results_s_to_c = {
        "source_img": source_img,
        "control_img": control_img,
        "k_results": {},
    }
    results_c_to_s = {
        "source_img": control_img,
        "control_img": source_img,
        "k_results": {},
    }
    use_circuit_label = config.manual_source_indices is not None

    if use_circuit_label:
        runs = [
            {
                "k": "manual",
                "source_indices": config.manual_source_indices,
                "control_indices": config.manual_control_indices or [],
            }
        ]
    else:
        src_feats, ctrl_feats = find_top_k_features(
            pipe=pipe,
            sae=sae,
            source_prompt=source_prompt,
            control_prompt=control_prompt,
            k=max(config.k_values),
            config=config,
            hook_name=hook_name,
            height=height,
            width=width,
        )

        runs = [
            {
                "k": k,
                "source_indices": src_feats[:k],
                "control_indices": ctrl_feats[:k],
            }
            for k in sorted(config.k_values)
        ]

    for run in runs:
        k, src_indices, ctrl_indices = (
            run["k"],
            run["source_indices"],
            run["control_indices"],
        )

        logger.info(f"Running bidirectional patch for k={k}...")
        s_add_vecs = extract_activations(
            pipe,
            sae,
            source_prompt,
            src_indices,
            config,
            hook_name,
            height,
            width,
        )
        c_add_vecs = None
        if config.patching_mode == "add_source_subtract_control":
            c_add_vecs = extract_activations(
                pipe,
                sae,
                control_prompt,
                ctrl_indices,
                config,
                hook_name,
                height,
                width,
            )

        # Source -> Control
        hook_fn_s_to_c = create_patching_hook(
            s_add_vecs, sae, config, c_add_vecs
        )

        generator = torch.Generator(config.device).manual_seed(config.seed)
        patched_s_to_c = pipe.run_with_hooks(
            prompt=control_prompt,
            position_hook_dict={hook_name: hook_fn_s_to_c},
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            generator=generator,
            height=height,
            width=width,
        )[0]

        results_s_to_c["k_results"][k] = patched_s_to_c

        # Control -> Source
        hook_fn_c_to_s = create_patching_hook(
            c_add_vecs if c_add_vecs else s_add_vecs, sae, config, s_add_vecs
        )

        generator = torch.Generator(config.device).manual_seed(config.seed)
        patched_c_to_s = pipe.run_with_hooks(
            prompt=source_prompt,
            position_hook_dict={hook_name: hook_fn_c_to_s},
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            generator=generator,
            height=height,
            width=width,
        )[0]

        results_c_to_s["k_results"][k] = patched_c_to_s

        k_id = "circuit" if use_circuit_label else f"{k}_features"
        (s_to_c_dir / k_id).mkdir(exist_ok=True)
        (c_to_s_dir / k_id).mkdir(exist_ok=True)
        patched_s_to_c.save(s_to_c_dir / k_id / "patched_image.png")
        patched_c_to_s.save(c_to_s_dir / k_id / "patched_image.png")
        with open(s_to_c_dir / k_id / "details.txt", "w") as f:
            f.write(
                f"Source: {source_prompt}\nControl: {control_prompt}\n"
                f"Src Feats(+): {src_indices}\n"
                + (
                    f"Ctrl Feats(-): {ctrl_indices}\n"
                    if config.patching_mode == "add_source_subtract_control"
                    else ""
                )
            )
        with open(c_to_s_dir / k_id / "details.txt", "w") as f:
            f.write(
                f"Source: {control_prompt}\nControl: {source_prompt}\n"
                f"Src Feats(+): {ctrl_indices}\n"
                + (
                    f"Ctrl Feats(-): {src_indices}\n"
                    if config.patching_mode == "add_source_subtract_control"
                    else ""
                )
            )

    k_list = [r["k"] for r in runs]
    create_paper_figure(
        results_s_to_c,
        k_list,
        s_to_c_dir,
        config.figure_dpi,
        use_circuit_label,
    )
    create_paper_figure(
        results_c_to_s,
        k_list,
        c_to_s_dir,
        config.figure_dpi,
        use_circuit_label,
    )
    logger.info(f"Bidirectional results saved to: {base_save_dir}")


# =========================================================================== #
#                        Plotting Functions                                   #
# =========================================================================== #


def create_paper_figure(
    results: Dict,
    k_values: List[int],
    save_dir: Path,
    dpi: int = 300,
    use_circuit_label: bool = False,
) -> None:
    """
    Create paper-ready figure with control on left, source on right,
    and k variations in between (lowest k to highest k).

    Args:
        results (Dict[Any]): Dictionary containing images for each k value.
        k_values (List[int]): List of k values used.
        save_dir (Path): Directory to save the figure.
        dpi (int): DPI for the figure.
        use_circuit_label (bool): Whether to label the middle images as
            "Circuit" instead of "{k} Features".
    """
    # Sort k values to ensure proper ordering
    sorted_k = sorted(k_values)

    # Create figure with appropriate size
    n_images = len(sorted_k) + 2  # +2 for control and source
    fig, axes = plt.subplots(1, n_images, figsize=(3 * n_images, 3.5))

    axes[0].imshow(results["control_img"])
    axes[0].set_title("Destination", fontsize=28, fontweight="bold")
    axes[0].axis("off")
    for i, k in enumerate(sorted_k, 1):
        axes[i].imshow(results["k_results"][k])
        title = "Circuit" if use_circuit_label else f"{k} Features"
        axes[i].set_title(title, fontsize=28, fontweight="bold")
        axes[i].axis("off")
    axes[-1].imshow(results["source_img"])
    axes[-1].set_title("Source", fontsize=28, fontweight="bold")
    axes[-1].axis("off")

    # Adjust layout and save
    plt.tight_layout(pad=0.5)
    for ext in ["pdf", "png"]:
        plt.savefig(
            save_dir / f"paper_figure.{ext}",
            dpi=dpi,
            bbox_inches="tight",
            format=ext,
        )
    plt.close()
    logger.info(f"Figure saved to {save_dir / 'paper_figure.png'}")


if __name__ == "__main__":
    main()
