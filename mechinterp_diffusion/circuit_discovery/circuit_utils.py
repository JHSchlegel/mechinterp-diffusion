"""
Utility functions needed for temporal attribution in circuit discovery
"""

# =========================================================================== #
#                             Packages and Presets                            #
# =========================================================================== #

import gc
import logging
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from activation_utils import SparseAct
from core.diffusion.hooked_sd_pipeline import HookedStableDiffusionPipeline
from core.sae.topk_sae import TopKSAE
from probe import LatentProbe

logger = logging.getLogger(__name__)


# =========================================================================== #
#                           Memory Management                                 #
# =========================================================================== #


def clear_memory_cache(device: str = "cuda") -> None:
    """
    Clear memory cache.

    Args:
        device (str): Device to clear cache on. Default is 'cuda'.
    """
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =========================================================================== #
#                        Activation and Gradient Extraction                  #
# =========================================================================== #


def compute_clean_pass_and_grads(
    model: HookedStableDiffusionPipeline,
    sae: TopKSAE,
    hook_name: str,
    timesteps_to_analyze: List[int],
    probe_timestep_idx: int,
    clean_prompt: str,
    guidance_scale: float,
    num_inference_steps: int,
    device: str,
    include_residuals: bool,
    metric_fn: Callable,
    height: int = 512,
    width: int = 512,
) -> Tuple[Dict[int, SparseAct], Dict[int, SparseAct], float]:
    """
    Performs a single forward and backward pass for the clean prompt to
    efficiently compute both the activations and their gradients.

    Args:
        model (HookedStableDiffusionPipeline): The diffusion model pipeline.
        sae (TopKSAE): The sparse autoencoder for activation compression.
        hook_name (str): The name of the module to hook into.
        timesteps_to_analyze (List[int]): List of timesteps to capture
            activations from
        probe_timestep_idx (int): The timestep index to compute the metric at.
        clean_prompt (str): The prompt to use for the clean pass.
        guidance_scale (float): The guidance scale for classifier-free
            guidance.
        num_inference_steps (int): Number of diffusion steps.
        device (str): The device to run the computations on.
        include_residuals (bool): Whether to include residuals in the analysis.
        metric_fn (callable): A function that takes in the probe input and
            returns a scalar
        height (int): Height of the generated images.
        width (int): Width of the generated images.

    Returns:
        clean_acts (Dict[int, SparseAct]): Captured activations at specified
            timesteps
        grads (Dict[int, SparseAct]): Gradients of the metric w.r.t. the
            activations
        metric_val (float): The value of the metric at the probe timestep
    """
    context = {
        "step_idx": 0,
        "probe_output": None,
        "captured_acts": {},
        # "preprocess_info": {},
    }

    def capture_and_reconstruct_hook(module, module_in, module_out):
        t_idx = context["step_idx"]

        if t_idx in timesteps_to_analyze:

            diff = (
                (module_out[0] - module_in[0])
                .permute(0, 2, 3, 1)
                .to(sae.device)
            )

            # sae_input, info = sae.preprocess_input(diff)
            # context["preprocess_info"][t_idx] = info

            b, h, w, c = diff.shape
            diff = diff.reshape(b * h * w, -1)

            sae_out_dict = sae(diff)

            feature_acts = sae_out_dict["feature_acts"].reshape(b, h, w, -1)
            x_reconstructed = sae_out_dict["sae_out"].reshape(b, h, w, -1)

            sae_input_3d = diff.reshape(b, h, w, c)
            residual = (
                sae_input_3d - x_reconstructed if include_residuals else None
            )

            feature_acts.requires_grad_(True)
            # Tell autograd to save the gradient for this specific non-leaf
            feature_acts.retain_grad()

            if residual is not None:
                residual.requires_grad_(True)
                # Also retain the gradient for the residual tensor
                residual.retain_grad()

            context["captured_acts"][t_idx] = SparseAct(
                act=feature_acts, res=residual
            )

            reconstructed_full = sae.decode(feature_acts)
            if include_residuals and residual is not None:
                # pass-through gradients:
                reconstructed_full += residual

            module_out = (
                module_in[0]
                + reconstructed_full.permute(0, 3, 1, 2).to(
                    module_out[0].device
                ),
            )

        if t_idx == probe_timestep_idx:
            probe_input = module_out[0].chunk(2, dim=0)[1]
            context["probe_output"] = metric_fn(probe_input)
        context["step_idx"] += 1
        return module_out

    model.unet.enable_gradient_checkpointing()
    # Run the single forward pass with hooks and gradient tracking
    model.run_with_hooks(
        prompt=clean_prompt,
        position_hook_dict={hook_name: capture_and_reconstruct_hook},
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        max_denoising_steps=max(max(timesteps_to_analyze), probe_timestep_idx)
        + 1,
        device=torch.device(device),
        output_type="latent",
        with_grad=True,
        height=height,
        width=width,
    )
    model.unet.disable_gradient_checkpointing()

    metric_val = context["probe_output"].item()
    context["probe_output"].sum().backward()

    clean_acts = {}
    grads = {}
    for t_idx, sparse_act in context["captured_acts"].items():
        grads[t_idx] = SparseAct(
            act=sparse_act.act.grad,
            res=(sparse_act.res.grad if sparse_act.res is not None else None),
        )

        clean_acts[t_idx] = SparseAct(
            act=sparse_act.act.detach(),
            res=(
                sparse_act.res.detach() if sparse_act.res is not None else None
            ),
        )

    return clean_acts, grads, metric_val


def extract_activations(
    model: HookedStableDiffusionPipeline,
    sae: TopKSAE,
    hook_name: str,
    timesteps_to_analyze: List[int],
    prompt: str,
    guidance_scale: float,
    num_inference_steps: int,
    device: str,
    use_cpu_offload: bool,
    include_residuals: bool,
    probe_timestep_idx: int,
    height: int,
    width: int,
    metric_fn: Callable[[torch.Tensor], float],
    **kwargs,
) -> Tuple[Dict[int, SparseAct], float]:
    """
    Extracts activations from a specific layer of the model at given timesteps.

    Args:
        model (HookedStableDiffusionPipeline): The diffusion model pipeline.
        sae (TopKSAE): The sparse autoencoder for activation extration.
        hook_name (str): The name of the module to hook into.
        timesteps_to_analyze (List[int]): List of timesteps to capture
            activations from
        prompt (str): The prompt to use for the forward pass.
        guidance_scale (float): The guidance scale for classifier-free
            guidance.
        num_inference_steps (int): Number of diffusion steps.
        device (str): The device to run the computations on.
        use_cpu_offload (bool): Whether to offload activations to CPU to save
            GPU memory.
        include_residuals (bool): Whether to include residuals in the analysis.
        probe_timestep_idx (int): The timestep index to compute the metric at.
        height (int): Height of the generated images.
        width (int): Width of the generated images.
        metric_fn (Callable[[torch.Tensor], float]): A function that takes in
            the probe input and returns a scalar metric value.
        **kwargs: Additional keyword arguments.

    Returns:
        acts (Dict[int, SparseAct]): Captured activations at specified
            timesteps
        metric_val (float): The value of the metric at the probe timestep

    """
    acts = {}
    # preprocess_info = {}
    metric_val = None

    max_timestep_needed = max(max(timesteps_to_analyze), probe_timestep_idx)

    _, cache = model.run_with_cache(
        prompt=prompt,
        positions_to_cache=[hook_name],
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        return_both_cond_uncond=True,
        save_input=True,
        save_output=True,
        with_grad=False,
        max_denoising_steps=max_timestep_needed + 1,
        device=torch.device(device),
        height=height,
        width=width,
    )

    for t_idx in timesteps_to_analyze:

        module_input = cache["input"][hook_name][:, t_idx].clone()
        module_output = cache["output"][hook_name][:, t_idx].clone()

        diff = (module_output - module_input).to(device)

        # Convert to 4D like in clean pass:
        # (batch, seq_len, channels) -> (batch, h, w, channels)
        b, seq_len, c = diff.shape
        h = w = int(seq_len**0.5)

        # Same processing as compute_clean_pass_and_grads
        with torch.no_grad():
            sae_out_dict = sae(diff)

        diff = diff.reshape(b, h, w, c)

        feature_acts = sae_out_dict["feature_acts"].reshape(b, h, w, -1)
        x_reconstructed = sae_out_dict["sae_out"].reshape(b, h, w, -1)

        sae_input = diff.reshape(b, h, w, c)
        residual = sae_input - x_reconstructed if include_residuals else None

        acts[t_idx] = SparseAct(
            act=(
                feature_acts.detach().cpu()
                if use_cpu_offload
                else feature_acts.detach()
            ),
            res=(
                residual.detach().cpu()
                if (residual is not None and use_cpu_offload)
                else (residual.detach() if residual is not None else None)
            ),
        )

    # Compute metric at probe timestep if metric_fn provided
    if metric_fn is not None:
        output = cache["output"][hook_name][:, probe_timestep_idx].clone()

        h = w = int(output.shape[1] ** 0.5)
        probe_input = output.chunk(2, dim=0)[1].reshape(1, -1, h, w)
        metric_val = metric_fn(probe_input.to(device))
        metric_val = metric_val.item()

    del cache
    clear_memory_cache(device)
    return acts, metric_val


# =========================================================================== #
#                            Sparse topk Utilities                            #
# =========================================================================== #


def get_topk_component_indices(sparse_act: SparseAct, k: int) -> List[int]:
    """ """
    # Concatenates [features; residuals]
    effects_tensor = sparse_act.to_tensor()
    # Average over all dimensions except the last (feature dimension)
    if effects_tensor.ndim > 1:
        dims_to_mean = tuple(range(effects_tensor.ndim - 1))
        # abs first to avoid cancellation
        feature_importance = effects_tensor.abs().mean(dim=dims_to_mean)
    else:
        feature_importance = effects_tensor.abs()
    # Find the indices of the top-k most important features
    _, topk_feature_indices = torch.topk(feature_importance, k)
    return topk_feature_indices.tolist()


# =========================================================================== #
#                                 Naive Testing                               #
# =========================================================================== #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Test extract_activations function
    logger.info("Testing extract_activations function...")

    from config import TopKSAEConfig

    # Default paths from run.sh
    sae_path = "../../checkpoints/sae/down_blocks.2.attentions.0/TopKSAE_dsae-5120_timesteps-all_20250816_083716/step_488282"  # noqa:E501
    probe_path = (
        "../../checkpoints/probe/cnn_birds_vs_cats_20250620_165530.safetensors"
    )

    # Model parameters
    model_id = "stabilityai/stable-diffusion-2-1"
    hook_name = "unet.down_blocks.2.attentions.0"
    device = "cpu"  # Force CPU for testing
    dtype = torch.float32

    print(f"Using device: {device}")
    print("Loading models...")

    # Load pipeline properly
    model = HookedStableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=dtype, use_safetensors=True
    ).to(device, dtype=dtype)

    sae = (
        TopKSAE.load_from_disk(
            sae_path, device=str(device), config_class=TopKSAEConfig
        )
        .to(dtype)
        .eval()
    )

    # Load probe
    probe = LatentProbe(
        n_latent_channels=sae.cfg.d_in,
        probe_type="cnn",
        spatial_resolution=(16, 16),
    )
    from safetensors.torch import load_file

    probe.load_state_dict(load_file(probe_path, device=str(device)))
    probe = probe.to(device).to(dtype).eval()

    # Test parameters - minimal for quick testing
    test_prompt = "A high-resolution image of a cat."
    timesteps_to_analyze = [0, 1]  # Just 2 timesteps for quick test
    probe_timestep_idx = 2
    guidance_scale = 7.5
    num_inference_steps = 3  # Very few steps for quick test
    height = 256  # Smaller for faster testing
    width = 256  # Smaller for faster testing
    use_cpu_offload = False
    include_residuals = True

    acts, _ = extract_activations(
        model=model,
        sae=sae,
        metric_fn=lambda x: -probe(x),
        hook_name=hook_name,
        timesteps_to_analyze=timesteps_to_analyze,
        prompt=test_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        device=device,
        use_cpu_offload=use_cpu_offload,
        include_residuals=include_residuals,
        probe_timestep_idx=probe_timestep_idx,
        height=height,
        width=width,
    )

    for t_idx, sparse_act in acts.items():
        logger.info(f"\nTimestep {t_idx}:")
        logger.info(f"\tFeature acts shape: {sparse_act.act.shape}")
        if sparse_act.res is not None:
            logger.info(f"\tResidual shape: {sparse_act.res.shape}")
        else:
            logger.info("\tResidual: None")

        # Show some statistics
        logger.info(
            f"\tFeature acts - min: {sparse_act.act.min():.4f}, "
            f"max: {sparse_act.act.max():.4f}, mean: "
            f"{sparse_act.act.mean():.4f}"
        )
        if sparse_act.res is not None:
            logger.info(
                f"\tResidual - min: {sparse_act.res.min():.4f}, max: "
                f"{sparse_act.res.max():.4f}, mean: "
                f"{sparse_act.res.mean():.4f}"
            )

    # Test compute_clean_pass_and_grads
    logger.info("\n\nTesting compute_clean_pass_and_grads function...")

    clean_acts, grads, probe_out_val = compute_clean_pass_and_grads(
        model=model,
        sae=sae,
        hook_name=hook_name,
        timesteps_to_analyze=timesteps_to_analyze,
        probe_timestep_idx=probe_timestep_idx,
        clean_prompt=test_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        device=device,
        include_residuals=include_residuals,
        metric_fn=lambda acts: -probe(acts),
        height=height,
        width=width,
    )

    logger.info(f"Probe output value: {probe_out_val:.6f}")

    for t_idx in timesteps_to_analyze:
        logger.info(f"\nTimestep {t_idx}:")
        logger.info(f"\tClean acts shape: {clean_acts[t_idx].act.shape}")
        if clean_acts[t_idx].res is not None:
            logger.info(f"\tResidual shape: {clean_acts[t_idx].res.shape}")
            logger.info(f"\tResidual grad shape: {grads[t_idx].res.shape}")

        # Show gradient stats
        logger.info(
            f"\tGradient stats - min: {grads[t_idx].act.min():.6f}, "
            f"\tmax: {grads[t_idx].act.max():.6f}, "
            f"\tmean: {grads[t_idx].act.mean():.6f}"
        )
