"""
Module for temporal circuit analysis in diffusion models.
Nodes are computed via gradient/IG, edges via JVP.

Based on SHIFT paper and attribution code:
- https://github.com/saprmarks/feature-circuits/blob/main/attribution.py
- https://arxiv.org/abs/2403.19647
"""

# =========================================================================== #
#                             Packages and Presets                            #
# =========================================================================== #


import logging
import sys
from collections import namedtuple
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional

import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from activation_utils import SparseAct
from circuit_utils import (
    clear_memory_cache,
    compute_clean_pass_and_grads,
    extract_activations,
)
from core.diffusion.hooked_sd_pipeline import HookedStableDiffusionPipeline
from core.sae.topk_sae import TopKSAE
from probe import LatentProbe

logger = logging.getLogger(__name__)

TemporalEffectOut = namedtuple(
    "TemporalEffectOut",
    ["nodes", "total_effect", "grads", "deltas"],
)

# NOTE: no smart caching yet for jvp s.t. computational time for t0->t1
# substantially lower than e.g. t3->t4; can be improved in future


# =========================================================================== #
#                           Nodes Computation                                 #
# =========================================================================== #


def compute_node_effects(
    model: HookedStableDiffusionPipeline,
    sae: TopKSAE,
    hook_name: str,
    timesteps_to_analyze: List[int],
    clean_prompt: str,
    dest_prompt: Optional[str] = None,
    method: Literal["grad", "ig", "exact"] = "grad",
    # feature_subset=None,
    height: int = 512,
    width: int = 512,
    probe_timestep: int = 10,
    guidance_scale: float = 9.0,
    num_inference_steps: int = 25,
    device: str = "cuda",
    include_residuals: bool = True,
    metric_fn: Optional[Callable[[torch.Tensor], float]] = None,
    **kwargs,
) -> TemporalEffectOut:
    match method:
        case "grad":
            temporal_effect = _compute_gradient_nodes(
                model=model,
                sae=sae,
                hook_name=hook_name,
                timesteps=timesteps_to_analyze,
                probe_ts=probe_timestep,
                clean_prompt=clean_prompt,
                dest_prompt=dest_prompt,
                metric_fn=metric_fn,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                device=device,
                include_residuals=include_residuals,
                height=height,
                width=width,
            )
            return temporal_effect
        case "ig":
            logger.info(
                "IG is slow as only iterative implemenation fits into memory"
            )
            return _compute_ig_nodes(
                model=model,
                sae=sae,
                hook_name=hook_name,
                timesteps=timesteps_to_analyze,
                probe_ts=probe_timestep,
                clean_prompt=clean_prompt,
                dest_prompt=dest_prompt,
                metric_fn=metric_fn,
                ig_steps=kwargs.get("ig_steps", 20),
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                device=device,
                include_residuals=include_residuals,
                height=height,
                width=width,
            )
        case "exact":
            # return _compute_exact_nodes(
            #     model=model,
            #     sae=sae,
            #     hook_name=hook_name,
            #     timesteps=timesteps_to_analyze,
            #     probe_ts=probe_timestep,
            #     clean_prompt=clean_prompt,
            #     dest_prompt=dest_prompt,
            #     metric_fn=metric_fn,
            #     feature_subset=kwargs.get("feature_subset", None),
            #     top_k_features=kwargs.get("top_k_features", 10),
            #     guidance_scale=guidance_scale,
            #     num_inference_steps=num_inference_steps,
            #     device=device,
            #     include_residuals=include_residuals,
            #     height=height,
            #     width=width,
            # )
            raise NotImplementedError("Exact method not implemented yet.")
        case _:
            raise ValueError(f"Unknown method {method}")


def _compute_gradient_nodes(
    model: HookedStableDiffusionPipeline,
    sae: TopKSAE,
    hook_name: str,
    timesteps: List[int],
    probe_ts: int,
    clean_prompt: str,
    dest_prompt: Optional[str] = None,
    # feature_subset: Optional[Dict[int, List[int]]] = None,
    metric_fn: Optional[Callable[[torch.Tensor], float]] = None,
    guidance_scale: float = 9.0,
    num_inference_steps: int = 25,
    device: str = "cuda",
    include_residuals: bool = True,
    height: int = 512,
    width: int = 512,
):
    """
    Computes node effects using attribution patching.

    Args:
        model (HookedStableDiffusionPipeline): The diffusion model with hooks.
        sae (TopKSAE): The sparse autoencoder for activations.
        hook_name (str): The name of the hook to attach to.
        timesteps (List[int]): List of timesteps to analyze.
        probe_ts (int): The timestep index where the probe is applied.
        clean_prompt (str): The prompt for the clean run.
        dest_prompt (Optional[str]): The prompt for the destination run.
        metric_fn (Optional[callable]): Function to compute the metric from
            model output.
        guidance_scale (float): The guidance scale for the diffusion model.
        num_inference_steps (int): Number of inference steps for the diffusion
            model.
        device (str): The device to run computations on.
        include_residuals (bool): Whether to include residuals in the analysis.
        height (int): Height of the generated images.
        width (int): Width of the generated images.

    Returns:
        TemporalEffectOut: Named tuple with nodes, total effect, grads,
            and deltas.
    """
    clean_acts, grads, metric_val_clean = compute_clean_pass_and_grads(
        model=model,
        sae=sae,
        hook_name=hook_name,
        timesteps_to_analyze=timesteps,
        probe_timestep_idx=probe_ts,
        clean_prompt=clean_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        device=device,
        include_residuals=include_residuals,
        metric_fn=metric_fn,
        height=height,
        width=width,
    )

    if dest_prompt is None:
        dest_acts = {}
        for t in timesteps:
            dest_acts[t] = SparseAct(
                act=torch.zeros_like(clean_acts[t].act),
                res=(
                    torch.zeros_like(clean_acts[t].res)
                    if clean_acts[t].res is not None
                    else None
                ),
            )
        total_effect = None
    else:
        dest_acts, metric_val_dest = extract_activations(
            model=model,
            sae=sae,
            hook_name=hook_name,
            timesteps_to_analyze=timesteps,
            prompt=dest_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            device=device,
            use_cpu_offload=False,
            include_residuals=include_residuals,
            probe_timestep_idx=probe_ts,
            height=height,
            width=width,
            metric_fn=metric_fn,
        )
        total_effect = metric_val_dest - metric_val_clean

    nodes = {}
    deltas = {}
    for t_idx in timesteps:
        dest_state, clean_state, grad = (
            dest_acts[t_idx],
            clean_acts[t_idx],
            grads[t_idx],
        )
        delta = (
            dest_state - clean_state.detach()
            if dest_state is not None
            else -clean_state.detach()
        )
        effect = delta @ grad
        deltas[t_idx] = delta
        nodes[t_idx] = effect

    total_effect = total_effect if total_effect is not None else None
    clear_memory_cache(device)
    return TemporalEffectOut(
        nodes=nodes,
        total_effect=total_effect,
        grads=grads,
        deltas=deltas,
    )


def _compute_ig_nodes(
    model: HookedStableDiffusionPipeline,
    sae: TopKSAE,
    hook_name: str,
    timesteps: List[int],
    probe_ts: int,
    clean_prompt: str,
    dest_prompt: Optional[str] = None,
    metric_fn: Optional[Callable] = None,
    ig_steps: int = 20,
    guidance_scale: float = 9.0,
    num_inference_steps: int = 25,
    device: str = "cuda",
    include_residuals: bool = True,
    height: int = 512,
    width: int = 512,
):
    """
    Computes node effects using Integrated Gradients. IMplemented using
    multiple separate backwards passes to save memory

    Args:
        model (HookedStableDiffusionPipeline): The diffusion model with hooks.
        sae (TopKSAE): The sparse autoencoder for activations.
        hook_name (str): The name of the hook to attach to.
        timesteps (List[int]): List of timesteps to analyze.
        probe_ts (int): The timestep index where the probe is applied.
        clean_prompt (str): The prompt for the clean run.
        dest_prompt (Optional[str]): The prompt for the destination run.
        metric_fn (Optional[callable]): Function to compute the metric from
            model output.
        ig_steps (int): Number of steps for the IG approximation.
        guidance_scale (float): The guidance scale for the diffusion model.
        num_inference_steps (int): Number of inference steps for the diffusion
            model.
        device (str): The device to run computations on.
        include_residuals (bool): Whether to include residuals in the analysis.
        height (int): Height of the generated images.
        width (int): Width of the generated images.

    Returns:
        TemporalEffectOut: Named tuple with nodes, total effect, grads,
            and deltas.
    """

    # Get the activations at the two endpoints of the integration path.
    # We use extract_activations for both to get detached tensors without
    # gradients.
    clean_acts, metric_val_clean = extract_activations(
        model=model,
        sae=sae,
        hook_name=hook_name,
        timesteps_to_analyze=timesteps,
        prompt=clean_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        device=device,
        use_cpu_offload=False,
        include_residuals=include_residuals,
        probe_timestep_idx=probe_ts,
        height=height,
        width=width,
        metric_fn=metric_fn,
    )

    if dest_prompt is None:
        # If no destination prompt, the baseline is a zero activation.
        dest_acts = {}
        for t in timesteps:
            dest_acts[t] = SparseAct(
                act=torch.zeros_like(clean_acts[t].act),
                res=(
                    torch.zeros_like(clean_acts[t].res)
                    if clean_acts[t].res is not None
                    else None
                ),
            )
        total_effect = None
    else:
        dest_acts, metric_val_dest = extract_activations(
            model=model,
            sae=sae,
            hook_name=hook_name,
            timesteps_to_analyze=timesteps,
            prompt=dest_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            device=device,
            use_cpu_offload=False,
            include_residuals=include_residuals,
            probe_timestep_idx=probe_ts,
            height=height,
            width=width,
            metric_fn=metric_fn,
        )
        total_effect = metric_val_dest - metric_val_clean

    # Initialize accumulators for the gradients.
    summed_grads = {
        t_idx: clean_acts[t_idx].zeros_like() for t_idx in timesteps
    }

    # NOTE: single backwards pass implementation would be too memory intensive
    # hence: iterative alternative that requires ig_steps backwards passses :(
    for step in tqdm(range(ig_steps), desc="IG Steps"):
        alpha = step / ig_steps

        interpolated_acts = {}

        # For each timestep, create the interpolated SparseAct.
        for t_idx in timesteps:
            # Linear interpolation: (1-alpha)*baseline + alpha*input
            interp_act = (1 - alpha) * dest_acts[t_idx] + alpha * clean_acts[
                t_idx
            ]

            # Compute gradients with respect to this interpolated
            # activation.
            interp_act.act.requires_grad_(True)
            interp_act.act.retain_grad()
            if interp_act.res is not None:
                interp_act.res.requires_grad_(True)
                interp_act.res.retain_grad()

            interpolated_acts[t_idx] = interp_act

        # Run a single forward/backward pass with the interpolated activations.
        context = {"step_idx": 0, "probe_output": None}

        def ig_hook(module, module_in, module_out):
            t_idx = context["step_idx"]  # noqa: B023

            if t_idx in timesteps:
                # Inject our interpolated acts.
                current_interp_act = interpolated_acts[t_idx]  # noqa: B023

                reconstructed_full = sae.decode(current_interp_act.act)
                if include_residuals and current_interp_act.res is not None:
                    reconstructed_full += current_interp_act.res

                # Intervene in the forward pass.
                module_out = (
                    module_in[0]
                    + reconstructed_full.permute(0, 3, 1, 2).to(
                        module_out[0].device
                    ),
                )

            if t_idx == probe_ts:
                probe_input = module_out[0].chunk(2, dim=0)[1]
                context["probe_output"] = metric_fn(probe_input)  # noqa: B023

            context["step_idx"] += 1  # noqa: B023

            return module_out

        model.unet.enable_gradient_checkpointing()
        # Prompt doesn't matter as much, as we overwrite activations
        model.run_with_hooks(
            prompt=clean_prompt,
            position_hook_dict={hook_name: ig_hook},
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_denoising_steps=max(max(timesteps), probe_ts) + 1,
            device=torch.device(device),
            output_type="latent",
            with_grad=True,
            height=height,
            width=width,
        )
        model.unet.disable_gradient_checkpointing()

        # Backpropagate to get the gradients at this interpolation step.
        context["probe_output"].sum().backward()

        # Accumulate the gradients.
        for t_idx in timesteps:
            summed_grads[t_idx] += interpolated_acts[t_idx].grad

        clear_memory_cache(device)

    nodes = {}
    deltas = {}
    avg_grads = {}
    for t_idx in timesteps:
        # Calculate the average gradient over all steps.
        avg_grads[t_idx] = summed_grads[t_idx] / ig_steps

        delta = clean_acts[t_idx] - dest_acts[t_idx]
        deltas[t_idx] = delta.detach()

        # The final IG attribution is delta * average_gradient.
        nodes[t_idx] = delta.detach() @ avg_grads[t_idx]

    return TemporalEffectOut(
        nodes=nodes,
        total_effect=total_effect,
        grads=avg_grads,
        deltas=deltas,
    )


# =========================================================================== #
#                           Edges Computation                                 #
# =========================================================================== #


def compute_edges_to_probe(
    grads: Dict[int, SparseAct],
    deltas: Dict[int, SparseAct],
    probe_timestep_idx: int,
) -> SparseAct:
    """
    Computes the edges from the features at the probe timestep to the final
    output. This is simply the gradient of the metric w.r.t these features,
    dotted with a delta of 1.

    Args:
        grads (Dict[int, SparseAct]): Dictionary of gradients at each timestep.
        deltas (Dict[int, SparseAct]): Dictionary of deltas at each timestep.
        probe_timestep_idx (int): The timestep index where the probe is
            applied.

    Returns:
        SparseAct: The edge effects from the probe timestep features to the
            output.
    """
    if probe_timestep_idx in grads and probe_timestep_idx in deltas:
        # The edge effect is the dot product of the change (delta) and the
        # sensitivity (grad); dot product to contract residual to single nr
        return deltas[probe_timestep_idx] @ grads[probe_timestep_idx]
    else:
        raise ValueError(
            f"Grads or Deltas for timestep {probe_timestep_idx} not found."
        )


def temporal_jvp_aggregated(
    model: HookedStableDiffusionPipeline,
    sae: TopKSAE,
    hook_name: str,
    t_upstream: int,
    t_downstream: int,
    downstream_feature_indices: List[int],
    left_vec: SparseAct,  # Gradient w.r.t. downstream activations
    right_vec: SparseAct,  # Delta of upstream activations
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    device: str,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Computes the Jacobian-Vector Product by aggregating spatially *before*
    backpropagation. This is vastly more efficient than the per-position method
    and pixels are less important than token positions in LLMs.

    Args:
        model (HookedStableDiffusionPipeline): The diffusion model with hooks.
        sae (TopKSAE): The sparse autoencoder for activations.
        hook_name (str): The name of the hook to attach to.
        t_upstream (int): The upstream timestep index.
        t_downstream (int): The downstream timestep index.
        downstream_feature_indices (List[int]): List of downstream feature
            indices
            to compute edges for.
        left_vec (SparseAct): The left vector (gradient) w.r.t. downstream
            activations.
        right_vec (SparseAct): The right vector (delta) of upstream
            activations.
        prompt (str): The prompt for the diffusion model.
        num_inference_steps (int): Number of inference steps for the diffusion
            model.
        guidance_scale (float): The guidance scale for the diffusion model.
        device (str): The device to run computations on.
        height (int): Height of the generated images.
        width (int): Width of the generated images.

    Returns:
        torch.Tensor: A sparse tensor representing the aggregated edge effects
            from upstream to downstream features.
    """
    # Define shapes for the final output matrix
    f_down = left_vec.act.shape[-1]
    f_up = right_vec.act.shape[-1]
    final_shape = (f_down + 1, f_up + 1)  # +1 for residuals

    # Single forward pass to get activations on the graph
    context = {"step_idx": 0, "upstream_act": None, "downstream_act": None}

    def jvp_hook(module, module_in, module_out):
        t_idx = context["step_idx"]
        if t_idx in [t_upstream, t_downstream]:
            diff = (
                (module_out[0] - module_in[0])
                .permute(0, 2, 3, 1)
                .to(sae.device)
            )

            # explicit reshaping to [bs*h*w, c] for clarity even though
            # technically not needed
            b, h, w, _ = diff.shape
            diff = diff.reshape(b * h * w, -1)
            sae_out = sae(diff)
            diff = diff.reshape(b, h, w, -1)

            sparse_act = SparseAct(
                act=sae_out["feature_acts"].reshape(b, h, w, -1),
                res=diff - sae_out["sae_out"].reshape(b, h, w, -1),
            )
            if t_idx == t_upstream:
                sparse_act.act.requires_grad_(True).retain_grad()
                if sparse_act.res is not None:
                    sparse_act.res.requires_grad_(True).retain_grad()
                context["upstream_act"] = sparse_act
                reconstructed = sae.decode(sparse_act.act)
                if sparse_act.res is not None:
                    reconstructed += sparse_act.res
                module_out = (
                    module_in[0] + reconstructed.permute(0, 3, 1, 2),
                )
            elif t_idx == t_downstream:
                context["downstream_act"] = sparse_act
        context["step_idx"] += 1
        return module_out

    model.run_with_hooks(
        prompt=prompt,
        position_hook_dict={hook_name: jvp_hook},
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        max_denoising_steps=t_downstream + 1,
        device=torch.device(device),
        output_type="latent",
        with_grad=True,
        height=height,
        width=width,
    )
    upstream_act = context["upstream_act"]
    downstream_act = context["downstream_act"]

    if upstream_act is None or downstream_act is None:
        logger.warning("JVP failed to capture activations.")
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), device=device),
            torch.zeros((0,), device=device),
            size=final_shape,
        )

    # Iteratively backpropagate for each downstream FEATURE (i.e. not once per
    # pixel position per feature)
    edge_matrix_rows = []

    # Combine feature activations and residual into single tensors
    downstream_act_tensor = downstream_act.to_tensor()
    left_vec_tensor = left_vec.to_tensor()

    # Iterate over the feature indices to compute edges for
    for f_down_idx in tqdm(
        downstream_feature_indices,
        desc=f"JVP t={t_upstream}->t={t_downstream}",
    ):
        # Zero out gradients from previous iteration
        if upstream_act.act.grad is not None:
            upstream_act.act.grad.zero_()
        if upstream_act.res is not None and upstream_act.res.grad is not None:
            upstream_act.res.grad.zero_()

        # Define the loss as the total effect on a single downstream feature
        # f_down_idx, SUMMED across all batch, height, and width dimensions.
        loss_for_one_feature = (
            left_vec_tensor[..., f_down_idx]
            * downstream_act_tensor[..., f_down_idx]
        ).sum()

        # Perform a SINGLE backward pass for this entire feature's aggregated
        # effect.
        loss_for_one_feature.backward(retain_graph=True)

        # Calculate the JVP result
        effect_tensor = (upstream_act.grad @ right_vec).to_tensor()

        # Aggregate the effects spatially to get a single value per upstream
        # feature.
        # This results in one row of our final (f_down x f_up) matrix.
        aggregated_row = effect_tensor.sum(
            dim=(0, 1, 2)
        )  # Sum over batch, h, w
        edge_matrix_rows.append(aggregated_row)

    if not edge_matrix_rows:
        return torch.zeros(final_shape, device=device)

    final_edge_matrix = torch.zeros(final_shape, device=device)
    computed_edges = torch.stack(edge_matrix_rows, dim=0)
    # Place them at the correct indices
    final_edge_matrix[downstream_feature_indices, :] = computed_edges

    return final_edge_matrix.to_sparse_coo()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import os

    from datasets import load_from_disk
    from diffusers import StableDiffusionPipeline
    from safetensors.torch import load_file

    sys.path.append(str(Path(__file__).resolve().parent.parent))

    from circuit_plotting import plot_causal_circuit
    from circuit_utils import get_topk_feature_indices
    from config import TopKSAEConfig
    from core.sae.topk_sae import TopKSAE

    device = "cpu"
    output_dir = "jvp_test_output"
    Path(output_dir).mkdir(exist_ok=True)
    cache_path = os.path.join(output_dir, "cached_circuit.pt")

    if os.path.exists(cache_path):
        # ---------------------------------------------------------------------
        # Load cached results
        # ---------------------------------------------------------------------
        logger.info(f"Loading cached circuit data from {cache_path}")
        cached_data = torch.load(cache_path, weights_only=False)
        final_nodes = cached_data["nodes"]
        final_edges = cached_data["edges"]
        num_inference_steps = cached_data["num_inference_steps"]
        timesteps_to_analyze = cached_data["timesteps_to_analyze"]
        probe_timestep_idx = cached_data["probe_timestep_idx"]

    else:
        # ---------------------------------------------------------------------
        # Load models and data, compute circuit from scratch
        # ---------------------------------------------------------------------
        logger.info("No cached circuit found. Computing from scratch...")
        # Load dataset
        dataset_path = "/media/Thesis/mechinterp-diffusion/data/prompts/circuit_discovery/birds_vs_cats/comparative_pairs/"  # noqa: E501
        dataset = load_from_disk(dataset_path)
        example = dataset["train"].shuffle().select(range(1))
        clean_prompt = example["clean_prompt"][0]
        patch_prompt = example["patch_prompt"][0]

        logger.info(f"Clean: {clean_prompt}")
        logger.info(f"Patch: {patch_prompt}")

        # Model paths
        sae_path = "../../checkpoints/sae/down_blocks.2.attentions.0/TopKSAE_dsae-5120_timesteps-all_20250816_083716/step_488282"  # noqa: E501
        probe_path = "../../checkpoints/probe/cnn_birds_vs_cats_20250620_165530.safetensors"  # noqa: E501
        hook_name = "unet.down_blocks.2.attentions.0"

        # Load models
        logger.info("Loading models...")
        base_pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float32,
            use_safetensors=True,
        )
        pipe = HookedStableDiffusionPipeline(base_pipe).to(device)
        pipe.set_progress_bar_config(disable=True)
        sae = TopKSAE.load_from_disk(
            sae_path, config_class=TopKSAEConfig, device=device
        )
        sae = sae.to(dtype=torch.float32).eval()
        probe = LatentProbe(
            n_latent_channels=sae.cfg.d_in,
            probe_type="cnn",
            spatial_resolution=(16, 16),
        )
        probe.load_state_dict(load_file(probe_path, device=device))
        probe = probe.to(device).to(dtype=torch.float32).eval()
        logger.info("Models loaded!")

        # Parameters for analysis
        timesteps_to_analyze = [0, 1, 2]
        probe_timestep_idx = 2
        num_inference_steps = 25
        guidance_scale = 9.0
        height, width = 512, 512
        top_k_jvp_nodes = 5

        # ---------------------------------------------------------------------
        # Nodes
        # ---------------------------------------------------------------------
        logger.info("Computing node effects...")
        node_results = compute_node_effects(
            model=pipe,
            sae=sae,
            hook_name=hook_name,
            timesteps_to_analyze=timesteps_to_analyze + [probe_timestep_idx],
            clean_prompt=clean_prompt,
            dest_prompt=patch_prompt,
            method="grad",
            probe_timestep=probe_timestep_idx,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            device=device,
            include_residuals=True,
            metric_fn=lambda x: -probe(x),
            height=height,
            width=width,
        )

        # ---------------------------------------------------------------------
        # Edges
        # ---------------------------------------------------------------------
        logger.info("\nComputing edges to probe...")
        edges_to_probe = compute_edges_to_probe(
            node_results.grads, node_results.deltas, probe_timestep_idx
        )

        # Step 3: Compute inter-timestep edges
        logger.info(
            "\nComputing inter-timestep edge effects "
            "(Markov assumption: t -> t+1)..."
        )
        edges = {}
        sorted_timesteps = sorted(timesteps_to_analyze) + [probe_timestep_idx]

        for i in range(len(sorted_timesteps) - 1):
            t_up = sorted_timesteps[i]
            t_down = sorted_timesteps[i + 1]

            logger.info(
                f"Identifying top {top_k_jvp_nodes} downstream nodes for edge "
                f"{t_up}->{t_down}..."
            )
            downstream_feature_indices = get_topk_feature_indices(
                node_results.nodes[t_down], k=top_k_jvp_nodes
            )

            if not downstream_feature_indices:
                logger.warning(
                    f"No important downstream nodes found for edge "
                    f"{t_up}->{t_down}. Skipping."
                )
                continue

            aggregated_edge_tensor = temporal_jvp_aggregated(
                model=pipe,
                sae=sae,
                hook_name=hook_name,
                t_upstream=t_up,
                t_downstream=t_down,
                downstream_feature_indices=downstream_feature_indices,
                left_vec=node_results.grads[t_down],
                right_vec=node_results.deltas[t_up],
                prompt=clean_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                device=device,
                height=height,
                width=width,
            )

            # Store the small, clean, aggregated tensor
            edges[(t_up, t_down)] = aggregated_edge_tensor

            # Memory cleanup
            del aggregated_edge_tensor
            clear_memory_cache(device)

        # ---------------------------------------------------------------------
        # Caching and Plotting
        # ---------------------------------------------------------------------

        # Prepare final data structures for plotting and caching
        final_nodes = node_results.nodes.copy()
        final_nodes["y"] = node_results.total_effect

        final_edges = edges.copy()
        final_edges[(probe_timestep_idx, "y")] = (
            edges_to_probe.to_tensor().mean(dim=(0, 1, 2))
        )

        # Save to cache
        logger.info(f"Saving computed circuit to {cache_path}")
        data_to_cache = {
            "nodes": final_nodes,
            "edges": final_edges,
            "num_inference_steps": num_inference_steps,
            "timesteps_to_analyze": timesteps_to_analyze,
            "probe_timestep_idx": probe_timestep_idx,
        }
        torch.save(data_to_cache, cache_path)

    plot_causal_circuit(
        nodes=final_nodes,
        edges=final_edges,
        timesteps=timesteps_to_analyze + [probe_timestep_idx],
        save_path="output_circuit/my_circuit.png",
        top_k_nodes_per_ts=5,
        top_k_edges=20,
        num_inference_steps=num_inference_steps,
    )
    logger.info("Done.")
