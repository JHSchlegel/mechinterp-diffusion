"""
Configuration file for extracting latent representations from multi-step
diffusion models and training sparse autoencoders on these representations.
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #

import datetime
import os
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Union

from simple_parsing import Serializable

# =========================================================================== #
#                           SAE Configuration                                 #
# =========================================================================== #


# -----------------------------------------------------------------------------
# Abstract Base Class SAE
# -----------------------------------------------------------------------------
@dataclass
class BaseSAEConfig(Serializable):
    dtype: Literal["float32", "float16"] = "float32"
    """
    Data type to use for the model weights. One of ['float16', 'float32']
    """

    device: str = "cuda"
    """
    Device to use for the model weights.
    """

    d_in: int = 1_280
    """
    Number of input channels of the latent representations of the U-Net
    cross-attention blocks.
    """

    d_sae: int = 5120
    """
    Number of columns of the decoder weight matrix i.e. the number of features.
    """

    standardize_input: bool = False
    """
    Whether to standardize input to zero mean and unit variance before
    encoding. If True, also undo standardization after decoding.
    """

    num_tokens_dead_threshold: int = 5_000_000
    """
    Number of tokens/samples without activation to consider a feature dead.
    """

    def __post_init__(self):
        if self.dtype not in ["float16", "float32"]:
            raise ValueError(
                f"Invalid dtype: {self.dtype}. Must be one of "
                f"['float16', 'float32']"
            )

        if not isinstance(self.device, str):
            raise ValueError(
                f"Invalid device: {self.device}. Must be of type str."
            )


# -----------------------------------------------------------------------------
# TopK
# -----------------------------------------------------------------------------
@dataclass
class TopKSAEConfig(BaseSAEConfig):
    k: int = 10
    """
    Number of TopK activations to keep during the forward pass.
    """

    k_aux: int = 256
    """
    How many topk dead features to use for auxiliary loss term.
    """

    auxk_loss_weight: float = 0.1  # 1 / 32
    """
    Weight for the auxiliary loss term in the TopK architecture.
    """

    l1_loss_weight: float = 0.0
    """
    Weight for the L1 loss term in the TopK architecture.
    """

    use_batch_topk: bool = False
    """
    Whether to use Batch-TopK SAE
    """

    standardize_input: bool = False
    """
    Whether to standardize input to zero mean and unit variance before
    encoding. If True, also undo standardization after decoding.
    """

    normalize_decoder: bool = True
    """
    Whether to normalize the decoder weights.
    """


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

    dataset_name: Literal["laion", "flickr30k"] = "laion"
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


# =========================================================================== #
#                        SAE Trainer Configuration                           #
# =========================================================================== #
@dataclass
class TrainerConfig(Serializable):
    """Configuration settings for the SAE Trainer."""

    # -------------------------------------------------------------------------
    # Dataloading settings
    # -------------------------------------------------------------------------
    dataset_path: str = (
        "../../../activations/stable-diffusion-2-1/flickr30k/train/subset_size-40000/25-inference-steps/every-1-steps/unet.down_blocks.2.attentions.0"
    )
    """Path to the directory containing the activation dataset."""

    seed: int = 42
    """Random seed for reproducibility."""

    buffer_size: int = 50
    """
    Number of examples to buffer at once for dataloading.
    """

    # -------------------------------------------------------------------------
    # Training settings
    # -------------------------------------------------------------------------

    lr: float | None = None
    """
    Learning rate for the optimizer. If None, scaling laws are used based on
    d_sae.
    """

    effective_batch_size: int = 4096
    """Number of activation vectors per training batch."""

    num_tokens: int = int(2e8)
    """
    Number of tokens to process during training. This is the number of
    activation vectors to process, not the number of training steps. The number
    of training steps is approximately to num_tokens // effective_batch_size.
    """

    target_timesteps: Optional[List[int]] = None
    """
    Specific diffusion timesteps to train on. If None, trains on all available
    timesteps in the dataset.
    """

    lr_scheduler_type: str = "constant"
    """
    Type of learning rate scheduler to use. See Hugging Face documentation for
    more details:
    https://huggingface.co/docs/transformers/en/main_classes/optimizer_schedules

    Scheduler types:
    - “linear” = get_linear_schedule_with_warmup
    - “cosine” = get_cosine_schedule_with_warmup
    - “cosine_with_restarts”=get_cosine_with_hard_restarts_schedule_with_warmup
    - “polynomial” = get_polynomial_decay_schedule_with_warmup
    - “constant” = get_constant_schedule
    - “constant_with_warmup” = get_constant_schedule_with_warmup
    - “inverse_sqrt” = get_inverse_sqrt_schedule
    - “reduce_lr_on_plateau” = get_reduce_on_plateau_schedule
    - “cosine_with_min_lr” = get_cosine_with_min_lr_schedule_with_warmup
    - “warmup_stable_decay” = get_wsd_schedule
    """

    warmup_steps: int = 0
    """Number of learning rate warmup steps."""

    adam_beta1: float = 0.9
    """Adam optimizer beta1."""

    adam_beta2: float = 0.999
    """Adam optimizer beta2."""

    max_grad_norm: float = 1.0
    """Maximum norm for gradient clipping."""

    # -------------------------------------------------------------------------
    # Wandb and logging
    # -------------------------------------------------------------------------
    wandb_project: Optional[str] = "sae_training"
    """Weights & Biases project name."""

    wandb_dir: Optional[str] = "../../../wandb"

    wandb_entity: Optional[str] = None
    """Weights & Biases entity name (optional)."""

    wandb_run_name: Optional[str] = None
    """Weights & Biases run name (optional, defaults to timestamp)."""

    wandb_log_code: bool = True
    """Whether to log the code to Weights & Biases."""

    log_frequency: int = 1
    """Log metrics to wandb every N steps."""

    plot_frequency: int = 5_000
    """Generate every N steps."""

    save_frequency: int = 5_000
    """Save model checkpoint every N steps."""

    checkpoint_path: str = "../../../checkpoints"
    """Directory to save model checkpoints."""


# =========================================================================== #
#                          SAE Training Configuration                         #
# =========================================================================== #
@dataclass
class TrainingConfig(Serializable):
    """Main configuration combining SAE and Trainer settings."""

    sae: Union[TopKSAEConfig] = field(default=None)
    """Specific SAE model configuration."""

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    """Trainer configuration settings."""

    def __post_init__(self) -> None:
        if self.trainer.wandb_run_name is None:
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            sae_type = type(self.sae).__name__.replace("Config", "")
            timesteps = (
                str(self.trainer.target_timesteps)
                if self.trainer.target_timesteps
                else "all"
            )
            self.trainer.wandb_run_name = f"{sae_type}_dsae-{self.sae.d_sae}_timesteps-{timesteps}_{now}"  # noqa: E501
        if self.trainer.lr is None:
            # scaling law for lr; use 2e-4 for d_sae = 2**14
            # from Figure 3 in https://arxiv.org/pdf/2406.04093
            self.trainer.lr = 2e-4 / (self.sae.d_sae / (2 << 13)) ** 0.5


# =========================================================================== #
#                       SAE Intervention Configuration                        #
# =========================================================================== #
@dataclass
class SAEInterventionConfig(Serializable):
    """
    Configuration for SAE-based feature interventions in diffusion models.
    """

    output_dir: str = "../../../intervention_outputs"
    """Directory to save outputs"""

    height: int = 512
    """Height of the generated images"""

    width: int = 512
    """Width of the generated images"""

    seed: int = 42
    """Random seed for reproducibility"""

    model_id: str = "stabilityai/stable-diffusion-2-1"
    """Hugging Face model ID"""

    sae_path: str = (
        "../../../checkpoints/TopKSAE_dsae-5120_20250429_192504/step_48829"
    )
    """Path to SAE model"""

    target_module: str = "unet.down_blocks.2.attentions.0"
    """Module to apply intervention to"""

    hook_type: Literal["add", "scale", "reconstruct"] = "add"
    """Type of hook to apply"""

    timesteps: List[int] = field(default_factory=lambda: [])
    """Space-separated list of timesteps to intervene at"""

    timestep_values: Dict[int, float] = field(default_factory=dict)
    """
    Space separated list of 'timestep=value' pairs for varying intervention
    strength
    """

    intervention_mode: Literal["grid", "trajectory"] = "grid"
    """
    Whether to generate grid of interventions with different intervention
    strengths or analyzing intervention over time
    """

    intervention_values: List[float] = field(
        default_factory=lambda: [-20.0, -10.0, 10.0, 20.0]
    )
    """List of values for intervention strength"""

    features: List[int] = field(default_factory=lambda: [0])
    """List of feature indices to intervene on"""

    dataset_path: str = "../../../laion-coco_captions"
    """Path to HuggingFace prompt dataset"""

    dataset_split: str = "test"
    """Dataset split to use"""

    num_prompts: int = 5
    """Number of prompts to process"""

    prompt_column: str = "caption"
    """Column name containing prompts"""

    capture_interval: int = 5
    """Interval for capturing intermediate diffusion steps"""

    save_activation_heatmap: bool = False
    """
    Whether to save heatmap of activations over time in
    trajectory/ reconstruction mode
    """

    # -------------------------------------------------------------------------
    # Diffusion model configuration
    # -------------------------------------------------------------------------
    num_inference_steps: int = 25
    """Number of inference steps for the diffusion process"""

    guidance_scale: float = 9.0
    """Guidance scale for classifier-free guidance"""

    torch_dtype: Literal["float16", "float32"] = "float32"
    """Torch data type"""

    def __post_init__(self) -> None:
        if self.timestep_values and self.intervention_mode == "grid":
            raise NotImplementedError(
                "Different intervention strenghts across timesteps are not "
                "supported for grid intervention mode."
            )

        if (self.intervention_mode == "trajectory") and len(
            self.intervention_values
        ) > 1:
            raise NotImplementedError(
                "Multiple intervention values are not supported for "
                "trajectory intervention and reconstruction mode."
            )

        if self.intervention_mode == "reconstruct":
            self.intervention_values = [0.0]

        if self.timestep_values and self.timesteps:
            raise ValueError(
                "Specify either 'timesteps' or 'timestep_values', not both."
            )
