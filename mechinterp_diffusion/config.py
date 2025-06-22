"""
Configuration file for extracting latent representations from multi-step
diffusion models and training sparse autoencoders on these representations.
"""

import datetime
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import torch
from hydra.core.config_store import ConfigStore
from simple_parsing import Serializable

# =========================================================================== #
#                           SAE Configuration                                 #
# =========================================================================== #


# -----------------------------------------------------------------------------
# Abstract Base Class SAE
# -----------------------------------------------------------------------------
@dataclass
class BaseSAEConfig(Serializable):
    dtype: str = "float32"
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

    num_tokens_dead_threshold: int = 10_000_000
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

    auxk_loss_weight: float = 1 / 32  #  0.1
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

    dataset_name: str = "laion"
    """
    Name of huggingface prompt dataset to use for extracting the latent
    activations. Must be one of ['laion', 'flickr30k', 'birds_vs_cats'].
    For 'laion', the guangyil/laion-coco-aestheticlaion-coco-aesthetic dataset
        is used.
    For 'flickr30k', the flickr30k dataset is used.
    For 'birds_vs_cats', a local dataset for comparing birds vs cats images
        is used.
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

    target_timesteps_idx: Optional[List[int]] = field(
        default=None,
    )
    """
    Target timesteps to cache. By default, all timesteps from 0 to
    num_inference_steps - 1 are cached. Note that indices are 0-based and index
    0 is the latest diffusion timestep i.e. t=1 resp. pure noise; and index
    num_inference_steps - 1 is the earliest diffusion timestep i.e. t=0
    """

    def __post_init__(self):
        if self.target_timesteps_idx is None:
            # Default to all timesteps
            self.target_timesteps_idx = list(range(self.num_inference_steps))
        # Sort to allow for easy extraction of
        self.target_timesteps_idx = sorted(self.target_timesteps_idx)
        if isinstance(self.hook_names, str):
            self.hook_names = [self.hook_names]

        if self.target_timesteps_idx == list(range(self.num_inference_steps)):
            step_name = "every-1-steps"
        else:
            step_name = (
                f"timesteps-{'_'.join(map(str, self.target_timesteps_idx))}"
            )

        if self.extracted_latents_path is None:
            self.extracted_latents_path = os.path.join(
                "../../data/activations",
                self.model_name.split("/")[-1],
                self.dataset_name.split("/")[-1],
                self.dataset_split,
                f"subset_size-{str(self.dataset_size)}",
                f"{self.num_inference_steps}-inference-steps",
                f"{step_name}",
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
        "../../data/activations/stable-diffusion-2-1/laion/train/subset_size-50000/25-inference-steps/every-1-steps/unet.down_blocks.2.attentions.0"  # noqa: E501
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

    num_tokens: int = int(2e9)
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

    wandb_dir: Optional[str] = "../../wandb"

    wandb_entity: Optional[str] = None
    """Weights & Biases entity name (optional)."""

    wandb_run_name: Optional[str] = None
    """Weights & Biases run name (optional, defaults to timestamp)."""

    wandb_log_code: bool = True
    """Whether to log the code to Weights & Biases."""

    log_frequency: int = 1
    """Log metrics to wandb every N steps."""

    plot_frequency: int = 50_000
    """Generate every N steps."""

    save_frequency: int = 50_000
    """Save model checkpoint every N steps."""

    checkpoint_path: str = "../../checkpoints/sae"
    """Directory to save model checkpoints."""


# =========================================================================== #
#                          SAE Training Configuration                         #
# =========================================================================== #
@dataclass
class TrainingConfig(Serializable):
    """Main configuration combining SAE and Trainer settings."""

    sae: Union[TopKSAEConfig] = field(default_factory=TopKSAEConfig)
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
            if self.sae.use_batch_topk:
                sae_name = "BatchTopKSAE"
            else:
                sae_name = sae_type
            self.trainer.wandb_run_name = f"{sae_name}_dsae-{self.sae.d_sae}_timesteps-{timesteps}_{now}"  # noqa: E501
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

    output_dir: str = "../../results/intervention_outputs"
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
        "../../checkpoints/sae/TopKSAE_dsae-5120_timesteps-all_20250523_212803/step_488282"
    )
    """Path to SAE model"""

    target_module: str = "unet.down_blocks.2.attentions.0"
    """Module to apply intervention to"""

    hook_type: str = "add"
    """Type of hook to apply"""

    timesteps: List[int] = field(default_factory=lambda: [])
    """Space-separated list of timesteps to intervene at"""

    timestep_values: Dict[int, float] = field(default_factory=dict)
    """
    Space separated list of 'timestep=value' pairs for varying intervention
    strength
    """

    intervention_mode: str = "grid"
    """
    Whether to generate grid of interventions with different intervention
    strengths, analyzing intervention over time or tracking highest
    activating features over time.
    """

    intervention_values: List[float] = field(
        default_factory=lambda: [-20.0, -10.0, 10.0, 20.0]
    )
    """List of values for intervention strength"""

    features: List[int] = field(default_factory=lambda: [0])
    """List of feature indices to intervene on"""

    dataset_path: str = "../../data/prompts/laion-coco_captions"
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

    prompts: Optional[List[str]] = None
    """Prompt to use for generation instead of dataset"""

    # -------------------------------------------------------------------------
    # Diffusion model configuration
    # -------------------------------------------------------------------------
    num_inference_steps: int = 25
    """Number of inference steps for the diffusion process"""

    guidance_scale: float = 9.0
    """Guidance scale for classifier-free guidance"""

    torch_dtype: str = "float32"
    """Torch data type"""

    topk_trace_k: int = 10
    """Number of highest activating features to track over time"""

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


# =========================================================================== #
#                        Ablation Study Configuration                         #
# =========================================================================== #
@dataclass
class AblationConfig:
    """Configuration for ablation studies with Hydra."""

    # SAE and trainer configs will be populated by Hydra
    sae: TopKSAEConfig = field(default_factory=TopKSAEConfig)
    """Sparse Autoencoder configuration for ablation studies."""

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    """Trainer configuration for ablation studies."""

    # Experiment settings
    sae_type: str = "topk"
    """
    Type of SAE to use for ablation studies. Currently only 'topk' is supported
    """

    seed: int = 42
    """Random seeds for reproducibility."""

    num_samples_per_timestep: int = 1  # 0_000
    """
    Number of test samples to use for each timestep during ablation
    """

    # Dataset paths (relative to ablation output directory)
    train_dataset: str = (
        "/media/Thesis/mechinterp-diffusion/data/activations/stable-diffusion-2-1/laion/train/subset_size-50000/25-inference-steps/every-1-steps/unet.down_blocks.2.attentions.0"  # noqa: E501
    )
    """Path to the training dataset containing activations """

    test_dataset: str = (
        "/media/Thesis/mechinterp-diffusion/data/activations/stable-diffusion-2-1/laion/test/subset_size-50000/25-inference-steps/every-1-steps/unet.down_blocks.2.attentions.0"  # noqa: E501
    )
    """Path to the test dataset containing activations """

    # Output directory
    output_dir: str = "../../results/ablation"
    """Directory to save ablation results and checkpoints."""

    def __post_init__(self) -> None:
        self.trainer.seed = self.seed


# =========================================================================== #
#                        Circuit Discovery Probe Configuration                #
# =========================================================================== #
@dataclass
class ProbeConfig(Serializable):
    """Configuration for probe training."""

    train_dataset_path: str = (
        "../../data/activations/stable-diffusion-2-1/birds_vs_cats/train/subset_size-10000/25-inference-steps/timesteps-4/unet.down_blocks.2.attentions.0"  # noqa: E501
    )
    """Path to the training dataset containing activations and labels."""

    test_dataset_path: str = (
        "../../data/activations/stable-diffusion-2-1/birds_vs_cats/test/subset_size-2000/25-inference-steps/timesteps-4/unet.down_blocks.2.attentions.0"  # noqa: E501
    )
    """Path to the test dataset. If None, derives from train_dataset_path."""

    num_epochs: int = 20
    """Number of epochs to train the probe model."""

    batch_size: int = 128
    """Batch size for training the probe model."""

    learning_rate: float = 1e-3
    """Learning rate for the AdamW optimizer."""

    weight_decay: float = 1e-5
    """AdamW optimizer weight decay."""

    n_hidden_channels: int = 64
    """Number of hidden channels in the probe network."""

    probe_type: Literal["cnn", "linear"] = "cnn"
    """Type of probe to use ('cnn' or 'linear')."""

    spatial_resolution: Tuple[int, int] = (16, 16)
    """Spatial resolution of the latent activations (height, width)."""

    seed: int = 42
    """Random seed for reproducibility."""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """Device to use for training, either 'cuda' or 'cpu'."""

    save_path: Optional[str] = None
    """Path to save the trained probe model. If None, auto-generates path."""

    def __post_init__(self):
        if self.save_path is None:
            dataset_parts = Path(self.train_dataset_path).parts
            if "birds_vs_cats" in dataset_parts:
                dataset_name = "birds_vs_cats"
            else:
                raise ValueError(
                    "invalid dataset path, please provide a valid path to a ",
                    "probing dataset.",
                )
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = Path("../../checkpoints") / "probe"
            save_dir.mkdir(parents=True, exist_ok=True)

            self.save_path = str(
                save_dir
                / f"{self.probe_type}_{dataset_name}_{timestamp}.safetensors"
            )


# =========================================================================== #
#                    Hydra Config Store Registration                          #
# =========================================================================== #

# Create a config store instance
cs = ConfigStore.instance()

# Register SAE configs
cs.store(name="topk", node=TopKSAEConfig, group="sae")

# Register trainer config
cs.store(name="base_trainer", node=TrainerConfig, group="trainer")

# Register ablation config
cs.store(name="ablation", node=AblationConfig)
