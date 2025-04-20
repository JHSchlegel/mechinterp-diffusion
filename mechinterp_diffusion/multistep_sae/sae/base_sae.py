"""
This module implements the Base Sparse Autoencoder class.


Inspired by concepts and implementations found in:
-SAEBench: https://github.com/adamkarvonen/SAEBench/blob/main/sae_bench/custom_saes/base_sae.py
-OpenAI SAE: https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/model.py
-SAeUron: https://github.com/cywinski/SAeUron/blob/main/SAE/sae.py
-SDXL-Unbox: https://github.com/surkovv/sdxl-unbox/blob/d5e383fea440aed59d533062f3d8f8435c9a3737/SAE/sae.py
"""

import json

# =========================================================================== #
#                           Packages and Presets                              #
# =========================================================================== #
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import torch
from safetensors.torch import load_model, save_model
from simple_parsing import Serializable
from torch import Tensor, nn

TORCH_STRING_DTYPE_MAP = {"float16": torch.float16, "float32": torch.float32}

logger = logging.getLogger(__name__)


# =========================================================================== #
#                            Base Sparse Autoencoder                          #
# =========================================================================== #
class BaseSAE(nn.Module, ABC):
    def __init__(
        self,
        cfg,
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.d_in = cfg.d_in
        self.d_sae = cfg.d_sae

        # ---------------------------------------------------------------------
        # Define SAE parameters
        # ---------------------------------------------------------------------
        # W_enc shape: [d_sae, d_in]
        self.W_enc: nn.Module = nn.Linear(cfg.d_in, cfg.d_sae, bias=False)
        # W_dec shape: [d_in, d_sae]
        self.W_dec: nn.Module = nn.Linear(cfg.d_sae, cfg.d_in, bias=False)

        self.pre_bias: nn.Parameter = nn.Parameter(torch.zeros(cfg.d_in))
        self.latent_bias: nn.Parameter = nn.Parameter(torch.zeros(cfg.d_sae))

        # ---------------------------------------------------------------------
        # Ensure consistent data types and devices
        # ---------------------------------------------------------------------
        self.dtype: torch.dtype = TORCH_STRING_DTYPE_MAP[cfg.dtype]
        self.device: torch.device = torch.device(cfg.device)
        self.to(dtype=self.dtype, device=self.device)

    def to(self, *args: Any, **kwargs: Any) -> "BaseSAE":
        """
        Move the model to the specified device and data type.

        Returns:
            BaseSAE: The model instance.
        """
        super().to(*args, **kwargs)
        if "device" in kwargs:
            device = kwargs["device"]
            self.device = (
                torch.device(device) if isinstance(device, str) else device
            )
        if "dtype" in kwargs:
            self.dtype = kwargs["dtype"]

        return self

    def preprocess_input(self, x: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        """Preprocess the input data before encoding.

        Args:
            x (Tensor): Input tensor of shape (batch_size, d_spatial, d_in).

        Returns:
            Tuple[Tensor, Dict[str, Tensor]]:
                - Tensor: Preprocessed input tensor.
                - Dict[str, Tensor]: Dictionary of standardization scale and
                    shift values
        """
        batch_size, d_spatial, d_in = x.shape

        assert (
            d_in == self.d_in
        ), f"Input tensor has {d_in} channels, but model expects {self.d_in}."

        x = x.reshape(batch_size * d_spatial, d_in)

        if self.cfg.standardize_input:
            mu = x.mean(dim=-1, keepdim=True)
            x_std = x.std(dim=-1, keepdim=True)
            x = (x - mu) / (x_std + 1e-5)
            logging.debug(
                "Preprocessing: Standardizing input to zero mean and unit var."
            )
            return x, {"mu": mu, "std": x_std}
        else:
            return x, {}

    def postprocess_output(
        self,
        x_reconstructed: Tensor,
        info: Dict[str, Any],
    ) -> Tensor:
        """Postprocess the output data after decoding.

        Args:
            x_reconstructed (Tensor): Reconstructed tensor of shape
                (batch_size * d_spatial, d_in).
            info (Dict[str, Any]): Dictionary of standardization scale and
                    shift values.

        Returns:
            Tensor: Postprocessed output tensor.
        """
        if self.cfg.standardize_input:
            # Undo standardization
            x_reconstructed = x_reconstructed * info["std"] + info["mu"]
            logging.debug(
                "Postprocessing: Undoing standardization of the output."
            )

        return x_reconstructed

    def save_to_disk(self, path_str: str) -> None:
        """Save the model and configuration to disk.

        Args:
            path_str (str): Path directory to save the model and config.
        """
        path = Path(path_str)
        path.mkdir(parents=True, exist_ok=True)

        weights_path = path / "sae_weights.safetensors"

        save_model(self, str(weights_path))

        with open(path / "config.json", "w") as f:
            json.dump(asdict(self.cfg), f, indent=4)

        logger.info(f"Model saved to {weights_path}")
        logger.info(f"Config saved to {path / 'config.json'}")

    @classmethod
    def load_from_disk(
        cls: Type["BaseSAE"],
        path_str: str,
        config_class: Type[Serializable],
        device: str,
    ) -> "BaseSAE":
        """Load the model from disk.

        Args:
            cls (Type[&quot;BaseSAE&quot;]): Class type of the model.
            path_str (str): Path to the directory containing the model weights
            config_class (Type[Serializable]): Class type of the configuration
            device (str): Device to load the model on (e.g., "cpu", "cuda:0").

        Returns:
            BaseSAE: An instance of the model loaded from disk.
        """
        path = Path(path_str)
        config_path = path / "config.json"
        weights_path = path / "sae_weights.safetensors"

        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist.")
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file {config_path} does not exist."
            )
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights file {weights_path} does not exist."
            )

        with open(config_path, "r") as f:
            cfg_dict = json.load(f)
            cfg: Serializable = config_class.from_dict(cfg_dict)

        cfg.device = device
        sae_instance = cls(cfg)

        load_model(
            model=sae_instance,
            filename=str(weights_path),
            strict=True,
            device=device,
        )

        logger.info(f"Model loaded from {weights_path}")

        return sae_instance

    @torch.no_grad()
    def unit_norm_decoder_(self, eps: float = 1e-12) -> None:
        """
        Normalize the decoder weights to have unit norm.

        Args:
            eps (float, optional): Small constant to avoid division by zero.
                Defaults to 1e-12.
        """

        norm: torch.Tensor = (
            self.W_dec.weight.data.norm(dim=0, keepdim=True) + eps
        )
        # Normalize decoder columns (d_in, d_sae)
        self.W_dec.weight.data /= norm

        logging.debug("Decoder columns normalized to unit norm.")

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions_(self) -> None:
        """Removes gradient components parallel to decoder vectors in-place.

        Assumes decoder vectors are already unit-normed. This prevents the
        optimizer from changing the norm of the decoder vectors.
        """
        if (
            self.W_dec.weight is not None
            and self.W_dec.weight.grad is not None
        ):
            # W_dec shape: (d_in, d_sae)
            # grad shape g: (d_in, d_sae)
            # Project gradient g onto weight vectors (columns) w:
            # g' = g - (g * w) * w

            # Calculate the parallel component: g * w
            parallel_component = torch.einsum(
                "d_in d_sae, d_in d_sae-> d_sae",
                self.W_dec.weight.grad,
                self.W_dec.weight.data,
            )
            # Subtract the parallel component: g' = g - parallel_component * w
            self.W_dec.weight.grad.sub_(
                torch.einsum(
                    "d_sae, d_in d_sae-> d_in d_sae",
                    parallel_component,
                    self.W_dec.weight.data,
                )
            )

            logger.debug(
                "Gradient component parallel to decoder directions removed."
            )
        else:
            logger.debug(
                "Skipping gradient adjustment: weights or gradients not found."
            )

    @abstractmethod
    def _initialize_weights(self) -> None:
        """
        Initialize the weights of the sparse autoencoder.
        """
        raise NotImplementedError(
            "No method defined for weight initialization in baseclass."
        )

    @abstractmethod
    def encode(self, x: Tensor) -> Any:
        """Encode the input data into a latent representation.

        Args:
            x (Tensor): Input tensor of shape (batch_size, d_spatial, d_in).

        Returns:
            Any: The encoded latent representation, and any additional
                information required for loss calculation.
        """
        raise NotImplementedError("Encoding not implemented in baseclass.")

    @abstractmethod
    def decode(self, latents: Tensor) -> Any:
        """Decode the latent representation back to the original space.

        Args:
            latents (Tensor): Latent representation of shape
                (batch_size * d_spatial, d_sae)

        Returns:
            Any: The decoded output, and any additional information required
                for loss calculation.
        """
        raise NotImplementedError("Decoding not implemented in baseclass.")

    @abstractmethod
    def forward(self, x: Tensor) -> Any:
        """Full forward pass: encode -> decode

        Args:
            x (Tensor): Input tensor of shape (batch_size, d_spatial , d_in).

        Returns:
            Any: The output of the decoder, as well as any additional
                information required for loss calculation.
        """
        raise NotImplementedError("Forward pass not implemented in baseclass.")

    @abstractmethod
    def calculate_loss(
        self,
        forward_output: Dict[str, Tensor],
        original_input: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Calculates the total loss and individual loss components.

        Args:
            forward_output (Dict[str, Tensor]): The dictionary returned by
                forward() method.
            original_input (Tensor): The original input tensor x passed to
                forward() method.

        Returns:
            Tuple[Tensor, Dict[str, Tensor]]:
                - Tensor: The total calculated loss, ready for backward().
                - Dict[str, Tensor]: A dictionary of individual loss components
                  (e.g., {'l2_loss': ..., 'l1_loss': ..., 'total_loss': ...})
                  for logging purposes. The total loss should also be included.
        """
        raise NotImplementedError(
            "Loss calculation not implemented in baseclass."
        )
