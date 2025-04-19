"""
Adapted from
https://github.com/adamkarvonen/SAEBench/blob/main/sae_bench/custom_saes/base_sae.py
"""

import json

# =========================================================================== #
#                           Packages and Presets                              #
# =========================================================================== #
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Any, Type

import torch
from safetensors.torch import load_model, save_model
from simple_parsing import Serializable
from torch import Tensor, nn

TORCH_STRING_DTYPE_MAP = {"float16": torch.float16, "float32": torch.float32}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
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
        self.W_enc: nn.Module = nn.Linear(cfg.d_in, cfg.d_sae, bias=False)
        self.W_dec: nn.Module = nn.Linear(cfg.d_sae, cfg.d_in, bias=False)

        self.pre_bias: nn.Parameter = nn.Parameter(torch.zeros(cfg.d_in))
        self.latent_bias: nn.Parameter = nn.Parameter(torch.zeros(cfg.d_sae))

        # ---------------------------------------------------------------------
        # Ensure consistent data types and devices
        # ---------------------------------------------------------------------
        self.dtype: torch.dtype = TORCH_STRING_DTYPE_MAP[cfg.dtype]
        self.device: torch.device = torch.device(cfg.device)
        self.to(dtype=self.dtype, device=self.device)

    @abstractmethod
    def _initialize_weights(self) -> None:
        """
        Initialize the weights of the sparse autoencoder.
        """
        raise NotImplementedError(
            "No method defined for weight initialization in baseclass."
        )

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

    def save_to_disk(self, path_str: str) -> None:
        path = Path(path_str)
        path.mkdir(parents=True, exist_ok=True)

        weights_path = path / "sae_weights.safetensors"

        save_model(self, str(weights_path))

        with open(path / "config.json", "w") as f:
            json.dump(asdict(self.cfg), f, indent=4)

    @classmethod
    def load_from_disk(
        cls: Type["BaseSAE"],
        path_str: str,
        config_class: Type[Serializable],
        device: str,
    ) -> "BaseSAE":
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

        return sae_instance

    @abstractmethod
    def encode(self, x: Tensor) -> Any:
        """
        Encode the input data into a latent representation.
        """
        raise NotImplementedError("Encoding not implemented in baseclass.")

    @abstractmethod
    def decode(self, latents: Tensor) -> Any:
        """
        Decode the latent representation back to the original space.
        """
        raise NotImplementedError("Decoding not implemented in baseclass.")

    @abstractmethod
    def forward(self, x: Tensor) -> Any:
        """
        Full forward pass: encode -> decode

        """
        raise NotImplementedError("Forward pass not implemented in baseclass.")

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
        self.W_dec.weight.data /= norm

    @torch.no_grad()
    def unit_norm_decoder_adjustment_(self) -> None:
        pass
