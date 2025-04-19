"""
Adapted from
https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/model.py
"""

# =========================================================================== #
#                           Packages and Presets                              #
# =========================================================================== #
from typing import Tuple

import torch
from torch import Tensor, nn

# =========================================================================== #
#                          Vanilla Sparse Autoencoder                         #
# =========================================================================== #


class VanillaSAE(nn.Module):
    def __init__(
        self,
        dim_in: int,
        cfg,
    ) -> None:
        self.cfg = cfg

        self.architecture: str = cfg.architecture
        self.num_features: int = dim_in * cfg.expansion_factor

        self.W_enc: nn.Module = nn.Linear(
            dim_in, self.num_features, bias=False
        )
        self.W_dec: nn.Module = nn.Linear(
            self.num_features, dim_in, bias=False
        )

        self.pre_bias = None

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        if self.architecture == "topk":
            self.unit_norm_decoder_()
        elif self.architecture == "jump_relu":
            raise NotImplementedError(
                "Jump ReLU architecture not implemented yet."
            )
        else:
            raise ValueError(
                f"Unknown architecture: {self.architecture}. "
                "Choose either 'topk' or 'jump_relu'."
            )

    @property
    def device(self) -> torch.device:
        return self.W_enc.device

    @property
    def dtype(self) -> torch.dtype:
        return self.W_enc.dtype

    def _encode_topk(self, x: Tensor) -> Tensor:
        pass

    def _encode_jump_relu(self, x: Tensor) -> Tensor:
        pass

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        pass

    @torch.no_grad()
    def unit_norm_decoder_(self) -> None:
        self.W_dec.data /= self.W_dec.data.norm(dim=0, keepdim=True)

    @torch.no_grad()
    def unit_norm_decoder_adjustment_(self) -> None:
        pass
