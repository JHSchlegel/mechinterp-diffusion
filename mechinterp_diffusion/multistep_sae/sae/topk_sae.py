"""
Adapted from
https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/model.py
"""

# =========================================================================== #
#                           Packages and Presets                              #
# =========================================================================== #
import logging
import os
import sys
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .base_sae import BaseSAE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import TopKSAEConfig

logger = logging.getLogger(__name__)

# TODO: check SAeUron auxiliary loss scale and auxk; also check rescaling with
# total variacne
# TODO: double check SAeUron batchtopk flattening
# =========================================================================== #
#                             TopK SAE Class                                  #
# =========================================================================== #


class TopKSAE(BaseSAE):
    def __init__(
        self,
        cfg: TopKSAEConfig,
    ) -> None:
        """Initializes the TopKSAE.

        Args:
            cfg (TopKSAEConfig): Configuration object containing parameters
        """
        # Input Validation specific to TopK
        assert isinstance(
            cfg, TopKSAEConfig
        ), "cfg must be an instance of TopKSAEConfig"

        super().__init__(cfg)

        if cfg.normalize_decoder:
            self.unit_norm_decoder_()

    def _initialize_weights(self) -> None:
        """Initializes decoder weights as the transpose of encoder weights."""
        # Tied initiialization
        self.W_dec.weight.data = self.W_enc.weight.data.T.clone()
        # store in column major layout for kernel
        self.W_dec.weight.data = self.W_dec.weight.data.T.contiguous().T
        logger.debug(
            "Decoder weights initialized as transpose of encoder weights."
        )

    def forward(self, x: Tensor) -> Any:
        """Forward pass of the TopKSAE.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            Any: Output tensor after passing through the network
        """
        batch_size, d_spatial, d_in = x.shape
        x_preprocessed, preprocess_info = self.preprocess_input(x)

        # ---------------------------------------------------------------------
        # Encoding
        # ---------------------------------------------------------------------
        feature_acts_pre_relu = self.encode(x_preprocessed)
        feature_acts = F.relu(feature_acts_pre_relu)
        acts_topk, _ = self._get_topk(
            acts=feature_acts, k=self.cfg.k, batch_size=batch_size
        )

        # ---------------------------------------------------------------------
        # Decoding
        # ---------------------------------------------------------------------
        x_reconstructed = self.decode(acts_topk)
        self.update_inactive_features(acts_topk)

        # TODO: potentially add multi topk logic here

        output = self._get_loss_dict(
            x=x_preprocessed,
            x_reconstructed=x_reconstructed,
            acts_topk=acts_topk,
            info=preprocess_info,
        )
        # output["sae_out"] = output["sae_out"].reshape(x.shape)
        return output

    def encode(self, x: Tensor) -> Tensor:
        """Encodes the input tensor using the encoder part of the network.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Encoded tensor pre activation and pre topk
        """
        x_cent = x - self.b_dec
        # feature activations pre relu and pre topk:
        return self.W_enc(x_cent) + self.b_enc

    def decode(self, z: Tensor) -> Tensor:
        """Decodes the encoded tensor using the decoder part of the network.

        Args:
            z (torch.Tensor): Encoded tensor

        Returns:
            torch.Tensor: Decoded tensor
        """
        return self.W_dec(z) + self.b_dec

    def _get_topk(
        self, acts: Tensor, k: int = 32, batch_size: int = 4092
    ) -> Tuple[Tensor, Tensor]:
        """Applies the Top-K operation to the input tensor.
        Args:
            acts (Tensor): Input tensor
            k (int, optional): Number of top activations to keep. Defaults to
                32.
            batch_size (int, optional): Batch size. Defaults to 4092.
        Returns:
            Tuple[Tensor, Tensor]: Tuple containing:
                - Tensor: Sparse tensor with only top-k activations being
                    non-zero
                - Tensor: Indices of the top-k activations
        """
        # ---------------------------------------------------------------------
        # Top-K Selection
        # ---------------------------------------------------------------------
        if self.cfg.use_batch_topk:
            # -----------------------------------------------------------------
            # Batch Top-K: Select top k*batch_size across the entire batch
            # -----------------------------------------------------------------
            # see https://arxiv.org/pdf/2412.06410

            acts_flat = acts.flatten()

            # Find the top k * batch_size * d_spatial values and their indices
            topk_values, topk_indices = torch.topk(
                acts_flat, k * acts.shape[0], dim=-1, sorted=False
            )
            # Create a zero tensor and scatter the top-k values back
            feature_acts_sparse = torch.zeros_like(acts_flat)
            feature_acts_sparse.scatter_(-1, topk_indices, topk_values)
            # Reshape back to the original activation shape
            feature_acts_sparse = feature_acts_sparse.view_as(acts)
        else:
            # -----------------------------------------------------------------
            # Standard Top-K: Select top k for each sample independently
            # -----------------------------------------------------------------
            topk_values, topk_indices = torch.topk(
                acts, k, dim=-1, sorted=False
            )
            # Create a zero tensor and scatter the top-k values
            feature_acts_sparse = torch.zeros_like(acts)
            feature_acts_sparse.scatter_(-1, topk_indices, topk_values)

        return feature_acts_sparse, topk_indices

    def _get_loss_dict(
        self,
        x: Tensor,
        x_reconstructed: Tensor,
        acts_topk: Tensor,
        info: Dict[str, Tensor],
    ) -> Dict[str, Tensor | float]:
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
        l0_loss = (acts_topk > 0.0).float().sum(-1).mean()
        l1_loss = acts_topk.float().abs().sum(-1).mean()
        l2_loss = (x_reconstructed.float() - x.float()).pow(2).mean()
        # total_var = (x - x.mean(0)).float().pow(2).mean()
        auxk_loss = self._get_auxiliary_loss(x, x_reconstructed, acts_topk)

        loss = (
            l2_loss
            + auxk_loss * self.cfg.auxk_loss_weight
            + l1_loss * self.cfg.l1_loss_weight
        )

        num_dead_features = (
            self.num_batches_inactive >= self.cfg.num_batches_dead_threshold
        ).sum()

        sae_out = self.postprocess_output(x, info)
        return {
            "loss": loss,
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "auxk_loss": auxk_loss,
            "l0_loss": l0_loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "num_dead_features": num_dead_features,
        }

    def _get_auxiliary_loss(
        self, x: Tensor, x_reconstructed: Tensor, acts: Tensor
    ) -> Tensor:
        """Calculates the auxiliary loss for the TopKSAE.

        Based on the repo of the BatchTopK paper:
        https://github.com/bartbussmann/BatchTopK/blob/main/sae.py

        Args:
            x (Tensor): Original input tensor.
            x_reconstructed (Tensor): Reconstructed output tensor.
            acts (Tensor): Activation tensor before topk operation is applied.

        Returns:
            Tensor: The calculated auxiliary loss.
        """
        dead_features: Tensor = (
            self.num_batches_inactive >= self.cfg.num_batches_dead_threshold
        )

        if dead_features.sum() > 0:
            residual = x.float() - x_reconstructed.float()
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.cfg.k_aux, dead_features.sum()),
                dim=-1,
            )

            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            selected_weights = self.W_dec.weight[:, dead_features]
            x_reconstruct_aux = acts_aux @ selected_weights.T

            aux_loss = (
                (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )

            return aux_loss
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
