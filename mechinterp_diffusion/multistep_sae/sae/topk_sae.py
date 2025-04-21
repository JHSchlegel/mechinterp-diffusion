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

from .base_sae import BaseSAE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from .config import TopKSAEConfig

logger = logging.getLogger(__name__)

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

        self.k = cfg.k
        self.l1_coeff = cfg.l1_coeff
        self.use_batch_topk = cfg.use_batch_topk

        super().__init__(cfg)

        if cfg.normalize_decoder:
            self.unit_norm_decoder_()

    def _initialize_weights(self) -> None:
        """Initializes decoder weights as the transpose of encoder weights."""
        # tied initiialization
        self.W_dec.weight.data = self.W_enc.weight.data.T.clone()
        # store in column major layout for kernel
        self.W_dec.weight.data = self.W_dec.weight.data.T.contiguous().T
        logger.debug(
            "Decoder weights initialized as transpose of encoder weights."
        )
