"""
Adapted from
https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/model.py
"""

# =========================================================================== #
#                           Packages and Presets                              #
# =========================================================================== #


from .base_sae import BaseSAE


# =========================================================================== #
#                       TopK Sparse Autoencoder Class                         #
# =========================================================================== #
class TopKSAE(BaseSAE):  # type: ignore
    """
    TopK Sparse Autoencoder class.
    """

    def __init__(
        self,
        cfg,
    ) -> None:
        super().__init__(cfg)
        self.architecture = cfg.architecture
        self.k = cfg.k
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        if self.architecture == "topk":
            # tied initiialization
            self.W_dec.weight.data = self.W_enc.weight.data.T.clone()
            # store in column major layout for kernel
            self.W_dec.weight.data = self.W_dec.weight.data.T.contiguous().T
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
