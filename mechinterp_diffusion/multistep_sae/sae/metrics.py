"""
This module implements various metrics for training and evaluating
Sparse Autoencoders (SAEs) in PyTorch.

Extracted from SAE implementations for modularity.
"""

# =========================================================================== #
#                             Packages and Presets                            #
# =========================================================================== #
import logging

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# =========================================================================== #
#                                 Metrics                                     #
# =========================================================================== #
def mse(pred: Tensor, target: Tensor) -> Tensor:
    """Calculates mean squared error between a prediction and target tensor
    over all elements.

    Args:
        pred (Tensor): Prediction tensor.
        target (Tensor): Target tensor.

    Returns:
        Tensor: Scalar tensor containing the MSE loss.
    """
    return (pred.float() - target.float()).pow(2).mean()


def normalized_mse(x_reconstructed: Tensor, x: Tensor) -> Tensor:
    """Calculates MSE normalized by variance of the input. Used e.g. for
    computing auxiliary loss for the TopK SAEs.

    Args:
        recon (Tensor): Reconstructed tensor
        x (Tensor): Input tensor

    Returns:
        Tensor: Scalar tensor containing the normalized MSE.
    """
    x_mu = x.mean(dim=0, keepdim=True)
    loss = mse(x_reconstructed, x) / (mse(x, x_mu) + 1e-12)
    return loss.nan_to_num(0.0)  # handle potential NaN values


def explained_variance(x_reconstructed: Tensor, x: Tensor) -> float:
    """Calculates the explained variance between the original input and
    reconstructed inputs.

    Args:
        x_reconstructed (Tensor): Reconstructed input tensor.
        x (Tensor): Input tensor.

    Returns:
        float: Explained variance score.
    """
    x_float = x.float()
    x_reconstructed_float = x_reconstructed.float()

    input_variance = torch.var(x_float, dim=0, unbiased=False).mean()
    # Calculate variance of the residual (difference)
    residual_variance = torch.var(
        x_float - x_reconstructed_float, dim=0, unbiased=False
    ).mean()

    # Calculate explained variance
    explained_variance = (
        1.0 - residual_variance / (input_variance + 1e-12)
    ).item()

    return explained_variance
