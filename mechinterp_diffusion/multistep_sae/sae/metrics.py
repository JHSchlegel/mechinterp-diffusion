"""
This module implements metric calculation functions for SAE training.

Detailed overview:
    - Purpose: Provide reusable functions to compute metrics like L0 norm,
               explained variance, and feature sparsity/density.
    - Author: Jan Schlegel
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import logging

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# =========================================================================== #
#                                  Metrics                                    #
# =========================================================================== #


# -----------------------------------------------------------------------------
# L0 Norm
# -----------------------------------------------------------------------------
@torch.no_grad()
def compute_l0_norm(x: Tensor, threshold: float = 0.0) -> Tensor:
    """
    Calculates the average L0 norm (number of non-zero elements) per sample.

    Args:
        x (Tensor): Input tensor.
        threshold (float): Threshold below which values are considered zero.

    Returns:
        Tensor: average L0 norm of the input tensor across the batch.
    """
    l0_norm_per_sample = torch.sum(x.abs() > threshold, dim=-1).float()
    return torch.mean(l0_norm_per_sample).item()
