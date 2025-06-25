"""
This module provides utility functions to ensure reproducibility in PyTorch by
setting seed values for random number generators in Python, NumPy, and PyTorch.
"""

import logging
import random

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import numpy as np
import torch

logger = logging.getLogger(__name__)


# =========================================================================== #
#                            Random Seeding Utility                           #
# =========================================================================== #
def set_all_seeds(seed: int = 42) -> None:
    """
    Set seed for random number generators in Python, NumPy, PyTorch,
    and ensure deterministic behavior in CuDNN.

    Args:
        seed (int, optional): Seed value. Defaults to 42.
    """
    # random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (CPU and GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU

    # # Ensure deterministic behavior in PyTorch's CuDNN backend
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    logging.info(f"Random seed set to {seed}")
