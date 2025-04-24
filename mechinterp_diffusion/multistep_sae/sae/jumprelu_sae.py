"""
This module implements the Jump ReLU activation function and its custom
backward pass using PyTorch's autograd functionality.

All functions defined in this script are based on the JAX implementation from:
    https://arxiv.org/pdf/2407.14435
with modifications in line with the Anthropic blogpost
    https://transformer-circuits.pub/2025/january-update/index.html
"""

from typing import Any, Optional, Tuple

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import torch
from torch import Tensor
from torch.autograd import Function

# =========================================================================== #
#                            Jump ReLU Class                                  #
# =========================================================================== #


# -----------------------------------------------------------------------------
# Rectangle helper function
# -----------------------------------------------------------------------------
def rectangle(x: Tensor) -> Tensor:
    """
    Elementwise rectangle function.
    """
    return ((x > 0.5) & (x < 0.5)).to(x.dtype)


# -----------------------------------------------------------------------------
# Step function with custom backward
# -----------------------------------------------------------------------------
class StepFunction(Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: Tensor,
        threshold: Tensor,
        bandwidth: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass for StepFunction

        Computes a step function where output is 1 when input exceeds threshold
        and 0 otherwise. The bandwidth parameter is stored to allow for sweeps
        similar to the ones in the original work by DeepMind.

            ctx (Any): Context object to store information for backward pass.
            x (Tensor): Input tensor.
            threshold (Tensor): Threshold values for activation.
            bandwidth (Optional[Tensor], optional): Controls the width of the
                approximation window in backward pass. Kernel bandwidth.
                Defaults to None.

            Tensor: Binary output tensor with same shape as input.
        """
        if bandwidth is None:
            # default bandwidth taken from the original work
            bandwidth = torch.tensor(0.001, dtype=x.dtype, device=x.device)
        # save inputs for backward pass
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        # compute the forward pass:
        return (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, Tensor, None]:
        """Defines the backward pass for StepFunction autograd function.

        Args:
            ctx (Any): Context object that stores information from the forward
                pass, including saved tensors and parameters.
            grad_output (Tensor): Gradient flowing back from subsequent layers.

        Returns:
            Tuple[Tensor, Tensor, None]: A tuple containing:
                - Gradient with respect to input x
                - Gradient with respect to threshold parameter
                - Gradient with respect to bandwidth
        """
        # retrieve saved inputs
        x, threshold = ctx.saved_tensors
        bw: Tensor = torch.clamp(
            ctx.bandwidth, min=1e-12
        )  # avoid division by 0

        # gradient w.r.t x
        grad_x: Tensor = torch.zeros_like(x)  # don't apply STE to x input
        # gradient w.r.t. threshold
        grad_threshold: Tensor = (
            -(1.0 / bw) * rectangle((x - threshold) / bw) * grad_output
        )

        # don't take gradient w.r.t. bandwidth
        grad_bandwidth: None = None
        return grad_x, grad_threshold, grad_bandwidth


# -----------------------------------------------------------------------------
# Step function wrapper
# -----------------------------------------------------------------------------
def step_fct(
    x: Tensor, threshold: Tensor, bandwidth: Optional[Tensor] = None
) -> Tensor:
    """Applies the StepFunction to the input tensor.

    Args:
        x (Tensor): Input tensor.
        threshold (Tensor): Threshold values for activation.
        bandwidth (Optional[Tensor], optional): Controls the width of the
            approximation window in backward pass. Kernel bandwidth.
            Defaults to None.

    Returns:
        Tensor: Output tensor after applying the step function.
    """
    if bandwidth is None:
        # default bandwidth taken from the original work
        bandwidth = torch.tensor(0.001, dtype=x.dtype, device=x.device)
    return StepFunction.apply(x, threshold, bandwidth)


# -----------------------------------------------------------------------------
# Jump ReLU custom backward
# -----------------------------------------------------------------------------
class JumpReLUFunction(Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: Tensor,
        threshold: Tensor,
        bandwidth: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass for JumpReLUFunction

        Computes a jump ReLU function where output is 1 when input exceeds
        threshold and 0 otherwise. The bandwidth parameter is stored to allow
        for sweeps similar to the ones in the original work by DeepMind.

            ctx (Any): Context object to store information for backward pass.
            x (Tensor): Input tensor.
            threshold (Tensor): Threshold values for activation.
            bandwidth (Optional[Tensor], optional): Controls the width of the
                approximation window in backward pass. Kernel bandwidth.
                Defaults to None.

            Tensor: Binary output tensor with same shape as input.
        """
        if bandwidth is None:
            # default bandwidth taken from the original work
            bandwidth = torch.tensor(0.001, dtype=x.dtype, device=x.device)

        # save inputs for backward pass
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

        return x * (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, Tensor, None]:
        """Defines the backward pass for JumpReLUFunction autograd function.

        Args:
            ctx (Any): Context object that stores information from the forward
                pass, including saved tensors and parameters.
            grad_output (Tensor): Gradient flowing back from subsequent layers.

        Returns:
            Tuple[Tensor, Tensor, None]: A tuple containing:
                - Gradient with respect to input x
                - Gradient with respect to threshold parameter
                - Gradient with respect to bandwidth
        """
        # retrieve saved inputs
        x, threshold = ctx.saved_tensors
        bw: Tensor = torch.clamp(
            ctx.bandwidth, min=1e-12
        )  # avoid division by 0

        # gradient w.r.t. x
        grad_x: Tensor = (x > threshold).to(x.dtype) * grad_output
        # gradient w.r.t. threshold
        grad_threshold: Tensor = (
            -(threshold / bw) * rectangle((x - threshold) / bw) * grad_output
        )
        # gradient w.r.t. bandwidth
        grad_bandwidth: None = None
        return grad_x, grad_threshold, grad_bandwidth


# -----------------------------------------------------------------------------
# Wrapper for JumpReLU
# -----------------------------------------------------------------------------
def jump_relu(
    x: Tensor, threshold: Tensor, bandwidth: Optional[Tensor] = None
) -> Tensor:
    """Applies the JumpReLUFunction elementwise to the input tensor.

    Args:
        x (Tensor): Input tensor.
        threshold (Tensor): Threshold values for activation.
        bandwidth (Optional[Tensor], optional): Controls the width of the
            approximation window in backward pass. Kernel bandwidth.
            Defaults to None.

    Returns:
        Tensor: Output tensor after applying the jump ReLU function.
    """
    if bandwidth is None:
        # default bandwidth taken from the original work
        bandwidth = torch.tensor(0.001, dtype=x.dtype, device=x.device)
    return JumpReLUFunction.apply(x, threshold, bandwidth)
