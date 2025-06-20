"""
This module contains hooks for manipulating the output of diffusers diffusion
models.

Source:
https://github.com/surkovv/sdxl-unbox/blob/d5e383fea440aed59d533062f3d8f8435c9a3737/utils/hooks.py
Adapted for use with the my SAE classes.

License:
MIT License

Copyright (c) 2024 Viacheslav Surkov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import logging
import os
import sys
from typing import Tuple

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import torch
import torch.nn.functional as F
from torch import Tensor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sae.base_sae import BaseSAE

logger = logging.getLogger(__name__)


# =========================================================================== #
#                            Classes and Functions                            #
# =========================================================================== #
class TimedHook:
    def __init__(self, hook_fn, total_steps, apply_at_steps=None):
        self.hook_fn = hook_fn
        self.total_steps = total_steps
        self.apply_at_steps = apply_at_steps
        self.current_step = 0

    def identity(self, module, input, output):
        """
        Pass-through fuction that returns the unmodified output.
        """
        return output

    def __call__(self, module, input, output):
        """
        Apply the hook at specified timesteps, otherwise return unmodified
        output.
        """
        if self.apply_at_steps is not None:
            if self.current_step in self.apply_at_steps:
                self.__increment()
                return self.hook_fn(module, input, output)
            else:
                self.__increment()
                return self.identity(module, input, output)

        return self.identity(module, input, output)

    def __increment(self) -> None:
        """
        Increment the current step counter or reset to 0 if we reach
        total_steps.
        """
        if self.current_step < self.total_steps:
            self.current_step += 1
        else:
            self.current_step = 0


class EnhancedTimedHook(TimedHook):
    """
    Enhanced version of TimedHook that supports different values at different
        timesteps.

    Args:
        hook_fn (callable): Base hook function to apply
        total_steps (int): Total number of diffusion steps
        timestep_values (Dict[int, float]): Dictionary mapping timesteps to
            values
        feature_idx (int): Feature index to activate
        sae (nn.Module): Sparse autoencoder model
    """

    def __init__(
        self,
        hook_fn,
        total_steps,
        timestep_values,
        feature_idx,
        sae,
        hook_type: str,
    ):
        super().__init__(hook_fn, total_steps)
        self.timestep_values = timestep_values
        self.feature_idx = feature_idx
        self.sae = sae
        assert hook_type in ["add", "scale", "reconstruct"], (
            f"Invalid hook_type: {hook_type}. "
            "Choose from 'add', 'scale', or 'reconstruct'."
        )

        match hook_type:
            case "add":
                self.hook_fn = add_feature_hook
            case "scale":
                self.hook_fn = scale_feature_hook
            case "reconstruct":
                self.hook_fn = reconstruct_sae_hook

    def __call__(self, module, input, output):
        if self.current_step in self.timestep_values:
            value = self.timestep_values.get(self.current_step, 0.0)
            logger.info(
                f"Applying intervention at step {self.current_step} "
                f"with value {value}"
            )

            def tmp_hook_fn(m, i, o):
                return self.hook_fn(
                    self.sae,
                    self.feature_idx,
                    value,
                    m,
                    i,
                    o,
                )

            result = tmp_hook_fn(module, input, output)
            self.__increment()
            return result
        else:
            self.__increment()
            return self.identity(module, input, output)


@torch.no_grad()
def add_feature_hook(
    sae: BaseSAE,
    feature_idx: int,
    value: float,
    module,
    input: Tuple[Tensor],
    output: Tuple[Tensor],
) -> Tuple[Tensor]:
    """_summary_

    Args:
        sae (BaseSAE): The Sparse Autoencoder model
        feature_idx (int): Feature index to activate
        value (float): Activation strength
        module (_type_): Module being hooked
        input (Tuple[Tensor]): Input tensor tuple
        output (Tuple[Tensor]): Output tensor tuple

    Returns:
        Tuple[Tensor]: Modified output using SAE reconstruction
    """
    diff = (output[0] - input[0]).permute((0, 2, 3, 1)).to(sae.device)
    sae_input, info = sae.preprocess_input(diff)
    activations: Tensor = F.relu(sae.encode(sae_input))
    mask = torch.zeros_like(activations, device=diff.device)
    mask[..., feature_idx] = value
    to_add = sae.postprocess_output(mask @ sae.W_dec.weight.T, info)
    return (output[0] + to_add.permute(0, 3, 1, 2).to(output[0].device),)


@torch.no_grad()
def scale_feature_hook(
    sae: BaseSAE,
    feature_idx: int,
    beta: float,
    module,
    input: Tuple[Tensor],
    output: Tuple[Tensor],
) -> Tuple[Tensor]:
    """
    Modulate the activation of a specific feature in the SAE. Based on equation
    (9) from https://arxiv.org/pdf/2410.22366

    Args:
        sae (BaseSAE): The Sparse Autoencoder model
        feature_idx (int): Feature index to activate
        beta (float): Scaling factor for activation modulation
        module: Module being hooked
        input (Tuple[Tensor]): Input tensor tuple
        output (Tuple[Tensor]): Output tensor tuple

    Returns:
        Tuple[Tensor]: Modified output using SAE reconstruction
    """
    diff = (output[0] - input[0]).permute((0, 2, 3, 1)).to(sae.device)
    sae_input, info = sae.preprocess_input(diff)
    activations: Tensor = F.relu(sae.encode(sae_input))
    original_activations: Tensor = activations[..., feature_idx].clone()
    mask: Tensor = torch.zeros_like(activations, device=diff.device)
    mask[..., feature_idx] = beta * original_activations.squeeze(-1)
    to_add: Tensor = sae.postprocess_output(mask @ sae.W_dec.weight.T, info)
    return (output[0] + to_add.permute(0, 3, 1, 2).to(output[0].device),)


@torch.no_grad()
def reconstruct_sae_hook(
    sae: BaseSAE, module, input: Tuple[Tensor], output: Tuple[Tensor]
) -> Tuple[Tensor]:
    """
    Reconstruct the activation map using the SAE.

    Args:
        sae (BaseSAE): The Sparse Autoencoder model
        module: Module being hooked
        input (Tuple[Tensor]): Input tensor tuple
        output (Tuple[Tensor]): Output tensor tuple

    Returns:
        Tuple[Tensor]: Modified output using SAE reconstruction
    """
    diff: Tensor = (output[0] - input[0]).permute((0, 2, 3, 1)).to(sae.device)
    sae_input, info = sae.preprocess_input(diff)
    activations: Tensor = F.relu(sae.encode(sae_input))
    top_acts, _ = sae._get_topk(activations, k=sae.cfg.k)
    reconstructed: Tensor = sae.decode(top_acts)
    sae_output: Tensor = sae.postprocess_output(reconstructed, info)
    return (input[0] + sae_output.permute(0, 3, 1, 2).to(output[0].device),)


@torch.no_grad()
def get_activation_map(
    sae: BaseSAE,
    feature_idx: int,
    module,
    input: Tuple[Tensor],
    output: Tuple[Tensor],
) -> Tensor:
    """
    Get the activation map for a specific feature index.

    Args:
        sae (BaseSAE): The Sparse Autoencoder model
        feature_idx (int): Feature index to activate
        module: Module being hooked
        input (Tuple[Tensor]): Input tensor tuple
        output (Tuple[Tensor]): Output tensor tuple

    Returns:
        Tensor: Activation map for the specified feature index
    """
    diff = (output[0] - input[0]).permute((0, 2, 3, 1)).to(sae.device)
    sae_input, info = sae.preprocess_input(diff)
    activations: Tensor = F.relu(sae.encode(sae_input))
    return activations[..., feature_idx].squeeze(-1)
