"""
This module implements a dataloader for loading activations of diffusion models
for training Sparse Autoencoders (SAEs).

Adapted from:
    - https://github.com/surkovv/sdxl-unbox/blob/d5e383fea440aed59d533062f3d8f8435c9a3737/SAE/dataset_iterator.py
"""

# =========================================================================== #
#                              Packages and Presets                           #
# =========================================================================== #

from typing import Any, Dict, Iterator

import torch
from torch.utils.data import Dataset


# =========================================================================== #
#                           Activation Iterator                               #
# =========================================================================== #
class CustomActivationsIterator:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        total_tokens: int = 1_000_000_000,
        buffer_size: int = 50,
    ) -> None:
        """
        Iterator for activation data that continues until a specified number of
        tokens is processed, with buffering for performance.

        Args:
            dataset (Dataset): A dataset object containing 'activations' and
                'timestep' keys. Activations should have shape
                [d_spatial, d_model].
            batch_size (int): Number of flattened activations per batch.
            total_tokens (int): Total number of tokens to see before ending
                iteration.
            buffer_size (int): Number of examples to buffer at once for
                efficiency.
        """

        self.dataset = dataset
        self.dataset_size = len(dataset)

        first_example = dataset[0]["activations"]
        if len(first_example.shape) != 2:
            raise ValueError(
                f"Expected activations with shape [d_spatial, d_model], got "
                f"{first_example.shape}"
            )

        self.seq_len = first_example.shape[0]  # Spatial dimension
        self.d_model = first_example.shape[1]  # Input dimension

        self.batch_size = batch_size
        self.total_tokens = int(total_tokens)
        self.num_in_buffer = buffer_size

        self.tokens_seen = 0
        self.buffer = None
        self.pointer = 0
        self.dataset_index = 0

        self.timestep_buffer = []

    def renew_buffer(
        self,
        to_retrieve: int,
    ) -> None:
        """
        Refreshes the buffer with new data from the dataset.Will cycle back to
        the beginning of the dataset if needed.

        Args:
            to_retrieve (int): Number of dataset examples to load.
        """

        to_merge = []
        timesteps_to_merge = []

        # Keep any unused data from the current buffer
        if self.buffer is not None and self.buffer.shape[0] > self.pointer:
            to_merge = [self.buffer[self.pointer :].clone()]
            timesteps_to_merge = self.timestep_buffer[self.pointer :]

        del self.buffer
        self.timestep_buffer = timesteps_to_merge

        # ---------------------------------------------------------------------
        # Load new examples into buffer and flatten them
        # ---------------------------------------------------------------------
        for _ in range(to_retrieve):
            batch_data = self.dataset[self.dataset_index]
            sample = batch_data["activations"]  # Shape: [seq_len, d_model]
            timestep = batch_data["timestep"]

            to_merge.append(sample)

            # Create repeated timestep values for each flattened token
            num_tokens = sample.shape[0]
            repeated_timesteps = [timestep] * num_tokens
            timesteps_to_merge.extend(repeated_timesteps)

            # Move to next example, cycling back if needed
            self.dataset_index += 1
            if self.dataset_index >= self.dataset_size:
                self.dataset_index = 0

        # Concatenate all examples into one buffer
        self.buffer = torch.cat(to_merge, dim=0)

        # Shuffle both buffer and timesteps together
        shuffled_indices = torch.randperm(self.buffer.shape[0])
        self.buffer = self.buffer[shuffled_indices]

        # Rearrange timesteps to match the shuffled buffer
        shuffled_timesteps = []
        for idx in shuffled_indices:
            shuffled_timesteps.append(timesteps_to_merge[idx])
        self.timestep_buffer = shuffled_timesteps

        self.pointer = 0

    def iterate(self) -> Iterator[Dict[str, Any]]:
        """
        Iterates through the dataset, yielding batches of flattened activations
        until the specified number of tokens has been seen.

        Yields:
            Dict[str, Any]: Dictionary containing:
                - 'activations': Tensor of shape [batch_size, d_model]
                - 'timestep': List of timestep values corresponding to each
                    activation
        """
        while self.tokens_seen < self.total_tokens:
            # Check if we need to replenish the buffer
            if (
                self.buffer is None
                or self.buffer.shape[0] - self.pointer < self.batch_size
            ):
                to_retrieve = (
                    self.num_in_buffer
                    if self.buffer is None
                    else self.num_in_buffer // 5
                )
                self.renew_buffer(to_retrieve)

            # Extract a batch from the buffer
            end_idx = min(self.pointer + self.batch_size, self.buffer.shape[0])
            batch = self.buffer[self.pointer : end_idx]
            batch_timesteps = self.timestep_buffer[self.pointer : end_idx]
            self.pointer = end_idx

            if (
                tokens_remaining := (self.total_tokens - self.tokens_seen)
            ) < batch.shape[0]:
                batch = batch[:tokens_remaining]
                batch_timesteps = batch_timesteps[:tokens_remaining]

            self.tokens_seen += batch.shape[0]

            # Skip partial batches for consistency
            if (
                batch.shape[0] < self.batch_size
                and tokens_remaining >= self.batch_size
            ):
                continue

            batch_dict = {"activations": batch, "timestep": batch_timesteps}

            yield batch_dict
