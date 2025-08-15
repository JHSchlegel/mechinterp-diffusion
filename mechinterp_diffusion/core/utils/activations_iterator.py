"""
This module implements a dataloader for loading activations of diffusion models
for training Sparse Autoencoders (SAEs).

Adapted from:
    - https://github.com/surkovv/sdxl-unbox/blob/d5e383fea440aed59d533062f3d8f8435c9a3737/SAE/dataset_iterator.py
"""

# =========================================================================== #
#                              Packages and Presets                           #
# =========================================================================== #

import logging
from typing import Any, Dict, Iterator

import torch
from datasets import Dataset
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


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
        assert (
            buffer_size >= 5
        ), "Buffer size should be at least 5 for ensuring diversity."

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
            # renew buffer if buffer not initialized or insufficient data
            # in buffer
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

                # Keep renewing until we have enough for a full batch
                while self.buffer.shape[0] - self.pointer < self.batch_size:
                    self.renew_buffer(self.num_in_buffer // 5)

                if self.buffer is None or self.buffer.shape[0] <= self.pointer:
                    logger.error(
                        "Buffer is empty or insufficient after renewal. "
                        "Exiting iteration."
                    )
                    break

            start_idx = self.pointer
            end_idx = start_idx + self.batch_size

            batch = self.buffer[start_idx:end_idx]
            batch_timesteps = self.timestep_buffer[start_idx:end_idx]

            tokens_remaining = self.total_tokens - self.tokens_seen

            # Only take the remaining tokens to ensure exactly total_tokens
            # are processed
            if tokens_remaining < batch.shape[0]:
                batch = batch[:tokens_remaining]
                batch_timesteps = batch_timesteps[:tokens_remaining]

            self.pointer += batch.shape[0]
            self.tokens_seen += batch.shape[0]

            batch_dict = {"activations": batch, "timestep": batch_timesteps}
            yield batch_dict


# =========================================================================== #
#                             Compatible Dataloader                           #
# =========================================================================== #
class FlatteningDataLoader:
    """
    DataLoader wrapper that outputs data in same format as
    CustomActivationsIterator
    """

    def __init__(self, dataset, batch_size, total_tokens, num_workers=4):
        first_example = dataset[0]["activations"]
        self.d_spatial = first_example.shape[0]
        self.d_in = first_example.shape[1]
        self.total_tokens = total_tokens
        self.tokens_yielded = 0

        # Calculate how many examples we need per batch
        examples_per_batch = max(1, batch_size // self.d_spatial)

        self.loader = DataLoader(
            dataset,
            batch_size=examples_per_batch,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def iterate(self):
        """Same method as CustomActivationsIterator.iterate()"""
        while self.tokens_yielded < self.total_tokens:
            for batch in self.loader:
                # Flatten activations
                acts = batch["activations"].view(-1, self.d_in)

                # Fast timestep expansion using torch
                timesteps = torch.repeat_interleave(
                    batch["timestep"], self.d_spatial
                )

                # Check if we'd exceed total_tokens
                tokens_in_batch = acts.shape[0]
                if self.tokens_yielded + tokens_in_batch > self.total_tokens:
                    # Truncate last batch
                    remaining = self.total_tokens - self.tokens_yielded
                    acts = acts[:remaining]
                    timesteps = timesteps[:remaining]

                self.tokens_yielded += acts.shape[0]

                yield {"activations": acts, "timestep": timesteps}

                if self.tokens_yielded >= self.total_tokens:
                    return
