"""
Unit tests for the CustomActivationsIterator class.
"""

import os

# =========================================================================== #
#                           Packages and Presets                              #
# =========================================================================== #
import sys
import unittest

import torch
from datasets import Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mechinterp_diffusion.multistep_sae.utils.activations_iterator import (
    CustomActivationsIterator,
)


# =========================================================================== #
#                           Test CustomActivationsIterator                    #
# =========================================================================== #
def create_simple_dataset(num_obs: int, seq_len: int, d_model: int) -> Dataset:
    dataset = Dataset.from_dict(
        {
            "activations": [
                torch.full((seq_len, d_model), float(i))
                for i in range(num_obs)
            ],
            "timestep": [i for i in range(num_obs)],
        }
    )

    dataset.set_format(type="torch", columns=["activations", "timestep"])

    return dataset


class TestSimpleDataloader(unittest.TestCase):

    def test_yields_correct_total_tokens(self) -> None:
        """Check if the total number of tokens yielded matches total_tokens."""
        seq_len, d_model = 10, 8
        dataset = create_simple_dataset(
            num_obs=200, seq_len=seq_len, d_model=d_model
        )

        batch_size = 32
        total_tokens_to_get = 155
        buffer_size = 50

        dataloader = CustomActivationsIterator(
            dataset=dataset,
            batch_size=batch_size,
            total_tokens=total_tokens_to_get,
            buffer_size=buffer_size,
        )

        yielded_token_count = 0
        for idx, batch in enumerate(dataloader.iterate()):
            if idx < total_tokens_to_get // batch_size:
                self.assertEqual(batch["activations"].shape[0], batch_size)
            yielded_token_count += batch["activations"].shape[0]

        self.assertEqual(yielded_token_count, total_tokens_to_get)

    def test_yields_correct_shapes(self) -> None:
        """
        Check if yielded batches have the correct shape
        (both activations and timestep).
        """
        seq_len, d_model = 10, 8
        dataset = create_simple_dataset(
            num_obs=10, seq_len=seq_len, d_model=d_model
        )
        batch_size = 32
        total_tokens_to_get = 155
        buffer_size = 50

        dataloader = CustomActivationsIterator(
            dataset=dataset,
            batch_size=batch_size,
            total_tokens=total_tokens_to_get,
            buffer_size=buffer_size,
        )

        batches = list(dataloader.iterate())
        self.assertTrue(len(batches) > 0)

        self.assertTrue(
            all(
                batch["activations"].shape[0] == batch_size
                for batch in batches[:-1]
            )
        )

        self.assertTrue(
            all(batch["activations"].shape[1] == d_model for batch in batches)
        )

        self.assertTrue(
            all(
                len(batch["timestep"]) == batch["activations"].shape[0]
                for batch in batches
            )
        )

        # Last batch:
        self.assertTrue(batches[-1]["activations"].shape[0] <= batch_size)

    def test_yields_correct_timesteps(self) -> None:
        """
        Test if the yielded timesteps are assigned correctly to the
        activations.
        """
        seq_len, d_model = 5, 4
        dataset = create_simple_dataset(
            num_obs=30, seq_len=seq_len, d_model=d_model
        )
        batch_size = 16
        total_tokens_to_get = 80
        buffer_size = 10

        dataloader = CustomActivationsIterator(
            dataset=dataset,
            batch_size=batch_size,
            total_tokens=total_tokens_to_get,
            buffer_size=buffer_size,
        )

        for batch in dataloader.iterate():
            activations = batch["activations"]
            timesteps = batch["timestep"]
            # Check each token in the batch
            for i in range(activations.shape[0]):
                token_data = activations[i]
                original_index = timesteps[i]
                # As simple dataset is filled with indices, check if
                # the token data matches the original index
                self.assertTrue(torch.all(token_data == float(original_index)))


if __name__ == "__main__":
    unittest.main()
