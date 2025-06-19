"""
Unit tests for the BaseSAE abstract class
"""

# =========================================================================== #
#                           Packages and Presets                              #
# =========================================================================== #

import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mechinterp_diffusion.multistep_sae.config import BaseSAEConfig
from mechinterp_diffusion.multistep_sae.sae.base_sae import BaseSAE

# =========================================================================== #
#                        Test Suite for BaseSAE Class                         #
# =========================================================================== #


# -----------------------------------------------------------------------------
# Concrete implementation of BaseSAE for testing
# -----------------------------------------------------------------------------
class SimpleSAE(BaseSAE):
    def _initialize_weights(self) -> None:
        """Simple weight initialization for testing"""
        torch.nn.init.normal_(self.W_enc.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.W_dec.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.b_enc)
        torch.nn.init.zeros_(self.b_dec)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Simple encoding step

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: encoded output
        """
        x = x - self.b_dec
        h = self.W_enc(x)
        return F.relu(h) + self.b_enc

    def decode(self, latents: Tensor) -> Tensor:
        """Simple decoding step

        Args:
            latents (Tensor): Latent tensor

        Returns:
            Tensor: decoded output
        """
        return self.W_dec(latents) + self.b_dec

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Full forward pass: encode -> decode"

        Args:
            x (Tensor): input tensor

        Returns:
            Tuple[Tensor, Tensor]: reconstructed input and latent
            representation
        """
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h


# -----------------------------------------------------------------------------
# Test class for BaseSAE
# -----------------------------------------------------------------------------
class TestBaseSAE(unittest.TestCase):
    def setUp(self):
        self.cfg = BaseSAEConfig(
            dtype="float32",
            device="cpu",
            d_in=128,
            d_sae=256,
        )

        self.sae = SimpleSAE(self.cfg)
        self.sae._initialize_weights()

    def test_initialization(self) -> None:
        """Test if the model initializes correctly"""
        self.assertEqual(self.sae.d_in, 128)
        self.assertEqual(self.sae.d_sae, 256)
        self.assertEqual(self.sae.dtype, torch.float32)
        self.assertEqual(self.sae.device.type, "cpu")

        # check shape of weights
        self.assertEqual(self.sae.W_enc.weight.shape, (256, 128))
        self.assertEqual(self.sae.W_dec.weight.shape, (128, 256))
        self.assertEqual(self.sae.b_enc.shape, (256,))
        self.assertEqual(self.sae.b_dec.shape, (128,))

    def test_unit_norm_decoder(self) -> None:
        """
        Test if unit_norm_decoder_ correctly normalizes the decoder weights
        """

        self.sae.unit_norm_decoder_()

        norms = self.sae.W_dec.weight.data.norm(dim=0, keepdim=False)
        for norm in norms:
            self.assertAlmostEqual(norm.item(), 1.0, places=6)

    def test_device_and_dtype(self) -> None:
        """Test if the model correctly handles device and dtype changes."""
        # skip test if no cuda
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping device test")

        # Change dtype, device and check whether update was successful
        self.sae.to(device="cuda", dtype=torch.float16)

        self.assertEqual(self.sae.device.type, "cuda")
        self.assertEqual(self.sae.dtype, torch.float16)

        # Check if all parameters have the correct device and dtype
        for param in self.sae.parameters():
            self.assertEqual(param.device.type, "cuda")
            self.assertEqual(param.dtype, torch.float16)

    def test_forward_pass(self) -> None:
        """Test the forward pass of the model."""
        # test inputs
        bs = 10
        x = torch.randn(bs, 128)

        # Check shapes fo forwa\rd
        x_hat, h = self.sae(x)
        self.assertEqual(x_hat.shape, (bs, self.sae.d_in))
        self.assertEqual(h.shape, (bs, self.sae.d_sae))

    def test_save_and_load(self) -> None:
        """Test saving and loading model weights."""
        # temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:

            # -----------------------------------------------------------------
            # Saving the model
            # -----------------------------------------------------------------
            save_path = os.path.join(temp_dir, "test_sae")
            self.sae.save_to_disk(save_path)

            # Files exist?
            self.assertTrue(Path(save_path).exists())
            self.assertTrue(
                Path(
                    os.path.join(save_path, "sae_weights.safetensors")
                ).exists()
            )
            self.assertTrue(
                Path(os.path.join(save_path, "config.json")).exists()
            )

            # -----------------------------------------------------------------
            # Loading the model
            # -----------------------------------------------------------------
            loaded_sae = SimpleSAE.load_from_disk(
                save_path, config_class=BaseSAEConfig, device="cpu"
            )

            # Check if loaded model has the same parameters
            for param_orig, param_loaded in zip(
                self.sae.parameters(), loaded_sae.parameters(), strict=False
            ):
                self.assertTrue(torch.allclose(param_orig, param_loaded))

            # Check if config values match
            self.assertEqual(loaded_sae.d_in, self.sae.d_in)
            self.assertEqual(loaded_sae.d_sae, self.sae.d_sae)
            self.assertEqual(loaded_sae.dtype, self.sae.dtype)

    def test_nonexistent_load_path(self) -> None:
        """
        Test that loading from a nonexistent path raises file not found error
        """
        with self.assertRaises(FileNotFoundError):
            SimpleSAE.load_from_disk(
                "nonexistent_path", config_class=BaseSAEConfig, device="cpu"
            )


if __name__ == "__main__":
    unittest.main()
