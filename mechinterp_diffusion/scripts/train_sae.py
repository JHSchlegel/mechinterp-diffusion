"""
This script trains a Sparse Autoencoder (SAE) model on activation data,
loading configuration directly from config.py and allowing for config overrides
via command line arguments.

Usage examples:
    # TopK SAE
    python train_sae.py TopK --k 10 --lr 0.001
"""

# =========================================================================== #
#                           Packages and Presets                              #
# =========================================================================== #
import logging
import os
import sys
from typing import Tuple

import torch
from datasets import Dataset, load_from_disk
from simple_parsing import (
    ArgumentParser,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",  # noqa: E501
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    TopKSAEConfig,
    TrainerConfig,
    TrainingConfig,
)
from sae.base_sae import BaseSAE
from sae.topk_sae import TopKSAE
from sae.trainer import SAETrainer
from utils.reproducibility import set_all_seeds


# =========================================================================== #
#                             Main Training Function                          #
# =========================================================================== #
def main() -> None:
    """
    Main function for training a Sparse Autoencoder (SAE) model.
    """
    config, sae_type = parse_arguments()

    os.makedirs(config.trainer.checkpoint_path, exist_ok=True)
    log_filename = config.trainer.checkpoint_path + "/train_sae.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",  # noqa: E501
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger.info(f"SAE Type: {sae_type}")
    logger.info(f"SAE Config: {config.sae}")
    logger.info(f"Trainer Config: {config.trainer}")

    set_all_seeds(config.trainer.seed)

    if not torch.cuda.is_available() and config.sae.device == "cuda":
        logger.warning("CUDA not available, switching device to CPU.")
        config.sae.device = "cpu"
    device = torch.device(config.sae.device)

    # -------------------------------------------------------------------------
    # Load the dataset
    # -------------------------------------------------------------------------
    logger.info(f"Loading dataset from: {config.trainer.dataset_path}")
    if not os.path.isdir(config.trainer.dataset_path):
        raise FileNotFoundError(
            f"Dataset path not found or not a directory: "
            f"{config.trainer.dataset_path}"
        )
    dataset: Dataset = load_from_disk(config.trainer.dataset_path)
    dataset.set_format(type="torch", columns=["activations", "timestep"])
    logger.info(f"Dataset loaded successfully with {len(dataset)} examples.")

    # -------------------------------------------------------------------------
    # Instantiate model and trainer; start training
    # -------------------------------------------------------------------------
    if sae_type == "TopK":
        SAEModelClass = TopKSAE
    else:
        raise TypeError(f"Unsupported SAE type: {sae_type}")
    logger.info(f"Initializing SAE model ({SAEModelClass.__name__})...")
    sae_model: BaseSAE = SAEModelClass(config.sae).to(device)
    logger.info(
        f"Model initialized with "
        f"{sum(p.numel() for p in sae_model.parameters()):,} parameters."
    )

    logger.info("Initializing Trainer...")
    trainer = SAETrainer(
        config=config,
        sae_model=sae_model,
        dataset=dataset,
    )

    logger.info("Starting training run...")
    logger.info(f"Effective configuration:\n{config}")
    trainer.train()

    logger.info("Training run finished.")


# =========================================================================== #
#                             Helper Functions                                #
# =========================================================================== #
def parse_arguments() -> Tuple[TrainingConfig, str]:
    """
    Parse command-line arguments for SAE training.

    Returns:
        tuple: (config, sae_type) where config is a TrainingConfig object and
               sae_type is a string ('TopK' or ...)
    """
    # basic parser just to get the SAE type
    parser = ArgumentParser()
    parser.add_argument(
        "sae_type", choices=["TopK"], help="Type of SAE to train"
    )
    args, remaining_args = parser.parse_known_args()

    sae_type = args.sae_type

    # Reset sys.argv for simple_parsing
    old_argv = sys.argv
    sys.argv = [sys.argv[0]] + remaining_args

    parser = ArgumentParser()

    # Add the right config classes based on SAE type
    if sae_type == "TopK":
        parser.add_arguments(TopKSAEConfig, dest="sae")
    else:
        raise ValueError(f"Unknown SAE type: {sae_type}")

    parser.add_arguments(TrainerConfig, dest="trainer")

    parsed_args = parser.parse_args()

    config = TrainingConfig(sae=parsed_args.sae, trainer=parsed_args.trainer)

    # Restore original sys.argv
    sys.argv = old_argv

    return config, sae_type


if __name__ == "__main__":
    main()
