"""
Script to create comparative and single-object prompt datasets.

It generates three types of datasets from a single configuration:
1.  Comparative Pairs: For path patching experiments (e.g., bird vs. cat).
2.  Object A Only: For knockout/ablation experiments (e.g., only bird prompts).
3.  Object B Only: For knockout/ablation experiments (e.g., only cat prompts).
"""

# =========================================================================== #
#                         Packages and Presets                                #
# =========================================================================== #

import itertools
import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset, DatasetDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =========================================================================== #
#                            Config Definition                                #
# =========================================================================== #
@dataclass
class DatasetConfig:
    """A dataclass to hold the configuration for a comparative dataset."""

    name: str
    object_a: str
    object_b: str
    prompt_template: str = "A {style} of a {color} {object} {action}."
    styles: List[str] = field(
        default_factory=lambda: [
            "photorealistic image",
            "digital art image",
            "close up image",
            "portrait image",
            "highly detailed image",
            "cinematic image",
            "vibrant image",
            "artistic image",
            "realistic image",
        ]
    )
    colors: List[str] = field(
        default_factory=lambda: [
            "white",
            "black",
            "orange",
            "brown",
            "grey",
            "brown and white",
            "black and white",
        ]
    )
    actions: List[str] = field(
        default_factory=lambda: [
            "in profile",
            "facing the camera",
            "in side view",
            "at rest",
            "sitting still",
            "resting",
            "sleeping",
        ]
    )


# -----------------------------------------------------------------------------
# Specific Example Configs
# -----------------------------------------------------------------------------
BIRD_VS_CAT_CONFIG = DatasetConfig(
    name="birds_vs_cats",
    object_a="bird",
    object_b="cat",
)

DOG_VS_CAT_CONFIG = DatasetConfig(
    name="dogs_vs_cats",
    object_a="dog",
    object_b="cat",
)

DREADLOCKS_VS_PERSON_CONFIG = DatasetConfig(
    name="dreadlocks_vs_person",
    object_a="person with dreadlocks",
    object_b="person",
    prompt_template="A {style} of a {object} {action}.",
    colors=[],  # Color doesn't make as much sense here.
    actions=[
        "facing the camera",
        "in profile",
        "from the back",
        "looking away",
        "smiling",
        "with a neutral expression",
    ],
)

CONFIG_REGISTRY = {
    "bird_vs_cat": BIRD_VS_CAT_CONFIG,
    "dog_vs_cat": DOG_VS_CAT_CONFIG,
    "dreadlocks_vs_person": DREADLOCKS_VS_PERSON_CONFIG,
}


# =========================================================================== #
#                           Main Functionality                                #
# =========================================================================== #


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Create datasets for circuit discovery."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=CONFIG_REGISTRY.keys(),
        help="The dataset configuration to use.",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=200,
        help="Number of training examples.",
    )
    parser.add_argument(
        "--test_size", type=int, default=100, help="Number of test examples."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--base_save_path",
        type=str,
        default="../../data/prompts/circuit_discovery",
        help="Base directory to save all datasets.",
    )

    args = parser.parse_args()

    # Select and generate prompts for the chosen config
    config = CONFIG_REGISTRY[args.config]

    prompt_generator = PromptGenerator(config).generate()

    save_path_for_config = Path(args.base_save_path) / config.name
    builder = DatasetBuilder(
        base_save_path=save_path_for_config,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    # Cnstruct all dataset types
    builder.build_and_save(
        "comparative_pairs", prompt_generator.comparative_pairs
    )
    builder.build_and_save(
        f"{config.object_a.replace(' ', '_')}_only",
        prompt_generator.object_a_prompts,
    )
    builder.build_and_save(
        f"{config.object_b.replace(' ', '_')}_only",
        prompt_generator.object_b_prompts,
    )


# =========================================================================== #
#                            Prompt Generation                                #
# =========================================================================== #


class PromptGenerator:
    """Generates prompt lists based on a DatasetConfig."""

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self.comparative_pairs: List[Dict[str, Any]] = []
        self.object_a_prompts: List[Dict[str, Any]] = []
        self.object_b_prompts: List[Dict[str, Any]] = []

    def generate(self) -> "PromptGenerator":
        """Generates all prompt variations (i.e. comparative, object A only,
        objecst B only) for the given configuration.

        Returns:
            PromptGenerator: Self, with populated prompt lists.
        """
        logger.info(f"Generating prompts for config: '{self.config.name}'")

        # Handle cases with empty attribute lists by providing a placeholder
        attributes = [
            self.config.styles or [None],
            self.config.colors or [None],
            self.config.actions or [None],
        ]

        combinations = list(itertools.product(*attributes))
        logger.info(f"Created {len(combinations)} attribute combinations.")

        # Shuffle combinations to randomize direction assignment
        random.shuffle(combinations)

        # Determine the exact midpoint for a 50/50 split
        split_point = len(combinations) // 2

        for i, (style, color, action) in enumerate(combinations):
            prompt_a_text = self._format_prompt(
                self.config.object_a, style, color, action
            )
            prompt_b_text = self._format_prompt(
                self.config.object_b, style, color, action
            )

            shared_attrs = {"style": style, "color": color, "action": action}

            # -----------------------------------------------------------------
            # Comparative Pairs (with guaranteed 50/50 balance)
            # -----------------------------------------------------------------

            # The first half of the shuffled list becomes A -> B
            if i < split_point:
                clean_prompt, patch_prompt = prompt_a_text, prompt_b_text
                clean_answer, patch_answer = (
                    self.config.object_a,
                    self.config.object_b,
                )
            # The second half of the shuffled list becomes B -> A
            else:
                clean_prompt, patch_prompt = prompt_b_text, prompt_a_text
                clean_answer, patch_answer = (
                    self.config.object_b,
                    self.config.object_a,
                )

            self.comparative_pairs.append(
                {
                    "clean_prompt": clean_prompt,
                    "patch_prompt": patch_prompt,
                    "clean_answer": clean_answer,
                    "patch_answer": patch_answer,
                    "shared_attributes": shared_attrs,
                }
            )

            # -----------------------------------------------------------------
            # Single Object Prompts
            # -----------------------------------------------------------------
            ## Clean / Object A
            self.object_a_prompts.append(
                {
                    "prompt": prompt_a_text,
                    "answer": self.config.object_a,
                    "attributes": shared_attrs,
                }
            )

            ## Destination / Object B
            self.object_b_prompts.append(
                {
                    "prompt": prompt_b_text,
                    "answer": self.config.object_b,
                    "attributes": shared_attrs,
                }
            )

        # Log the balance for confirmation
        a_to_b_count = sum(
            1
            for p in self.comparative_pairs
            if p["clean_answer"] == self.config.object_a
        )
        b_to_a_count = sum(
            1
            for p in self.comparative_pairs
            if p["clean_answer"] == self.config.object_b
        )
        logger.info(
            f"Comparative pairs created with balance: "
            f"{a_to_b_count} (A->B) and {b_to_a_count} (B->A)."
        )

        return self

    def _format_prompt(
        self, obj: str, style: str, color: str, action: str
    ) -> str:
        """
        Formats a prompt string based on the template and provided attributes.

        Args:
            obj (str): Object name (e.g., "cat", "dog").
            style (str): Style descriptor (e.g., "photorealistic image").
            color (str): Color descriptor (e.g., "black and white").
            action (str): Action descriptor (e.g., "facing the camera").

        Returns:
            str: Formatted prompt string.
        """
        return (
            self.config.prompt_template.format(
                style=style, color=color, object=obj, action=action
            )
            .replace("  ", " ")
            .strip()
        )


# =========================================================================== #
#                              Dataset Builder                                #
# =========================================================================== #


class DatasetBuilder:
    """Builds and saves datasets from generated prompt lists."""

    def __init__(
        self, base_save_path: str, train_size: int, test_size: int, seed: int
    ) -> None:
        self.base_save_path = Path(base_save_path)
        self.train_size = train_size
        self.test_size = test_size
        self.seed = seed
        random.seed(self.seed)

    def build_and_save(self, name: str, data: List[Dict]) -> None:
        """Builds a single DatasetDict and saves it to disk.

        Args:
            name (str): Name of the dataset (used for directory naming).
            data (List[Dict]): List of prompt dictionaries to include in
                the dataset.
        """

        dataset_path = self.base_save_path / name
        os.makedirs(dataset_path, exist_ok=True)

        random.shuffle(data)

        total_needed = self.train_size + self.test_size
        assert len(data) >= total_needed, (
            f"Not enough data ({len(data)}) to fulfill "
            f"train_size ({self.train_size}) + test_size ({self.test_size})."
        )

        train_size, test_size = self.train_size, self.test_size

        train_split = data[:train_size]
        test_split = data[train_size : train_size + test_size]

        dataset_dict = DatasetDict(
            {
                "train": Dataset.from_list(train_split),
                "test": Dataset.from_list(test_split),
            }
        )

        dataset_dict.save_to_disk(str(dataset_path))
        logger.info(f"Saved Hugging Face dataset to: {dataset_path}")

        # Save JSON for inspection
        json_path = dataset_path / f"{name}_data.json"
        with open(json_path, "w") as f:
            json.dump(
                {
                    "train": train_split,
                    "test": test_split,
                    "metadata": {
                        "train_size": len(train_split),
                        "test_size": len(test_split),
                    },
                },
                f,
                indent=2,
            )
        logger.info(f"Saved JSON version to: {json_path}")

        logger.info("Example prompts:", train_split[0])


if __name__ == "__main__":
    main()
