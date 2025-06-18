"""
Script to create prompt dataset for sparse feature circuit discovery.
"""

import itertools
import os

# =========================================================================== #
#                             Packages and Presets                            #
# =========================================================================== #
import random
from pathlib import Path

from datasets import Dataset, DatasetDict

BIRDS = [
    "sparrow",
    "owl",
    "crow",
    "dove",
    "hawk",
    "pigeon",
    "finch",
    "thrush",
    "wren",
    "starling",
    "blackbird",
    "grouse",
    "quail",
]
CATS = [
    "tabby cat",
    "kitten",
    "kitty",
    "domestic cat",
    "street cat",
    "siamese cat",
    "persian cat",
    "maine coon",
    "calico cat",
    "british shorthair",
    "ragdoll cat",
    "bengal cat",
    "gray cat",
    "tuxedo cat",
    "hairless cat",
]
COLORS = ["white", "black", "orange", "brown", "grey"]
ACTIONS = [
    "in profile",
    "facing camera",
    "in side view",
    "at rest",
    "sitting still",
    "eating",
]
STYLES = [
    "photorealistic image",
    "oil painting",
    "watercolor painting",
    "digital art image",
    "sketch",
    "close up image",
    "portrait image",
]


# =========================================================================== #
#                              Dataset Creation                               #
# =========================================================================== #
def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Create a dataset of prompts in birds vs cats setting."
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=2000,
        help="Size of train set. Defaults to 2000.",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=500,
        help="Size of test set. Defaults to 500.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed to use. Defaults to 42.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent.parent.parent
            / "data/birds_vs_cats_dataset"
        ),
        help="Path to save dataset. Defaults to 'data/birds_vs_cats_dataset'.",
    )

    args = parser.parse_args()
    create_dataset(
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
        save_path=args.save_path,
    )


def create_dataset(
    train_size: int = 2_000,
    test_size: int = 500,
    seed: int = 42,
    save_path: str = str(
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "birds_vs_cats_dataset"
    ),
) -> DatasetDict:
    """Create a dataset of prompts for circuit discovery in birds vs cats
        setting.

    Args:
        train_size (int, optional): Size of train set. Defaults to 2_000.
        test_size (int, optional): Size of test set. Defaults to 500.
        seed (int, optional): Random seed to use. Defaults to 42.

    Returns:
        DatasetDict: A Hugging Face DatasetDict object containing the prompts.
    """
    assert train_size % 2 == 0, "Train size must be even."
    assert test_size % 2 == 0, "Test size must be even."

    random.seed(seed)

    bird_prompts = [
        f"A {style} {color} {bird} that is {action}."
        for color, bird, action, style in itertools.product(
            COLORS, BIRDS, ACTIONS, STYLES
        )
    ]
    cat_prompts = [
        f"A {style} {color} {cat} that is {action}."
        for color, cat, action, style in itertools.product(
            COLORS, CATS, ACTIONS, STYLES
        )
    ]

    random.shuffle(bird_prompts)
    random.shuffle(cat_prompts)

    # calculate number of samples per class:
    n_train = train_size // 2
    n_test = test_size // 2

    train_data = {
        "prompt": bird_prompts[:n_train] + cat_prompts[:n_train],
        "label": [0] * n_train + [1] * n_train,  # 0 for birds, 1 for cats
        "class_name": ["bird"] * n_train + ["cat"] * n_train,
    }
    test_prompts = {
        "prompt": bird_prompts[n_train : n_train + n_test]
        + cat_prompts[n_train : n_train + n_test],
        "label": [0] * n_test + [1] * n_test,  # 0 for birds, 1 for cats
        "class_name": ["bird"] * n_test + ["cat"] * n_test,
    }

    train_dataset = Dataset.from_dict(train_data).shuffle(seed=seed)
    test_dataset = Dataset.from_dict(test_prompts).shuffle(seed=seed)

    dataset = DatasetDict(
        {
            "train": train_dataset,
            "test": test_dataset,
        }
    )
    os.makedirs(save_path, exist_ok=True)
    dataset.save_to_disk(save_path)

    return dataset


if __name__ == "__main__":
    main()
