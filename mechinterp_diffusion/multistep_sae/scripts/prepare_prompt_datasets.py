"""
This script creates the prompt datasets for the multistep diffusion models.
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import argparse
import os
import random

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset


# =========================================================================== #
#                            Main Execution Function                          #
# =========================================================================== #
def main() -> None:
    """Main function to create prompt datasets."""
    args = parse_args()

    if args.dataset_name == "flickr30k":
        sample_flickr30k_captions(
            num_train_samples=args.num_train_samples,
            num_test_samples=args.num_test_samples,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    elif args.dataset_name == "laion":
        sample_laion_captions(
            num_train_samples=args.num_train_samples,
            num_test_samples=args.num_test_samples,
            output_dir=args.output_dir,
            seed=args.seed,
        )


# -----------------------------------------------------------------------------
# Command line argument parsing
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sample and split captions")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="laion",
        choices=["flickr30k", "laion"],
        help="Name of the dataset to sample from.",
    )
    parser.add_argument(
        "--num_train_samples",
        type=int,
        default=100_000,
        help="Total number of train caption samples to extract.",
    )

    parser.add_argument(
        "--num_test_samples",
        type=int,
        default=50_000,
        help="Total number of test caption samples to extract.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../../laion-coco_captions",
        help="Directory to save the split dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    return parser.parse_args()


# =========================================================================== #
#                     Prompt Dataset Creation Functions                       #
# =========================================================================== #


# -----------------------------------------------------------------------------
# Function to sample and save Flickr30k captions
# -----------------------------------------------------------------------------
def sample_flickr30k_captions(
    num_train_samples: int = 100_000,
    num_test_samples: int = 50_000,
    output_dir: str = "../../../flickr30k_captions",
    seed: int = 42,
) -> None:
    """Create disjoint train and test sets from Flickr30k captions
    and save them to disk in HuggingFace format.

    Args:
        num_train_samples (int, optional): Size of train set.
            Defaults to 100_000.
        num_test_samples (int, optional): Size of test set.
            Defaults to 50_000.
        output_dir (str, optional): Output directory for dataset.
            Defaults to "../../../flickr30k_captions".
        seed (int, optional): Seed for reproducibility. Defaults to 42.
    """
    print("Loading Flickr30k dataset...")
    dataset = load_dataset("nlphuji/flickr30k", split="test")
    print(f"Loaded {len(dataset)} images with 5 captions each from Flickr30k")

    total_samples_requested = num_train_samples + num_test_samples

    # Extract captions
    all_captions = []

    for idx in range(len(dataset)):
        example = dataset[idx]
        for caption_idx in range(len(example["caption"])):
            caption = example["caption"][caption_idx].replace('"', "")
            all_captions.append(
                {
                    "caption": caption,
                }
            )

    # Shuffle the dataset
    random.seed(seed)
    random.shuffle(all_captions)

    # ensure we have enough captions:
    if len(all_captions) < total_samples_requested:
        raise ValueError(
            f"Requested {total_samples_requested} samples, "
            f"but only {len(all_captions)} are available."
        )

    train_captions = all_captions[:num_train_samples]
    test_captions = all_captions[num_train_samples:total_samples_requested]

    train_dataset = Dataset.from_dict(
        {"caption": [item["caption"] for item in train_captions]}
    )

    test_dataset = Dataset.from_dict(
        {"caption": [item["caption"] for item in test_captions]}
    )

    dataset_dict = DatasetDict(
        {
            "train": train_dataset,
            "test": test_dataset,
        }
    )

    os.makedirs(output_dir, exist_ok=True)

    # save in huggingface format:
    dataset_dict.save_to_disk(output_dir)
    print(f"Dataset saved to: {output_dir}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Also save as CSV for convenience during inspection:
    train_csv_path = os.path.join(output_dir, "train.csv")
    test_csv_path = os.path.join(output_dir, "test.csv")
    pd.DataFrame(
        {"caption": [item["caption"] for item in train_captions]}
    ).to_csv(train_csv_path, index=False)
    pd.DataFrame(
        {"caption": [item["caption"] for item in test_captions]}
    ).to_csv(test_csv_path, index=False)
    print(f"Train captions saved as CSV: {train_csv_path}")
    print(f"Test captions saved as CSV: {test_csv_path}")

    print("Sample captions:")
    for i in range(5):
        print(f"Train: {train_captions[i]['caption']}")
        print(f"Test: {test_captions[i]['caption']}")


# -----------------------------------------------------------------------------
# Function to sample and save Laion-coco captions
# -----------------------------------------------------------------------------
def sample_laion_captions(
    num_train_samples: int = 200_000,
    num_test_samples: int = 50_000,
    output_dir: str = "../../../laion-coco_captions",
    seed: int = 42,
) -> None:
    """Create disjoint train and test sets from Laion-Coco captions
    and save them to disk in HuggingFace format.

    Args:
        num_train_samples (int, optional): Size of train set.
            Defaults to 100_000.
        num_test_samples (int, optional): Size of test set.
            Defaults to 50_000.
        output_dir (str, optional): Output directory for dataset.
            Defaults to "../../../laion-coco_captions".
        seed (int, optional): Seed for reproducibility. Defaults to 42.
    """
    dataset = load_dataset(
        "guangyil/laion-coco-aesthetic",
        split="train",
        columns=["caption"],
        streaming=True,
    ).shuffle(seed=seed)

    train_dataset = list(dataset.take(num_train_samples))
    train_dataset = Dataset.from_dict(
        {"caption": [item["caption"] for item in train_dataset]}
    )
    test_dataset = list(dataset.skip(num_train_samples).take(num_test_samples))
    test_dataset = Dataset.from_dict(
        {"caption": [item["caption"] for item in test_dataset]}
    )
    dataset_dict = DatasetDict(
        {
            "train": train_dataset,
            "test": test_dataset,
        }
    )
    os.makedirs(output_dir, exist_ok=True)

    # save in  huggingface format:
    dataset_dict.save_to_disk(output_dir)
    print(f"Dataset saved to: {output_dir}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    train_csv_path = os.path.join(output_dir, "train.csv")
    test_csv_path = os.path.join(output_dir, "test.csv")
    pd.DataFrame(
        {"caption": [item["caption"] for item in train_dataset]}
    ).to_csv(train_csv_path, index=False)
    pd.DataFrame(
        {"caption": [item["caption"] for item in test_dataset]}
    ).to_csv(test_csv_path, index=False)
    print(f"Train captions saved as CSV: {train_csv_path}")
    print(f"Test captions saved as CSV: {test_csv_path}")

    print("Sample captions:")
    for i in range(5):
        print(f"Train: {train_dataset[i]['caption']}")
        print(f"Test: {test_dataset[i]['caption']}")


if __name__ == "__main__":
    main()
