"""
Script to cache activations from a pretrained Stable Diffusion model
and save them to disk.

Source:
https://github.com/cywinski/SAeUron/blob/main/SAE/cache_activations_runner_unlearn_canvas.py

Changes made to original code:
 - typing hints, docstrings, comments, and formatting according to black
 - rewrote script to combine class and main function into one script
 - adapted to work with the new config and dataset structure
 - removed the push_to_hub functionality and adjusted the dataset loading
 - included extensive logging
"""

# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import logging
import time

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import gather_object
from config import LatentsExtractionConfig
from datasets import Array2D, Dataset, DatasetDict, Features, Value
from datasets.fingerprint import generate_fingerprint
from diffusers import DDIMScheduler
from diffusers.utils import is_xformers_available
from simple_parsing import parse
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

TORCH_STRING_DTYPE_MAP = {"float16": torch.float16, "float32": torch.float32}

# accelerate logging:
logger = get_logger(__name__, log_level="INFO")


# =========================================================================== #
#                    Main Function for Latents Extraction                     #
# =========================================================================== #
def main() -> None:
    """
    Main function to extract latent activations from a pretrained Stable
    Diffusion model. Allows the file to be run as a standalone script that
    extracts activations from a pretrained model and saves them to disk.
    """
    # -------------------------------------------------------------------------
    # Parse command line arguments and run the latents extraction script
    # -------------------------------------------------------------------------

    run_start_time = time.time()
    cfg = parse(LatentsExtractionConfig)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    runner = CacheActivationsRunner(cfg)
    datasets = runner.run()

    # save config to json
    if runner.accelerator.is_main_process:
        logger.info(
            f"Successfully cached activations from {len(datasets)} hooks"
        )
        logger.info(
            f"Saved datasets to subdirectories of {cfg.extracted_latents_path}"
        )

        config_path = os.path.join(
            str(cfg.extracted_latents_path), "activations_config.json"
        )
        with open(config_path, "w") as f:
            json.dump(asdict(cfg), f, indent=4)
        logger.info(f"Configuration saved to {config_path}")
    run_end_time = time.time()
    logger.info(
        f"Activation caching run finished. Duration: "
        f"{format_duration(run_end_time - run_start_time)}"
    )


# -----------------------------------------------------------------------------
# Helper function for time formatting
# -----------------------------------------------------------------------------
def format_duration(seconds: float) -> str:
    """Formats duration in seconds into HH:MM:SS format.

    Args:
        seconds (float): Duration in seconds.

    Returns:
        str: Duration formatted as HH:MM:SS.
    """
    total_int_seconds = int(seconds)
    hours = total_int_seconds // 3600
    minutes = (total_int_seconds % 3600) // 60
    secs = total_int_seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02} ({seconds:.2f} seconds)"


# =========================================================================== #
#                      Class to Cache Latent Activations                      #
# =========================================================================== #
class CacheActivationsRunner:
    """
    Class to extract latent activations from a pretrained Stable Diffusion
    model
    """

    def __init__(self, cfg: LatentsExtractionConfig) -> None:
        """

        Args:
            cfg (LatentsExtractionConfig): _description_
        """
        self.cfg = cfg
        self.accelerator = Accelerator()

        logger.info(f"Configuration loaded: {asdict(cfg)}")
        logger.info(
            f"Accelerator state: device={self.accelerator.device}, "
            f"is_main_process={self.accelerator.is_main_process}, "
            f"num_processes={self.accelerator.num_processes}"
        )

        # hacky way to prevent initializing those objects when
        # loading activations from disk
        if self.cfg.hook_names is not None:
            from diffusion.hooked_sd_pipeline import (
                HookedStableDiffusionPipeline,
                HookedStableDiffusionXLPipeline,
            )

            if (
                self.cfg.model_name
                == "stabilityai/stable-diffusion-xl-base-1.0"
            ):
                self.pipe = HookedStableDiffusionXLPipeline.from_pretrained(
                    self.cfg.model_name,
                    torch_dtype=TORCH_STRING_DTYPE_MAP[self.cfg.dtype],
                    safety_checker=None,
                )

            else:
                try:
                    self.pipe = HookedStableDiffusionPipeline.from_pretrained(
                        self.cfg.model_name,
                        torch_dtype=TORCH_STRING_DTYPE_MAP[self.cfg.dtype],
                        safety_checker=None,
                    )

                except Exception as e:
                    logger.error(
                        f"Error loading model {self.cfg.model_name}: {e}"
                    )
                    raise e

            logger.info(f"Loaded model {self.cfg.model_name}")

            assert isinstance(
                self.pipe.scheduler, DDIMScheduler
            ), "Scheduler is not a DDIMScheduler."
            logger.info("Scheduler type verified (DDIMScheduler)")

            if is_xformers_available():
                logger.info("Enabling xFormers memory efficient attention")
                self.pipe.unet.enable_xformers_memory_efficient_attention()

            self.pipe.to(self.accelerator.device)
            self.pipe.vae.to("cpu")
            self.pipe.set_progress_bar_config(disable=True)

            self.scheduler = self.pipe.scheduler

            # Prepare timesteps
            self.scheduler.set_timesteps(
                self.cfg.num_inference_steps, device="cpu"
            )
            self.scheduler_timesteps = self.scheduler.timesteps

            self.features_dict = {
                hookpoint: None for hookpoint in self.cfg.hook_names
            }

            self._load_prompt_dataset()

            self.num_examples = len(self.dataset)

            logger.info(
                f"Loaded {self.num_examples} prompts from dataset "
                f"'{self.cfg.dataset_name}' split '{self.cfg.dataset_split}'."
            )

            self.dataloader = self.get_batches(
                self.dataset, self.cfg.batch_size
            )
            self.n_buffers = len(self.dataloader)

    def _load_prompt_dataset(self) -> None:
        """
        Load the prompt dataset from disk, shuffle it, and select a subset of
        the dataset size specified in the config.
        """

        assert self.cfg.dataset_name in [
            "laion",
            "flickr30k",
        ], "Only laion-coco-aesthetic and flickr30k are supported for now"

        assert self.cfg.dataset_split in [
            "train",
            "test",
        ], "Only train and test splits are supported for now"

        if self.cfg.dataset_name == "flickr30k":
            self.dataset = DatasetDict.load_from_disk(
                "../../../flickr30k_captions"
            )[self.cfg.dataset_split]

        else:
            logger.error(
                f"Dataset {self.cfg.dataset_name} is not implemented yet"
            )
            raise NotImplementedError(
                f"Dataset {self.cfg.dataset_name} is not implemented yet"
            )

        self.dataset = self.dataset.shuffle(seed=self.cfg.seed)
        if self.cfg.dataset_size:
            self.dataset = self.dataset.select(range(self.cfg.dataset_size))

    @staticmethod
    def get_batches(items, batch_size):
        num_batches = (len(items) + batch_size - 1) // batch_size
        batches = []

        for i in range(num_batches):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, len(items))
            batch = items[start_index:end_index]
            batches.append(batch)

        return batches

    @staticmethod
    def _consolidate_shards(
        source_dir: Path, output_dir: Path, copy_files: bool = True
    ) -> Dataset:
        """Consolidate sharded datasets into a single directory without
        rewriting data.

        Each of the shards must be of the same format, aka the full dataset
        must be able to be recreated like so:

        ```
        ds = concatenate_datasets(
            [Dataset.load_from_disk(str(shard_dir)) for shard_dir in
            sorted(source_dir.iterdir())]
        )

        ```

        Sharded dataset format:
        ```
        source_dir/
            shard_00000/
                dataset_info.json
                state.json
                data-00000-of-00002.arrow
                data-00001-of-00002.arrow
            shard_00001/
                dataset_info.json
                state.json
                data-00000-of-00001.arrow
        ```

        And flattens them into the format:

        ```
        output_dir/
            dataset_info.json
            state.json
            data-00000-of-00003.arrow
            data-00001-of-00003.arrow
            data-00002-of-00003.arrow
        ```

        allowing the dataset to be loaded like so:

        ```
        ds = datasets.load_from_disk(output_dir)
        ```

        Args:
            source_dir: Directory containing the sharded datasets
            output_dir: Directory to consolidate the shards into
            copy_files: If True, copy files; if False, move them and delete
                source_dir
        """
        first_shard_dir_name = "shard_00000"  # shard_{i:05d}

        assert source_dir.exists() and source_dir.is_dir()
        assert (
            output_dir.exists()
            and output_dir.is_dir()
            and not any(
                p for p in output_dir.iterdir() if not p.name == ".tmp_shards"
            )
        )
        if not (source_dir / first_shard_dir_name).exists():
            raise Exception(f"No shards in {source_dir} exist!")

        transfer_fn = shutil.copy2 if copy_files else shutil.move

        # Move dataset_info.json from any shard (all the same)
        transfer_fn(
            source_dir / first_shard_dir_name / "dataset_info.json",
            output_dir / "dataset_info.json",
        )

        arrow_files = []
        file_count = 0

        for shard_dir in sorted(source_dir.iterdir()):
            if not shard_dir.name.startswith("shard_"):
                continue

            # state.json contains arrow filenames
            state = json.loads((shard_dir / "state.json").read_text())

            for data_file in state["_data_files"]:
                src = shard_dir / data_file["filename"]
                shard_count = len(list(source_dir.iterdir()))
                new_name = f"data-{file_count:05d}-of-{shard_count:05d}.arrow"
                dst = output_dir / new_name
                transfer_fn(src, dst)
                arrow_files.append({"filename": new_name})
                file_count += 1

        new_state = {
            "_data_files": arrow_files,
            "_fingerprint": None,  # temporary
            "_format_columns": None,
            "_format_kwargs": {},
            "_format_type": None,
            "_output_all_columns": False,
            "_split": None,
        }

        # fingerprint is generated from dataset.__getstate__
        # (not including _fingerprint)
        with open(output_dir / "state.json", "w") as f:
            json.dump(new_state, f, indent=2)

        ds = Dataset.load_from_disk(str(output_dir))
        fingerprint = generate_fingerprint(ds)
        del ds

        with open(output_dir / "state.json", "r+") as f:
            state = json.loads(f.read())
            state["_fingerprint"] = fingerprint
            f.seek(0)
            json.dump(state, f, indent=2)
            f.truncate()

        if not copy_files:  # cleanup source dir
            shutil.rmtree(source_dir)

        return Dataset.load_from_disk(output_dir)

    @torch.no_grad()
    def _create_shard(
        self,
        # buffer shape: "bs num_inference_steps+1 d_sample_size d_in",
        buffer: torch.Tensor,
        hook_name: str,
    ) -> Dataset:
        batch_size, n_steps, d_sample_size, d_in = buffer.shape

        # Filter buffer based on every N steps
        buffer = buffer[:, :: self.cfg.extract_every_n_timesteps, :, :]

        activations = buffer.reshape(-1, d_sample_size, d_in)
        timesteps = self.scheduler_timesteps[
            :: self.cfg.extract_every_n_timesteps
        ].repeat(batch_size)

        shard = Dataset.from_dict(
            {
                "activations": activations,
                "timestep": timesteps,
            },
            features=self.features_dict[hook_name],
        )
        return shard

    def create_dataset_feature(self, hook_name, d_in, d_out):
        self.features_dict[hook_name] = Features(
            {
                "activations": Array2D(
                    shape=(
                        d_in,
                        d_out,
                    ),
                    dtype=self.cfg.dtype,
                ),
                "timestep": Value(dtype="uint16"),
            }
        )

    @torch.no_grad()
    def run(self) -> dict[str, Dataset]:
        ### Paths setup
        assert self.cfg.extracted_latents_path is not None

        final_cached_activation_paths = {
            n: Path(os.path.join(self.cfg.extracted_latents_path, n))
            for n in self.cfg.hook_names
        }

        if self.accelerator.is_main_process:
            for path in final_cached_activation_paths.values():
                path.mkdir(exist_ok=True, parents=True)
                if any(path.iterdir()):
                    raise Exception(
                        f"""Activations directory ({path}) is not empty. \
                        Please delete it or specify a different path. Exiting \
                        the script to prevent accidental deletion of files."""
                    )

            tmp_cached_activation_paths = {
                n: path / ".tmp_shards/"
                for n, path in final_cached_activation_paths.items()
            }
            for path in tmp_cached_activation_paths.values():
                path.mkdir(exist_ok=False, parents=False)

            # set up logging to output folder:
            log_file = os.path.join(
                str(self.cfg.extracted_latents_path),
                "activations_extraction.log",
            )
            logging.basicConfig(
                format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                level=logging.INFO,
                handlers=[
                    logging.StreamHandler(sys.stdout),
                    logging.FileHandler(log_file),
                ],
            )
        self.accelerator.wait_for_everyone()

        ### Create temporary sharded datasets
        if self.accelerator.is_main_process:
            logging.info(f"Started caching {self.num_examples} activations")

        for i, batch in tqdm(
            enumerate(self.dataloader),
            desc="Caching activations",
            total=self.n_buffers,
            disable=not self.accelerator.is_main_process,
        ):

            with self.accelerator.split_between_processes(batch) as prompt:
                prompt = prompt[self.cfg.column_name]

                _, acts_cache = self.pipe.run_with_cache(
                    prompt=prompt,
                    output_type="latent",
                    num_inference_steps=self.cfg.num_inference_steps,
                    save_input=(
                        True if self.cfg.output_or_diff == "diff" else False
                    ),
                    save_output=True,
                    positions_to_cache=self.cfg.hook_names,
                    guidance_scale=self.cfg.guidance_scale,
                    height=self.cfg.height,
                    width=self.cfg.width,
                    unconditional=self.cfg.unconditional,
                )

            self.accelerator.wait_for_everyone()

            # Gather and process each hook's activations separately
            gathered_buffer = {}
            for hook_name in self.cfg.hook_names:
                if self.cfg.output_or_diff == "diff":
                    gathered_buffer[hook_name] = (
                        acts_cache["output"][hook_name]
                        - acts_cache["input"][hook_name]
                    )
                else:
                    gathered_buffer[hook_name] = acts_cache["output"][
                        hook_name
                    ]

            gathered_buffer = gather_object([gathered_buffer])  # list of dicts

            if self.accelerator.is_main_process:
                for hook_name in self.cfg.hook_names:
                    gathered_buffer_acts = torch.cat(
                        [
                            gathered_buffer[i][hook_name]
                            for i in range(len(gathered_buffer))
                        ],
                        dim=0,
                    )
                    if self.features_dict[hook_name] is None:
                        self.create_dataset_feature(
                            hook_name,
                            gathered_buffer_acts.shape[-2],
                            gathered_buffer_acts.shape[-1],
                        )

                    logging.info(f"{hook_name=} {gathered_buffer_acts.shape=}")

                    shard = self._create_shard(gathered_buffer_acts, hook_name)
                    shard_path = os.path.join(
                        tmp_cached_activation_paths[hook_name],
                        f"shard_{i:05d}",
                    )
                    shard.save_to_disk(
                        shard_path,
                        num_shards=1,
                    )
                    del gathered_buffer_acts, shard
                del gathered_buffer

        ## Concat sharded datasets together and shuffle
        datasets = {}

        if self.accelerator.is_main_process:
            for hook_name, path in tmp_cached_activation_paths.items():
                datasets[hook_name] = self._consolidate_shards(
                    path,
                    final_cached_activation_paths[hook_name],
                    copy_files=False,
                )
                logging.info(f"Consolidated the dataset for hook {hook_name}")

        return datasets


if __name__ == "__main__":
    main()
