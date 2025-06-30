"""
This script caches SAE activations for a given set of samples, aggregates 
them across samples and visualizes them.


Usage example:
python mechinterp_diffusion/experiments/visualize_sae_spatial.py \
    --model_path path/to/model
"""

# TODO: recompute option if not cached
# TODO: Avg. number of active latents -> don't use topk
# TODO: Log feature density
# TODO: Sharding?
# TODO: features activating differently across time?
# TODO: How to do density ridges if too samples?


# =========================================================================== #
#                            Packages and Presets                             #
# =========================================================================== #


import os
import argparse
import gc
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union, List,Tuple
from umap import UMAP

import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn.functional as F
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm

# Caching:
import h5py

sys.path.append(str(Path(__file__).parent.parent))

from config import TopKSAEConfig
from core.sae.base_sae import BaseSAE
from core.sae.topk_sae import TopKSAE
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


PAPER_COLORS = {
    # see: https://nanx.me/ggsci/reference/pal_jama.html
    "jama": [
        "#DF8F44FF",
        "#00A1D5FF",
        "#B24745FF",
        "#374E55FF",
        "#79AF97FF",
        "#6A6599FF",
        "#80796BFF",
    ],
    # see: https://nanx.me/ggsci/reference/pal_aaas.html
    "aaas": [
        "#3B4992FF",
        "#EE0000FF",
        "#008B45FF",
        "#631879FF",
        "#008280FF",
        "#BB0021FF",
        "#5F559BFF",
        "#A20056FF",
        "#808180FF",
        "#1B1919FF",
    ],
}
plt.style.use('default')


# =========================================================================== #
#                            Main Functionality                               #
# =========================================================================== #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot SAE Activations"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved SAE model",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset in Hugging Face format",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the evaluation results",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate per timestep. "
        "If None, evaluates all samples.",
    )
    
    parser.add_argument(
        "--colors",
        type=str,
        choices=["jama", "aaas"],
        default="aaas",
        help="Color palette to use for plots",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for evaluation."
    )
    
    parser.add_argument(
        "spatial_resolution",
        type=int,
        nargs=2,
        default=[16, 16],
        help="Spatial resolution of latents (height, width)."
    )
        

    args = parser.parse_args()
    sae = TopKSAE.load_from_disk(
        args.model_path,
        config_class=TopKSAEConfig,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    test_dataset = load_from_disk(args.dataset_path)
    
    spatial_visualizer = SpatialVisualizer(
        sae=sae,
        dataset=test_dataset,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_samples=args.num_samples,
        resolution=args.spatial_resolution,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
    spatial_visualizer.run()


class SpatialVisualizer:
    def __init__(
        self,
        sae: BaseSAE,
        dataset: Dataset,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        apply_topk: bool = True,
        load_from_cache: bool = True,
        num_samples: Optional[int] = None,
        resolution: List[int] = [16, 16],
        batch_size: int = 100,
        output_dir: Union[str, Path] = "../../results/evaluation_results",
    ) -> None:
        """
        Initialize the Visualizer for Sparse Autoencoders (SAEs) Activations.
        
        Args:
            sae (BaseSAE): Sparse Autoencoder to evaluate.
            dataset (Dataset): Test dataset in Hugging Face format.
            device (str, optional): Device to run evaluaiton on. Defaults to 
                "cuda" if it is available, otherwise "cpu".
            apply_topk (bool, optional): Whether to apply top-k selection on
                activations. Defaults to True.
            load_from_cache (bool, optional): Whether to extract 
                activations or load from cache. Defaults to True. If set to
                False, it will rerun extraction and overwrite the cache.
            num_samples (Optional[int], optional): Number of test samples to 
                use for evaluation. Defaults to None.
            resolution (List[int], optional): Spatial resolution of the latents
                (height, width). Defaults to [16, 16].
            batch_size (int, optional): Batch size to use for test loaders.
                Defaults to 100.
            output_dir (Union[str, Path], optional): Directory to save the
                evaluation results. Defaults to 
                "../../results/evaluation_results".
        """
        num_timesteps = len(np.unique(dataset["timestep"]))
        self.max_timestep = max(dataset["timestep"]).item()
        assert (
            num_samples is None or num_samples <= len(dataset) // num_timesteps
        ), "num_samples must be less than or equal to number of test examples"
        
        self.resolution = resolution
        self.sae = sae.to(device=device, dtype=torch.float32)
        self.sae.eval()
        self.dataset = dataset
        self.device = torch.device(device)
        self.timesteps = sorted(
            list(set([ts.item() for ts in dataset["timestep"]]))
        )
        self.apply_topk = apply_topk

        # Subset test dataset:
        if num_samples is None:
            self.total_samples = len(self.dataset) // len(self.timesteps)
        else:
            self.total_samples = min(
                len(self.dataset), num_samples * len(self.timesteps)
            )
            
        # test set ordered by timestep -> can just take the first n samples
        self.dataset = self.dataset.select(range(self.total_samples))
        self.dataset.set_format(
            type = "torch",
            columns = ["activations", "timestep"]
        )

        # avoid distribution of across time
        self.batch_size = 1 if self.sae.use_batch_topk else batch_size

        self.sae.eval()
        
        self.output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        self.cache_file = self.output_dir / "sae_activations.h5"
        # Rerun activation extraction if cache file does not exist or if 
        # specified to not load from cache.
        self.run_activation_extraction = (
            not load_from_cache or not self.cache_file.exists()
        )
        
    
    def run(self) -> None:
        """Run the visualization pipeline."""
        logger.info("-" * 50)
        logger.info("Starting Spatial Activation Visualization".center(50))
        logger.info("-" * 50)
        
        # Process samples and get aggregated data
        aggregated_data, feature_data = self._process_samples()
        
        # Create visualizations
        logger.info("Creating visualizations...")
        self._plot_density_ridges(aggregated_data)
        self._plot_spatial_evolution(aggregated_data)
        self._plot_aggregated_patterns(aggregated_data)
        self._plot_top_features(feature_data)
        
        # Save summary
        self._save_summary()
        
        logger.info("-" * 50)
        logger.info("Visualization Complete".center(50))
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("-" * 50)
    
    def _plot_density_ridges(self, data: Dict[int, Dict[str, np.ndarray]]) -> None:
        raise NotImplementedError()
    
    def _plot_spatial_evolution(self, data: Dict[int, Dict[str, np.ndarray]]) -> None:
        raise NotImplementedError()
    
    def _plot_aggregated_patterns(self, data: Dict[int, Dict[str, np.ndarray]]) -> None:
        raise NotImplementedError()
    
    def _plot_top_features(self, data: Dict[str, np.ndarray]) -> None:
        raise NotImplementedError()
        
    @torch.no_grad()
    def _process_samples(self) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Process samples, aggregate activations, and cache them.
        """
        self.sae.eval()
        if not self.run_activation_extraction:
            data = {}
            with h5py.File(self.cache_file, "r") as f:
                for t in self.timesteps:
                    data[t] = {
                        "spatial_mean" : f["t{t}/mean"][:],
                        # "values_sample": f["t{t}/values"][:],
                        "spatial_nonzeros": f[f"t{t}/spatial_nonzeros"][:],
                        
                    }
                data["features"] = {
                    # [n_features, h, w]
                    "spatial_mean" : f["features/spatial_mean"][:],
                    # [n_features, h, w]
                    "spatial_variance": f["features/spatial_variance"][:],
                    # [n_features]
                    "mean_activation": f["features/mean_activation"][:],
                    # [n_features]
                    "feature_ranking": f["features/feature_ranking"][:],
                }
            return data
        
        
        logger.info(
            f"Loading cached activations from {self.cache_file}"
        )
        h, w = self.resolution
        data = {
            t: {
                "spatial_mean": np.zeros((h, w)),
                "spatial_nonzeros": np.zeros((h, w)),
                "values_sample": [],
                "nonzero_count": 0,
                "count" : 0
            } for t in self.timesteps
        }
        
        feature_stats = {
            "sum": np.zeros((self.sae.cfg.d_sae, h, w)),
            "sum_sq": np.zeros((self.sae.cfg.d_sae, h, w)),
            "count": 0
        }
        
        
        for timestep in tqdm(self.timesteps, desc="Evaluating timesteps"):
            # Get indices for this timestep
            timestep_indices = [
                i
                for i, ts in enumerate(self.dataset["timestep"])
                if ts.item() == timestep
            ]
            
            timestep_dataset = self.dataset.select(timestep_indices)
            
            # Create separate test loaders for each timestep
            test_loader = DataLoader(
                timestep_dataset,
                batch_size=self.batch_size,
                shuffle=False,
            )
            
            for batch_idx, batch in enumerate(test_loader):
                acts = batch["activations"].to(
                    dtype=torch.float32, device=self.device
                ) #[bs, h*w, c]
                bs = acts.shape[0]
                
                
                sae_input, _ = self.sae.preprocess_input(acts)
                sae_acts: Tensor = F.relu(self.sae.encode(sae_input))
                
                # Apply top-k selection if configured
                if self.apply_topk:
                    sae_acts, _ = self.sae._get_topk(
                        sae_acts, 
                        k=self.sae.cfg.k
                    )
                
                del sae_input, acts
                
                # -------------------------------------------------------------
                # Timestep-specific updates
                # -------------------------------------------------------------
                sae_acts_spatial = sae_acts.reshape(
                        bs, h, w, -1
                    ) # [bs, h, w, d_sae]
                
                # [h, w]
                spatial_mean = sae_acts_spatial.mean(dim = (0, 1)).cpu().numpy() 
                spatial_nonzeros = (sae_acts_spatial > 0.0).sum(dim=(0, -1)).item()
                                
                # Update running statistics:
                if data[timestep]["count"] == 0:
                    data[timestep]["spatial_mean"] = spatial_mean
                    data[timestep]["spatial_nonzeros"] = spatial_nonzeros
                else:
                    n = data[timestep]["count"]
                    data[timestep]["spatial_mean"] = (
                        data[timestep]["spatial_mean"] * n
                        + spatial_mean
                    ) / (n + bs)
                    data[timestep]["spatial_nonzeros"] = (
                        data[timestep]["spatial_nonzeros"] * n
                        + spatial_nonzeros
                    ) / (n + bs)
                    
                
                
                data[timestep]
                data[timestep]["count"] += bs
                feature_np = sae_acts_spatial.permute(3, 0, 1, 2).cpu().numpy()
                feature_stats["sum"] += feature_np.sum(axis=1)  # [d_sae, h, w]
                feature_stats["sum_sq"] += (feature_np ** 2).sum(axis=1) # [d_sae, h, w]
                feature_stats["count"] += bs
                
                # Clear GPU periodically
                if batch_idx > 0 and batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
            
            
            gc.collect()
            torch.cuda.empty_cache()
        
        # ---------------------------------------------------------------------
        # Aggregate feature statistics
        # ---------------------------------------------------------------------
        n = feature_stats['count']
        
        # Spatial mean and variance for features
        feature_mean = feature_stats['sum'] / n  # [d_sae, h, w]
        feature_variance = (feature_stats['sum_sq'] / n) - feature_mean ** 2
        feature_variance = np.maximum(feature_variance, 0) # psd
        
        # Collapse to single number for ranking:
        mean_activation = feature_mean.mean(axis=(1, 2))  # [d_sae]
        feature_ranking = np.argsort(mean_activation)[::-1]
        
        data["features"] = {
            "spatial_mean": feature_mean,
            "spatial_variance": feature_variance,
            "mean_activation": mean_activation,
            "feature_ranking": feature_ranking,
        }
        
        with h5py.File(self.cache_file, "w") as f:
            for t in self.timesteps:
                group = f.create_group(f"t{t}")
                group.create_dataset(
                    "mean", data=data[t]["spatial_mean"]
                )
                # group.create_dataset(
                #     "values", data=np.array(data[t]["values_sample"])
                # )

                group.create_dataset(
                    "spatial_nonzeros", data=data[t]["spatial_nonzeros"]
                )
                
                del data[t]["nonzero_count"], data[t]["count"]
            
            features_group = f.create_group("features")
            features_group.create_dataset(
                "spatial_mean", data=data["features"]["spatial_mean"]
            ) 
            features_group.create_dataset(
                "spatial_variance", data=data["features"]["spatial_variance"]
            )
            features_group.create_dataset(
                "mean_activation", data=data["features"]["mean_activation"]
            )
            features_group.create_dataset(
                "feature_ranking", data=data["features"]["feature_ranking"]
            )
        
        logger.info(f"Cached to {self.cache_file}")
        
        gc.collect()
        torch.cuda.empty_cache()
        
        return data
    
    def _convert_timestep_to_diffusion_time(
        self, timestep: int
    ) -> float:
        """
        Convert discrete timestep to normalized diffusion time for plotting.

        Args:
            timestep (int): The current timestep (1-indexed).

        Returns:
            float: Normalized diffusion time in the range [0, 1].
        """
        return (timestep - 1) / (self.max_timestep - 1)
    
    def _save_summary() -> None:
        raise NotImplementedError()
    
    
if __name__ == "__main__":
    main()