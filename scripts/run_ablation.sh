#!/bin/bash
cd ../mechinterp_diffusion/experiments

# Quick test
python ablate_sae_hydra.py trainer.num_tokens=1000 num_eval_samples=100

# Small ablation
# python ablate_sae_hydra.py -m sae.k=10,20,32 trainer.num_tokens=1000000 seed=42,43,44 num_eval_samples=5000

# Full ablation
# python ablate_sae_hydra.py -m sae.k=10,20,32,64 trainer.num_tokens=50000000 seed=42,43,44,45,46 num_eval_samples=10000

# Visualize results (replace with your actual results directory)
# python visualize_ablation_results.py --results_dir ../results/ablation/topk_oaat_20250612_151230
