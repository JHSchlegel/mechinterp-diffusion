#!/bin/bash
cd ../mechinterp_diffusion/experiments

# Feature intervention example
python intervene_sae.py \
    --features 2958 \
    --intervention_values 15.0 \
    --hook_type scale \
    --seed 42 \
    --intervention_mode trajectory \
    --save_activation_heatmap \
    --prompts "A pair of black boots with a silver buckle."

# Grid intervention example
python intervene_sae.py \
    --intervention_mode grid \
    --hook_type scale \
    --features 1756 2818 3008 \
    --prompts "A raccoon wearing a tuxedo." \
    --intervention_values 10.0 25.0 50.0 200.0 \
    --timesteps 19 20 21 22 23 24

# Most important features example
python intervene_sae.py \
    --intervention_mode topk_trace \
    --topk_trace_k 10 \
    --hook_type reconstruct \
    --prompts "completely blue image"
