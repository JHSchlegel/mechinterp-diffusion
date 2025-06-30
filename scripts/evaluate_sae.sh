#!/bin/bash
cd ../mechinterp_diffusion/experiments

screen -S "evaluate_sae" bash -c "
    python evaluate_sae.py \
        --model_path ../../checkpoints/sae/TopKSAE_dsae-5120_timesteps-all_20250523_212803/step_488282 \
        --dataset_path ../../data/activations/stable-diffusion-2-1/laion/test/subset_size-50000/25-inference-steps/every-1-steps/unet.down_blocks.2.attentions.0 \
        --output_dir ../../results/sae/evaluation/down_blocks.2.attentions.0/ \
        --num_samples 10000 \
        --colors aaas \
        --batch_size 100 \
        --spatial_resolution 16,16
"
