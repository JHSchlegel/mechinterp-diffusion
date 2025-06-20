#!/bin/bash
cd ../mechinterp_diffusion/multistep_sae/scripts

screen -S "extract_probe_latents" bash -c "
    CUDA_VISIBLE_DEVICES=0 accelerate launch collect_latents.py \
        --hook_names unet.down_blocks.2.attentions.0 \
        --dataset_name birds_vs_cats \
        --dataset_split train \
        --dataset_size 10000 \
        --batch_size 50 \
        --output_or_diff "output" \
        --num_inference_steps 25 \
        --target_timesteps_idx 4

    CUDA_VISIBLE_DEVICES=0 accelerate launch collect_latents.py \
        --hook_names unet.down_blocks.2.attentions.0 \
        --dataset_name birds_vs_cats \
        --dataset_split test \
        --dataset_size 2000 \
        --batch_size 50 \
        --output_or_diff "output" \
        --num_inference_steps 25 \
        --target_timesteps_idx 4
"
