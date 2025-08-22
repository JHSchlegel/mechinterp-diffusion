#!/bin/bash
cd ../mechinterp_diffusion/scripts

screen -S "collect_latents" bash -c "
    CUDA_VISIBLE_DEVICES=0 accelerate launch collect_latents.py \
        --hook_names unet.down_blocks.2.attentions.0 unet.up_blocks.1.attentions.1 \
        --dataset_split train \
        --dataset_size 50000 \
        --batch_size 50 \
        --num_inference_steps 25

    CUDA_VISIBLE_DEVICES=0 accelerate launch collect_latents.py \
        --hook_names unet.down_blocks.2.attentions.0 unet.up_blocks.1.attentions.1 \
        --dataset_split test \
        --dataset_size 10000 \
        --batch_size 50 \
        --num_inference_steps 25

"
