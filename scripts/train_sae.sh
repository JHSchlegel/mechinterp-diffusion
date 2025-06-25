#!/bin/bash
cd ../mechinterp_diffusion/scripts

screen -S "train_sae" bash -c "
    python train_sae.py TopK \
        --dataset_path ../../data/activations/stable-diffusion-2-1/laion/train/subset_size-50000/25-inference-steps/every-1-steps/unet.down_blocks.2.attentions.0 \
        --checkpoint_path ../../checkpoints/sae/down_blocks.2.attentions.0/
    python train_sae.py TopK --dataset_path ../../data/activations/stable-diffusion-2-1/laion/train/subset_size-50000/25-inference-steps/every-1-steps/unet.up_blocks.1.attentions.1 \
        --checkpoint_path ../../checkpoints/sae/up_blocks.1.attentions.1/

"
