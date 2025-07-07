#!/bin/bash
cd ../mechinterp_diffusion/scripts

#screen -S "collect_top_activating_examples" bash -c "
/usr/bin/time -v    python collect_top_dataset_examples.py \
        --output_dir ../../results/top_examples/down_blocks.2.attentions.0/ \
        --sae_paths ../../checkpoints/sae/down_blocks.2.attentions.0/TopKSAE_dsae-5120_timesteps-all_20250706_201216/step_9766 ../../checkpoints/sae/up_blocks.1.attentions.1/TopKSAE_dsae-5120_timesteps-all_20250706_221127/step_9766 \
        --target_modules unet.down_blocks.2.attentions.0 unet.up_blocks.1.attentions.1 \
        --num_prompts 5 \
        --num_examples_per_feature 9 \
        --number_of_seeds 10
#"
