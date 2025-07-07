#!/bin/bash
cd ../mechinterp_diffusion/scripts

#screen -S "collect_top_activating_examples" bash -c "
python collect_top_dataset_examples.py \
    --sae_path ../../checkpoints/sae/down_blocks.2.attentions.0/TopKSAE_dsae-5120_timesteps-all_20250706_201216/step_9766 \
    --output_dir ../../results/top_examples/down_blocks.2.attentions.0/ \
    --target_module unet.down_blocks.2.attentions.0 \
    --num_prompts 100 \
    --num_examples_per_feature 9
#"
