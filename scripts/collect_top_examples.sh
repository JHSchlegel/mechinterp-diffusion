#!/bin/bash
cd ../mechinterp_diffusion/scripts

screen -S "collect_top_activating_examples" bash -c "
    python collect_top_dataset_examples.py \
        --target_module unet.down_blocks.2.attentions.0 \
        --num_prompts 5000 \
        --num_examples_per_feature 9
"
