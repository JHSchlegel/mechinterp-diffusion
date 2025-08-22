#!/bin/bash
cd ../mechinterp_diffusion/experiments

screen -S "evaluate_sae" bash -c "
        python evaluate_sae.py \
                --mode reconstruction \
                --num_prompts 1000 \
                --seeds 42 43 44 45 46 \
                --device cuda \
                --torch_dtype float16 \
                --num_inference_steps 25 \
                --guidance_scale 9.0 \
                --height 512 \
                --width 512

        python evaluate_sae.py \
                --mode feature_removal \
                --num_prompts 1000 \
                --seeds 42 43 44 45 46 \
                --device cuda \
                --torch_dtype float16 \
                --num_inference_steps 25 \
                --guidance_scale 9.0 \
                --height 512 \
                --width 512

"
