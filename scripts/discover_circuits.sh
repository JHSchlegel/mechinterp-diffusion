#!/bin/bash
cd ../mechinterp_diffusion/circuit_discovery

screen -S "discover_circuits" bash -c "
    python discover_circuits.py \
        --num_prompts 50 \
        --num_seeds 2 \
        --top_k_jvp_nodes 10 \
        --ig_steps 10
"
