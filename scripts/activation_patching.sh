#!/bin/bash
cd /media/Thesis/mechinterp-diffusion/mechinterp_diffusion/experiments/

screen -S "activation_patching" bash -c "
    for res in '--height 768 --width 768' '--height 512 --width 512'; do
        python activation_patching.py \$res
        python activation_patching.py --patching_timestep_indices 0 1 2 3 4 5 6 7 8 9 \$res
        python activation_patching.py --patching_timestep_indices 0 1 2 3 4 \$res
        python activation_patching.py --patching_timestep_indices 5 6 7 8 9 \$res
        python activation_patching.py --patching_timestep_indices 10 11 12 13 14 \$res
        python activation_patching.py --patching_timestep_indices 15 16 17 18 19 20 21 22 23 24 \$res
    done
"
