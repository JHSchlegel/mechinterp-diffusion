#!/bin/bash

cd ../mechinterp_diffusion/experiments

echo "Starting ablations..."

screen -S "run_ablations" bash -c "
    python ablate_sae.py --config-name sweep_k --multirun
    python visualize_ablation_results.py --color aaas --use_std


    python ablate_sae.py --config-name sweep_batch_size --multirun
    python visualize_ablation_results.py --color aaas --use_std


    python ablate_sae.py --config-name sweep_learning_rate --multirun
    python visualize_ablation_results.py --color aaas --use_std


    python ablate_sae.py --config-name sweep_d_sae --multirun
    python visualize_ablation_results.py --color aaas --use_std


    python ablate_sae.py --config-name sweep_batch_topk --multirun
    python visualize_ablation_results.py --color aaas --use_std
"

echo "Done."
