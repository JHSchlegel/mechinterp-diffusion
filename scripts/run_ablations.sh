#!/bin/bash

cd ../mechinterp_diffusion/experiments

echo "Starting ablations..."

python ablate_sae.py --config-name sweep_k --multirun
python ablate_sae.py --config-name sweep_batch_size --multirun
python ablate_sae.py --config-name sweep_learning_rate --multirun
python ablate_sae.py --config-name sweep_d_sae --multirun
python ablate_sae.py --config-name sweep_batch_topk --multirun

echo "Done."
