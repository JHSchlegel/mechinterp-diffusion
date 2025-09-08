# Mechanistic Interpretability in Text-to-Image Diffusion Models

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/) [![Diffusers](https://img.shields.io/badge/🤗%20Diffusers-FF6F00?style=flat)](https://github.com/huggingface/diffusers) [![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A framework for understanding the internal mechanisms of text-to-image diffusion models through Sparse Autoencoders (SAEs) and automated sparse feature circuit discovery. This work enables researchers to identify and analyze the specific model components responsible for generating particular visual concepts, providing interpretable insights into how diffusion models transform text into images over time.

<table>
<tr>
<td width="50%" align="center">
<b>Circuit Discovery</b><br>
<img src="img/birds_vs_cats_circuit.png" width="400px"><br>
<i>Discovered circuit distinguishing bird/cat generation</i>
</td>
<td width="50%" align="center">
<b>Activation Patching</b><br>
<img src="img/down20_512_ap.png" width="400px"><br>
<i>Effects of patching SD v2.1 down.2.0 SAE features</i>
</td>
</tr>
</table>

## Installation

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/jhschlegel/mechinterp-diffusion.git
cd mechinterp-diffusion
uv pip install -e .
```

## Usage Workflow

The following sections describe the main workflow for using this repository. For more detailed usage, please refer to the scripts in the `scripts/` directory as well as to the docstrings in the beginning of the python scripts.

### 1. Prepare Prompt Datasets

Before we can extract activations or train models, we need to prepare the prompt datasets. This is done using the `prepare_prompt_datasets.py` script.

```bash
cd mechinterp_diffusion/scripts
python prepare_prompt_datasets.py
```

This will create the necessary datasets for the following steps. For circuit discovery, a specific dataset is needed, which can be created with `prepare_circuit_prompt_dataset.py`.

### 2. Activation Extraction

Extract and cache activations from diffusion models at specific layers and timesteps. The `collect_latents.sh` script automates this process.

```bash
cd scripts
./collect_latents.sh
```

This script runs `collect_latents.py` to generate and cache activations from the diffusion model for the specified prompts. The script is configured to collect activations for both the training and testing datasets.

### 3. SAE Training

Train Sparse Autoencoders to learn interpretable features from the cached activations. The `train_sae.sh` script automates this process.

```bash
cd scripts
./train_sae.sh
```

This script trains SAEs on the cached activations for different layers of the diffusion model. The trained SAEs are saved to the `checkpoints/` directory.

### 4. Feature Interventions

Test the causal influence of learned features on the final image through targeted interventions in the diffusion process. The `intervene_sae.sh` script provides a comprehensive set of experiments.

```bash
cd scripts
./intervene_sae.sh
```

This script runs a variety of intervention experiments, including trajectory analysis and grid searches over intervention strengths.

### 5. Circuit Discovery

This workflow discovers circuits of causally important features for generating a specific concept.

#### 5a. Prepare Prompt Dataset for Probing

First, prepare a prompt dataset for the probing task. This is a binary classification task (e.g., "birds vs. cats").

```bash
cd mechinterp_diffusion/scripts
python prepare_circuit_prompt_dataset.py
```

#### 5b. Extract Activations for Probing

Next, extract the latent activations for the prompt dataset we just created.

```bash
cd scripts
./collect_latents_for_probe.sh
```

This script collects latent activations for the binary classification task.

#### 5c. Train Probe

Train a latent probe whose logit differences are used to identify influential nodes for birds vs cats generation:

```bash
cd mechinterp_diffusion/circuit_discovery
python train_probe.py
```

This script trains a simple CNN probe on the dataset created in the previous step.

#### 5d. Discover Circuits

Finally, discover the circuit of features that are causally responsible for the probe's classification.

```bash
cd scripts
./discover_circuits.sh
```

This script uses integrated gradients and vector-Jacobian products to identify the most important features and their connections.

### 6. Activation Patching

Validate the discovered circuits by patching the activations of the identified features between different images.

```bash
cd scripts
./activation_patching.sh
```

This script runs a series of activation patching experiments, providing a qualitative validation of the discovered circuits.

## Repository Structure

```
mechinterp-diffusion/
├── mechinterp_diffusion/
│   ├── __init__.py                         
│   ├── config.py                           # Centralized configuration dataclasses for training and intervention
│   ├── core/                               
│   │   ├── diffusion/                      
│   │   │   ├── hooked_sd_pipeline.py       # Stable Diffusion with activation hooks
│   │   │   └── hooked_scheduler.py         # Scheduler with caching support
│   │   ├── sae/                            
│   │   │   ├── base_sae.py                 # Abstract SAE base class
│   │   │   ├── topk_sae.py                 # TopK and Batch-TopK SAE variants
│   │   │   ├── trainer.py                  # SAE training loop and optimization
│   │   │   └── metrics.py                  # Evaluation metrics (MSE, R2, etc.)
│   │   └── utils/                          
│   │       ├── hooks.py                    # Hook management for interventions
│   │       ├── activations_iterator.py     # Activation loaders with buffer for SAE training
│   │       ├── analysis_utils.py           # Utilities for analysis and plotting
│   │       └── reproducibility.py          # Seed and determinism utilities
│   ├── scripts/                            
│   │   ├── collect_latents.py              # Extract and cache model activations
│   │   ├── train_sae.py                    # Train SAEs on cached activations
│   │   ├── prepare_prompt_datasets.py          # Dataset preparation utilities
│   │   ├── collect_top_dataset_examples.py     # Extract and visualize top examples for each SAE feature
│   │   └── prepare_circuit_prompt_dataset.py   # Create prompt dataset for sparse feature circuit discovery
│   ├── experiments/                       
│   │   ├── activation_patching.py          # Activation patching between prompt runs
│   │   ├── evaluate_sae.py                 # SAE reconstruction metrics
│   │   ├── intervene_sae.py                # Various feature interventions
│   │   ├── visualize_sae_spatial.py        # Spatial feature visualizations
│   │   ├── ablate_sae.py                   # Ablation study framework for SAEs
│   │   ├── analyze_feature_activity.py     # Analyze SAE feature activity over time
│   │   └── visualize_sae_ablation_results.py # Visualize SAE ablation results
│   ├── circuit_discovery/                  
│   │   ├── discover_circuits.py            # Main circuit discovery pipeline
│   │   ├── train_probe.py                  # Train concept probes that serves as metric
│   │   ├── temporal_attribution.py         # Attribution patching and edge weight calculation
│   │   ├── circuit_utils.py                # Utility functions for attribution patching
│   │   ├── activation_utils.py             # Helper class for sparse feature activations
│   │   ├── circuit_plotting.py             # Plotting functions for causal circuits
│   │   ├── coo_utils.py                    # Utilities for handling sparse COO tensors
│   │   ├── plot_saved_circuit.py           # Load and plot a pre-computed circuit
│   │   ├── probe.py                        # Defines a CNN probe for latent representations
│   │   └── circuit_discovery_datasets.py   # Create comparative and single-object prompt datasets
│   └── ablation_configs/                   # Hydra configs for hyperparameter sweeps
├── scripts/                                
│   ├── train_sae.sh                        # SAE training with default settings
│   ├── discover_circuits.sh                # Main circuit discovery pipeline
│   ├── activation_patching.sh              # Activation patching experiments
│   ├── collect_latents.sh                  # Collect latent representations for SAE training
│   ├── collect_latents_for_probe.sh        # Collect latents for probe training
│   ├── collect_top_examples.sh             # Collect top activating examples for SAE features
│   ├── evaluate_sae.sh                     # Evaluate SAE performance
│   ├── intervene_sae.sh                    # Run SAE intervention experiments
│   └── run_ablations.sh                    # Run SAE ablation studies
├── notebooks/                             
│   ├── diffusers_model_comparison.ipynb    # Compares image synthesis of different diffusion models
│   └── sae_evaluation.ipynb                # Creates tables summarizing the results from `scripts/evaluate_sae.sh`
```


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
