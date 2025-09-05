# Mechanistic Interpretability in Text-to-Image Diffusion Models

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/) [![Diffusers](https://img.shields.io/badge/🤗%20Diffusers-FF6F00?style=flat)](https://github.com/huggingface/diffusers) [![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A framework for understanding the internal mechanisms of text-to-image diffusion models through Sparse Autoencoders (SAEs) and automated circuit discovery. This work enables researchers to identify and analyze the specific model components responsible for generating particular visual concepts, providing interpretable insights into how diffusion models transform text into images over time.

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
