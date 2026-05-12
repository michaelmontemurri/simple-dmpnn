# Directed Message Passing Neural Network: Implemented From Scratch

This repository provides a straightforward PyTorch implementation of the Directed Message Passing Neural Network (D-MPNN) introduced by [Yang et al. (2019)](https://arxiv.org/abs/1904.01561). The goal is to make the architecture easy to understand, inspect, and adapt outside the full Chemprop ecosystem.


<p align="center">
  <img src="figs/dmpnn_schematic.png" alt="DMPNN schematic" width="48%">
  <img src="figs/attribution_plot.png" alt="DMPNN attribution figure" width="48%">
</p>

<p align="center">
  <em>Left: D-MPNN schematic from Yang et al. (2019). Right: attribution plot from this repository's synthetic example.</em>
</p>


## Repository Outline

```text
dmpnn/
├── model.py
├── training.py
├── graph_utils.py
└── adapters.py

notebooks/
├── demo.ipynb
│   └── Annotated walkthrough of the D-MPNN theory and implementation
└── testing.ipynb
    └── Tests using PyG graph objects, plus minimal benchmarking against GINEConv

demo_train_script.py
└── Example training script using simulated graphs

demo_inference_script.py
└── Example inference script using a trained model
```

## Requirements

The notebooks and scripts are designed to run on:

- Apple Silicon with MPS
- CPU
- NVIDIA GPU with CUDA

Install dependencies with:

```bash
pip install -r requirements.txt
```

## How to Use

The reusable implementation is contained in `dmpnn/`.

For examples, see:

- `demo_train_script.py` for training a D-MPNN on simulated graphs
- `demo_inference_script.py` for running inference with a trained model
- `notebooks/demo.ipynb` for an annotated explanation of the architecture
- `notebooks/testing.ipynb` for testing with PyG graph objects and comparison to `GINEConv`

## Citation

This implementation is based on the Directed Message Passing Neural Network architecture introduced in:

Yang, K., Swanson, K., Jin, W., Coley, C., Eiden, P., Gao, H., Guzman-Perez, A., Hopper, T., Kelley, B., Mathea, M., Palmer, A., Settels, V., Jaakkola, T., Jensen, K., and Barzilay, R.  
“Analyzing Learned Molecular Representations for Property Prediction.”  
*Journal of Chemical Information and Modeling* 59, no. 8 (2019): 3370–3388.
