#!/usr/bin/env bash

# Input env name, if empty, use default acorn-tutorial-seb.
name=acorn-tutorial-seb

for arg in "$@"; do
    case "$arg" in
        *)
            name=$arg
            ;;
    esac
done

# This script installs a CPU-only environment (no CUDA/GPU support).

conda create --yes --name "$name" python=3.10
conda activate "$name"

# CPU-only PyTorch build
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu

# Rest of the stack
pip install -r requirements.txt

# PyG CPU extension libraries (matched to torch 2.4.0 + CPU)
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cpu.html

pip install -e ./vendor/acorn