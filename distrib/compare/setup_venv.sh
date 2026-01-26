#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"
python3 -m venv venv
source venv/bin/activate
pip install numpy scipy

pip install torch

# Install PyTorch with CUDA 12.1 support (for NVIDIA GPUs)
# Use --index-url https://download.pytorch.org/whl/cu118 for CUDA 11.8
# Use --index-url https://download.pytorch.org/whl/cpu for CPU-only

#pip install torch --index-url https://download.pytorch.org/whl/cu121
