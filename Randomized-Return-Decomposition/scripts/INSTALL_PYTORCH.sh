#!/bin/bash
# Installation script for PyTorch dependencies
# Run this script after activating your conda environment

echo "Installing PyTorch and dependencies..."
echo "Make sure you have activated your conda environment first!"
echo "  conda activate WorldModelEval  (or openpi, or your env name)"
echo ""

# Get script directory and repo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Install all dependencies from requirements file
echo "Installing from requirements/requirements_pytorch.txt..."
pip install -r "$REPO_ROOT/requirements/requirements_pytorch.txt"

# Alternative: Install individually
# pip install opencv-python gym tensorboard termcolor beautifultable
# pip install torch torchvision

# For CUDA support (recommended for GPU training), use:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "Installation complete!"
echo ""
echo "To use PyTorch backend, run from repo root:"
echo "  export USE_PYTORCH=1"
echo "  python scripts/train.py --tag='RRD Ant-v2' --alg=rrd --basis_alg=sac --env=Ant-v2"
echo ""
echo "Or use the example script:"
echo "  python scripts/train_pytorch_example.py --tag='RRD Ant-v2' --alg=rrd --basis_alg=sac --env=Ant-v2"

