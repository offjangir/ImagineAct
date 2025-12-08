#!/bin/bash
# Quick dependency installer for PyTorch backend
# Run this in your active conda environment (openpi or WorldModelEval)

echo "Installing PyTorch backend dependencies..."
echo ""

# Install core dependencies
pip install beautifultable termcolor

# Install PyTorch and TensorBoard
pip install torch torchvision tensorboard

# These should already be installed, but just in case:
pip install opencv-python gym numpy

echo ""
echo "✓ Installation complete!"
echo ""
echo "Now run with PyTorch backend (from repo root):"
echo "  export USE_PYTORCH=1"
echo "  python scripts/train.py --tag='RRD Ant-v2' --alg=rrd --basis_alg=sac --env=Ant-v2"






