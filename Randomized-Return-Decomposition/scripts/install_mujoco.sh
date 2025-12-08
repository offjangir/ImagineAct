#!/bin/bash
# MuJoCo installation script for Linux
# This installs MuJoCo 2.1.0 (free version, no license needed)

echo "Installing MuJoCo for Linux..."
echo ""

# 1. Install system dependencies
echo "Step 1: Installing system dependencies..."
echo "Note: This may require sudo password"

# Common dependencies for MuJoCo
sudo apt-get update
sudo apt-get install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    libglfw3 \
    libglfw3-dev

# 2. Download and install MuJoCo
echo ""
echo "Step 2: Downloading MuJoCo 2.1.0..."

MUJOCO_DIR="$HOME/.mujoco"
mkdir -p "$MUJOCO_DIR"
cd "$MUJOCO_DIR"

# Download MuJoCo 2.1.0 (free version)
if [ ! -d "$MUJOCO_DIR/mujoco210" ]; then
    wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz
    tar -xzf mujoco210.tar.gz
    rm mujoco210.tar.gz
    echo "✓ MuJoCo downloaded and extracted"
else
    echo "✓ MuJoCo already installed at $MUJOCO_DIR/mujoco210"
fi

# 3. Set environment variables
echo ""
echo "Step 3: Setting environment variables..."

# Add to current session
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# Add to bashrc for persistence
if ! grep -q "MUJOCO" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# MuJoCo paths" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$HOME/.mujoco/mujoco210/bin" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/lib/nvidia" >> ~/.bashrc
    echo "✓ Added MuJoCo paths to ~/.bashrc"
else
    echo "✓ MuJoCo paths already in ~/.bashrc"
fi

# 4. Install mujoco-py
echo ""
echo "Step 4: Installing mujoco-py Python package..."
# Change to script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

pip install mujoco-py

echo ""
echo "=========================================="
echo "MuJoCo Installation Complete!"
echo "=========================================="
echo ""
echo "IMPORTANT: Run this command to reload environment variables:"
echo "  source ~/.bashrc"
echo ""
echo "Or restart your terminal/shell."
echo ""
echo "Then test with:"
echo "  python -c 'import mujoco_py; print(\"MuJoCo works!\")'"
echo ""
echo "If successful, run your training (from repo root):"
echo "  export USE_PYTORCH=1"
echo "  python scripts/train.py --tag='RRD Ant-v2' --alg=rrd --basis_alg=sac --env=Ant-v2"






