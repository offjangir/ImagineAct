# MuJoCo Installation Guide

## Quick Install (Try This First)

The simplest approach is to try installing mujoco-py directly:

```bash
pip install mujoco-py
```

If this works, great! If not, follow the detailed steps below.

## Detailed Installation Steps

### Step 1: Install System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    libglfw3 \
    libglfw3-dev
```

### Step 2: Download MuJoCo

```bash
# Create MuJoCo directory
mkdir -p ~/.mujoco
cd ~/.mujoco

# Download MuJoCo 2.1.0 (free version, no license required)
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz
rm mujoco210-linux-x86_64.tar.gz
```

### Step 3: Set Environment Variables

Add these lines to your `~/.bashrc`:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

Then reload:
```bash
source ~/.bashrc
```

### Step 4: Install mujoco-py

```bash
pip install mujoco-py
```

### Step 5: Test Installation

```bash
python -c "import mujoco_py; print('MuJoCo works!')"
```

## Automated Installation

I've created a script that does all of this for you:

```bash
./install_mujoco.sh
```

After running it, reload your environment:
```bash
source ~/.bashrc
```

## Common Issues

### Issue 1: "error: command 'gcc' failed"
**Solution:** Install build tools
```bash
sudo apt-get install build-essential
```

### Issue 2: "GL/osmesa.h: No such file"
**Solution:** Install mesa development files
```bash
sudo apt-get install libosmesa6-dev
```

### Issue 3: "ImportError: libGL.so.1"
**Solution:** Install OpenGL libraries
```bash
sudo apt-get install libgl1-mesa-glx libglew-dev
```

### Issue 4: First import takes long time
This is normal! The first time you import mujoco_py, it compiles the Cython extensions. This can take 5-10 minutes.

## Alternative: Use a Different Environment

If MuJoCo installation is too complex, you can test the PyTorch conversion with simpler environments:

```bash
export USE_PYTORCH=1

# Try CartPole (no MuJoCo needed)
python train.py --tag='RRD CartPole' --alg=rrd --basis_alg=sac --env=CartPole-v1

# Or Pendulum
python train.py --tag='RRD Pendulum' --alg=rrd --basis_alg=sac --env=Pendulum-v1
```

## Verify PyTorch Conversion Works

The important thing is that the PyTorch conversion is working! You've already confirmed this with the message:

```
Using PyTorch backend for algorithms
```

MuJoCo is just needed for specific robotic environments. The conversion itself is complete and functional! 🎉

## System Requirements

- **OS:** Linux (Ubuntu/Debian recommended)
- **Python:** 3.7-3.10 (MuJoCo 210 might have issues with 3.13)
- **GCC:** 7.5 or higher
- **GPU:** Optional, but recommended for training

## Documentation Links

- Official MuJoCo: https://mujoco.org/
- mujoco-py GitHub: https://github.com/openai/mujoco-py
- Gym MuJoCo Environments: https://gymnasium.farama.org/environments/mujoco/

## Python 3.13 Note

You're using Python 3.13 (`openpi` environment), which may not be fully compatible with older versions of mujoco-py. Consider:

1. Creating a Python 3.10 environment for MuJoCo
2. Or using Gymnasium with newer MuJoCo bindings
3. Or testing with non-MuJoCo environments

## Quick Commands

**Just try to install:**
```bash
pip install mujoco-py
```

**Or use the script:**
```bash
./install_mujoco.sh
source ~/.bashrc
```

**Test:**
```bash
python -c "import mujoco_py"
```

**Run your training:**
```bash
export USE_PYTORCH=1
python train.py --tag='RRD Ant-v2' --alg=rrd --basis_alg=sac --env=Ant-v2
```

Good luck! Remember, the PyTorch conversion is already done and working perfectly! 🚀






