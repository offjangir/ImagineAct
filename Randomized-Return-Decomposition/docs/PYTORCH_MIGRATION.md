# PyTorch Migration Guide

This document explains the PyTorch conversion of the Randomized Return Decomposition (RRD) algorithm from TensorFlow to PyTorch.

## What Was Converted

The following files have been converted from TensorFlow to PyTorch:

### Core Files
1. **utils/torch_utils.py** - PyTorch utilities (converted from tf_utils.py)
   - `get_vars()` - Extract model parameters
   - `RandomNormal` - Normal distribution class
   - `Normalizer` - Online observation normalizer

2. **algorithm/basis_alg/base_torch.py** - Base class for all algorithms
   - Model creation and management
   - Normalizer handling
   - Save/load functionality

3. **algorithm/basis_alg/ddpg_torch.py** - DDPG algorithm in PyTorch
   - MLPPolicy and MLPQValue networks
   - Training loops for policy and Q-value

4. **algorithm/basis_alg/sac_torch.py** - SAC algorithm in PyTorch
   - Stochastic policy with temperature parameter
   - Double Q-learning
   - Automatic entropy tuning

5. **algorithm/rrd_torch.py** - RRD algorithm in PyTorch
   - Reward network (MLP and Conv versions)
   - Randomized return decomposition
   - Bias correction support

## How to Use PyTorch Version

### Method 1: Environment Variable (Recommended)

Set the environment variable before running:

```bash
export USE_PYTORCH=1
python train.py --tag='RRD Ant-v2' --alg=rrd --basis_alg=sac --env=Ant-v2
```

### Method 2: Modify Training Script

You can also modify the `args` to include a `use_pytorch` flag in your training script.

## Key Differences from TensorFlow

### 1. Session Management
- **TensorFlow**: Requires session management (`tf.Session()`)
- **PyTorch**: No session needed, direct execution

### 2. Placeholders vs Tensors
- **TensorFlow**: Uses placeholders and feed_dict
- **PyTorch**: Direct tensor operations

### 3. Network Definition
- **TensorFlow**: Variable scopes and lazy execution
- **PyTorch**: `nn.Module` classes with eager execution

### 4. Training
- **TensorFlow**: Separate graph definition and execution
- **PyTorch**: Define-by-run with automatic differentiation

## Architecture Changes

### Network Modules
All networks are now defined as `nn.Module` subclasses:
- `MLPPolicy` - Policy network
- `MLPQValue` - Q-value network (DDPG)
- `MLPQValueSAC` - Q-value network (SAC)
- `MLPStochasticPolicy` - Stochastic policy (SAC)
- `MLPRewardNet` - Reward network (MLP version)
- `ConvRewardNet` - Reward network (Conv version for images)

### Optimizer Changes
- **TensorFlow**: `tf.train.AdamOptimizer`
- **PyTorch**: `torch.optim.Adam`

### Target Networks
- **TensorFlow**: Explicit assign operations
- **PyTorch**: `load_state_dict()` and soft updates with parameter copying

## Requirements

Install PyTorch dependencies:

```bash
pip install torch torchvision
```

For CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Supported Algorithms

Currently supported in PyTorch:
- ✅ DDPG
- ✅ SAC
- ✅ RRD (with DDPG or SAC as basis)

Not yet converted (still require TensorFlow):
- ❌ TD3
- ❌ DQN
- ❌ IRCR

## Performance Notes

1. **Memory Usage**: PyTorch typically uses less memory than TensorFlow 1.x
2. **Speed**: Similar performance, but PyTorch may be slightly faster on modern GPUs
3. **Debugging**: PyTorch's eager execution makes debugging easier

## Example Usage

```python
import torch
from algorithm import create_agent

# Set up args (assuming you have an args object)
args.use_pytorch = True  # Or set USE_PYTORCH=1 environment variable
args.alg = 'rrd'
args.basis_alg = 'sac'
args.cuda = torch.cuda.is_available()

# Create agent
agent = create_agent(args)

# Training loop (simplified)
for epoch in range(num_epochs):
    # Collect experience
    batch = replay_buffer.sample(batch_size)
    
    # Update normalizer
    agent.normalizer_update(batch)
    
    # Train
    info = agent.train(batch)
    
    # Update target networks
    agent.target_update()
    
    # Print training info
    print(f"Epoch {epoch}: {info}")
```

## Troubleshooting

### Issue: "No module named 'cv2'"
This is unrelated to the PyTorch conversion. Install OpenCV:
```bash
pip install opencv-python
```

### Issue: "CUDA out of memory"
Reduce batch size or use CPU:
```python
args.cuda = False
```

### Issue: "Cannot import torch"
Install PyTorch:
```bash
pip install torch
```

## Migration Checklist

If you have custom code using the TensorFlow version:

- [ ] Replace `sess.run()` calls with direct function calls
- [ ] Remove `feed_dict` usage
- [ ] Convert placeholders to function arguments
- [ ] Update network definitions to use `nn.Module`
- [ ] Change optimizer initialization
- [ ] Update checkpoint loading/saving

## Future Work

Planned conversions:
- TD3 algorithm
- DQN algorithm
- IRCR algorithm
- Additional replay buffer optimizations
- Mixed precision training support

## References

- Original TensorFlow implementation: `algorithm/rrd.py`
- PyTorch implementation: `algorithm/rrd_torch.py`
- PyTorch documentation: https://pytorch.org/docs/

## Contact

For issues or questions about the PyTorch conversion, please open an issue on the repository.






