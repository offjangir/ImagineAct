# Troubleshooting Guide

## Common Issues and Solutions

### 1. "No module named 'tensorflow'" (when using PyTorch)

**Error:**
```
ModuleNotFoundError: No module named 'tensorflow'
```

**Solution:**
This was fixed! The `utils/os_utils.py` file now conditionally imports based on the backend. Make sure:
1. You've set `USE_PYTORCH=1` environment variable
2. PyTorch is installed: `pip install torch`
3. The file has been updated with the latest changes

### 2. "No module named 'cv2'"

**Error:**
```
ModuleNotFoundError: No module named 'cv2'
```

**Solution:**
```bash
pip install opencv-python
```

### 3. "No module named 'gym'"

**Error:**
```
ModuleNotFoundError: No module named 'gym'
```

**Solution:**
```bash
pip install gym
```

Note: You may see a warning about gym being unmaintained. For now, gym 0.26.2 still works, but consider upgrading to gymnasium in the future.

### 4. Missing Dependencies

**Error:**
Various import errors for `termcolor`, `beautifultable`, `tensorboard`, etc.

**Solution:**
Install all dependencies at once:
```bash
pip install -r requirements_pytorch.txt
```

Or individually:
```bash
pip install termcolor beautifultable tensorboard
```

### 5. CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
1. Reduce batch size in your training config
2. Use CPU instead: Set `args.cuda = False` or don't pass `--cuda` flag
3. Clear GPU cache: `torch.cuda.empty_cache()`

### 6. "RuntimeError: Expected all tensors to be on the same device"

**Error:**
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

**Solution:**
This is a code issue. Make sure all tensors are moved to the correct device. Check that:
- Observations are moved to device: `obs.to(self.device)`
- Actions are moved to device: `acts.to(self.device)`
- Model is on correct device: `model.to(self.device)`

### 7. Different Results from TensorFlow Version

**Issue:**
Results differ when switching from TensorFlow to PyTorch.

**Possible Causes:**
1. **Random Seeds:** Make sure to set the same random seed
   ```python
   import torch
   import numpy as np
   torch.manual_seed(seed)
   np.random.seed(seed)
   ```

2. **Initialization:** Verify network initialization is the same (Xavier/Glorot)

3. **Numerical Precision:** Minor differences are normal due to different backends

### 8. Import Error on Algorithm Load

**Error:**
```
ImportError: cannot import name 'DDPG' from 'algorithm.basis_alg.ddpg_torch'
```

**Solution:**
1. Make sure `USE_PYTORCH=1` is set before importing
2. Check that PyTorch files exist in the correct locations
3. Try reimporting or restarting Python:
   ```bash
   python -c "import os; os.environ['USE_PYTORCH']='1'; from algorithm import create_agent"
   ```

### 9. Conda Environment Issues

**Error:**
```
CondaError: Run 'conda init' before 'conda activate'
```

**Solution:**
Initialize conda first:
```bash
conda init bash
source ~/.bashrc
conda activate WorldModelEval
```

### 10. Training Not Using PyTorch

**Issue:**
Code still uses TensorFlow even though `USE_PYTORCH=1` is set.

**Checklist:**
1. ✅ Environment variable is set: `echo $USE_PYTORCH` should show `1`
2. ✅ Set before importing: The variable must be set BEFORE Python imports the modules
3. ✅ Correct shell: Make sure it's set in the same shell where you run Python
4. ✅ Check import messages: Should see "Using PyTorch backend for algorithms"

**Debug:**
```bash
export USE_PYTORCH=1
python -c "import os; print('USE_PYTORCH:', os.environ.get('USE_PYTORCH')); from algorithm import create_agent"
```

## Installation Steps (Complete)

For a fresh setup on a new environment:

```bash
# 1. Activate your conda environment
conda activate WorldModelEval  # or openpi, or your env name

# 2. Install dependencies
pip install -r requirements_pytorch.txt

# Or install individually:
# pip install torch torchvision tensorboard
# pip install opencv-python gym numpy
# pip install termcolor beautifultable

# 3. For CUDA/GPU support (optional, recommended):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Test installation
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## Verification

To verify everything is set up correctly:

```bash
# Set environment variable
export USE_PYTORCH=1

# Test import
python -c "
import os
os.environ['USE_PYTORCH'] = '1'
from algorithm import create_agent
print('✓ PyTorch backend loaded successfully!')
"

# Test with actual args (adjust as needed)
python -c "
import os
os.environ['USE_PYTORCH'] = '1'
import argparse
from algorithm import create_agent

class Args:
    use_pytorch = True
    alg = 'rrd'
    basis_alg = 'sac'
    cuda = False
    obs_normalization = True
    # Add other required args...

args = Args()
print('Creating agent...')
# agent = create_agent(args)  # Uncomment to test
print('✓ Success!')
"
```

## Getting Help

If you encounter an issue not listed here:

1. **Check the documentation:**
   - `QUICK_START_PYTORCH.md` - Quick reference
   - `PYTORCH_MIGRATION.md` - Detailed migration guide
   - `CONVERSION_SUMMARY.md` - Technical details

2. **Enable debug mode:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. **Check versions:**
   ```bash
   python --version
   pip list | grep torch
   pip list | grep numpy
   pip list | grep gym
   ```

4. **Clean install:**
   Sometimes a clean reinstall helps:
   ```bash
   pip uninstall torch torchvision -y
   pip cache purge
   pip install torch torchvision
   ```

## Known Limitations

1. **TD3 and DQN:** Not yet converted to PyTorch (still need TensorFlow)
2. **IRCR:** Not yet converted to PyTorch
3. **Atari Environments:** Conv networks are converted but not extensively tested
4. **Multi-GPU:** Not yet implemented for PyTorch version

## Performance Tips

1. **Use GPU:** Set `args.cuda = True` and ensure CUDA is available
2. **Batch Size:** Larger batches are more efficient on GPU
3. **Mixed Precision:** Consider using `torch.cuda.amp` for faster training
4. **DataLoader:** Use PyTorch's DataLoader for efficient data loading
5. **Pin Memory:** Use `pin_memory=True` when moving data to GPU

## Debug Mode

To enable verbose logging:

```python
import os
os.environ['USE_PYTORCH'] = '1'
os.environ['PYTORCH_DEBUG'] = '1'  # Enable PyTorch debug mode

import torch
torch.autograd.set_detect_anomaly(True)  # Detect NaN/Inf in gradients
```

## Contact

For persistent issues:
- Check the GitHub issues page
- Review the code in `algorithm/rrd_torch.py`
- Compare with TensorFlow version in `algorithm/rrd.py`






