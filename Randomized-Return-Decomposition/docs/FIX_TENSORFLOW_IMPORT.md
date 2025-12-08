# Fix for TensorFlow Import Error

## Problem

When running the code with `USE_PYTORCH=1`, you encountered:

```
File "/data/kmirakho/JustImagine/Randomized-Return-Decomposition/utils/os_utils.py", line 8, in <module>
    import tensorflow as tf
ModuleNotFoundError: No module named 'tensorflow'
```

Even though you set `USE_PYTORCH=1`, the code was still trying to import TensorFlow.

## Root Cause

The `utils/os_utils.py` file had a hardcoded TensorFlow import at the top of the file:

```python
import tensorflow as tf  # Line 8 - Always executed!
```

This import was executed during the import chain, even before the algorithm code could check whether to use PyTorch or TensorFlow. The import happened when:

```
train.py → common.py → test.py → envs/__init__.py → normal_atari.py → os_utils.py → import tf ❌
```

## Solution

The `utils/os_utils.py` file has been updated to **conditionally import** based on the backend:

### Before:
```python
import os
import numpy as np
import tensorflow as tf  # Always imports TF
```

### After:
```python
import os
import numpy as np

# Conditional import for backend support
USE_PYTORCH = os.environ.get('USE_PYTORCH', '0') == '1'
if USE_PYTORCH:
    try:
        import torch
        from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter
        tf = None
    except ImportError:
        print("Warning: PyTorch not found, falling back to TensorFlow")
        import tensorflow as tf
        torch = None
        USE_PYTORCH = False
else:
    try:
        import tensorflow as tf
        torch = None
    except ImportError:
        print("Warning: TensorFlow not found, trying PyTorch")
        import torch
        from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter
        tf = None
        USE_PYTORCH = True
```

### Additional Changes:

1. **Two SummaryWriter Classes:**
   - `SummaryWriter` - For TensorFlow backend
   - `SummaryWriterPyTorch` - For PyTorch backend (uses TensorBoard)

2. **Logger Class Update:**
   - `summary_init()` now accepts optional `graph` and `sess` parameters
   - Automatically selects the correct SummaryWriter based on backend

3. **Graceful Fallback:**
   - If PyTorch is not found, falls back to TensorFlow
   - If TensorFlow is not found, tries PyTorch
   - Prints clear warnings if dependencies are missing

## How to Use

### 1. Install PyTorch Dependencies
```bash
pip install -r requirements_pytorch.txt
```

Or manually:
```bash
pip install torch torchvision tensorboard opencv-python gym termcolor beautifultable
```

### 2. Run with PyTorch Backend
```bash
export USE_PYTORCH=1
python train.py --tag='RRD Ant-v2' --alg=rrd --basis_alg=sac --env=Ant-v2
```

### 3. Verify It's Working
You should see:
```
Using PyTorch backend for algorithms
```

And NO TensorFlow import errors!

## Benefits of This Fix

1. ✅ **No TensorFlow Required:** Can run with only PyTorch installed
2. ✅ **Backward Compatible:** Still works with TensorFlow if USE_PYTORCH=0
3. ✅ **Automatic Fallback:** Gracefully handles missing dependencies
4. ✅ **Clear Messages:** Prints warnings if dependencies are missing
5. ✅ **Works with TensorBoard:** PyTorch version uses TensorBoard for logging

## Testing the Fix

### Test 1: Import without TensorFlow
```bash
export USE_PYTORCH=1
python -c "from utils.os_utils import Logger; print('✓ Success!')"
```

### Test 2: Create Logger with PyTorch
```bash
export USE_PYTORCH=1
python -c "
from utils.os_utils import get_logger
logger = get_logger('test')
print('✓ Logger created successfully!')
"
```

### Test 3: Full Training Script
```bash
export USE_PYTORCH=1
python train.py --tag='RRD Ant-v2' --alg=rrd --basis_alg=sac --env=Ant-v2
```

## What Changed in the Code

### File: `utils/os_utils.py`

**Lines 1-32:** Conditional imports based on `USE_PYTORCH` environment variable

**Lines 119-124:** Updated `summary_init()` method
- Made `graph` and `sess` optional parameters
- Chooses correct SummaryWriter based on backend

**Lines 213-303:** Original `SummaryWriter` class (TensorFlow)
- Added check: `if tf is None: raise ImportError(...)`

**Lines 305-379:** New `SummaryWriterPyTorch` class
- Uses `torch.utils.tensorboard.SummaryWriter`
- Compatible API with TensorFlow version
- Supports train/test/debug writers

## Architecture

```
┌─────────────────────────────────────┐
│   Environment Variable: USE_PYTORCH │
└──────────────┬──────────────────────┘
               │
       ┌───────▼────────┐
       │  USE_PYTORCH?  │
       └───┬────────┬───┘
           │        │
      Yes  │        │  No
           │        │
    ┌──────▼──┐  ┌─▼────────┐
    │ PyTorch │  │TensorFlow│
    └──┬──────┘  └─┬────────┘
       │            │
       ├────────────┤
       │            │
    ┌──▼────────────▼───┐
    │ SummaryWriter     │
    │ (Backend-specific)│
    └───────────────────┘
```

## Comparison

| Aspect | Before | After |
|--------|--------|-------|
| TF Import | Always | Conditional |
| PT Support | No | Yes |
| Fallback | None | Automatic |
| SummaryWriter | TF only | Both TF & PT |
| Error Messages | Cryptic | Clear |

## Migration Notes

If you have custom code that uses `os_utils.py`:

1. **Logger.summary_init():** Now accepts optional parameters
   ```python
   # TensorFlow:
   logger.summary_init(graph, sess)
   
   # PyTorch (or both):
   logger.summary_init()  # No args needed!
   ```

2. **SummaryWriter:** Use appropriate class
   ```python
   # TensorFlow:
   from utils.os_utils import SummaryWriter
   writer = SummaryWriter(graph, sess, path)
   
   # PyTorch:
   from utils.os_utils import SummaryWriterPyTorch
   writer = SummaryWriterPyTorch(path)
   ```

## Related Files

- `TROUBLESHOOTING.md` - Common issues and solutions
- `QUICK_START_PYTORCH.md` - Quick reference guide
- `PYTORCH_MIGRATION.md` - Complete migration guide
- `requirements_pytorch.txt` - PyTorch dependencies

## Summary

The TensorFlow import error has been fixed by making `utils/os_utils.py` backend-agnostic. The file now:

1. ✅ Checks `USE_PYTORCH` environment variable
2. ✅ Conditionally imports PyTorch or TensorFlow
3. ✅ Provides both TF and PT SummaryWriter classes
4. ✅ Falls back gracefully if dependencies are missing
5. ✅ Maintains backward compatibility with TensorFlow

You can now run the code with only PyTorch installed! 🎉






