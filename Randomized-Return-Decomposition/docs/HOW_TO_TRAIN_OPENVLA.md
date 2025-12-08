# How to Train Reward Models with OpenVLA Features

This guide shows you how to train reward models using OpenVLA language and vision features.

## Prerequisites

1. **OpenVLA Checkpoint**: You need a trained OpenVLA checkpoint
   - Example: `/path/to/openvla/openvla-7b-finetuned-libero-goal`
   - Should contain: `config.json`, `model files`, `dataset_statistics.json`

2. **LIBERO Dataset**: RLDS format dataset
   - Example: `/data/kmirakho/JustImagine/modified_libero_rlds/libero_10_no_noops/1.0.0`

3. **Dependencies**: 
   - PyTorch (required for OpenVLA)
   - Transformers library
   - Flash Attention 2 (optional but recommended)

## Quick Start

### Option 1: Using the Training Script (Recommended)

```bash
cd Randomized-Return-Decomposition

python scripts/train_with_openvla.py \
    --env libero-10 \
    --env_type normal \
    --alg rrd \
    --basis_alg sac \
    --use_openvla_features True \
    --openvla_checkpoint /path/to/openvla/openvla-7b-finetuned-libero-goal \
    --openvla_task_description "pick up the red cup" \
    --libero_dataset_path /data/kmirakho/JustImagine/modified_libero_rlds/libero_10_no_noops/1.0.0 \
    --libero_image_size 256 \
    --cuda \
    --device cuda:2 \
    --epochs 10 \
    --cycles 50 \
    --iterations 100 \
    --train_batches 50 \
    --batch_size 32 \
    --rrd_batch_size 32 \
    --rrd_sample_size 32 \
    --rrd_reward_only True \
    --r_lr 3e-4 \
    --tag openvla_reward_libero10 \
    --save_freq 2 \
    --checkpoint_dir log/checkpoints \
    --save_final True
```

### Option 2: Modify Existing Training Script

If you prefer to use the standard `train.py`, you can load OpenVLA before calling `experiment_setup`:

```python
# In scripts/train.py or a custom script
import sys
sys.path.append('/path/to/openvla')

from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# ... load OpenVLA model (see train_with_openvla.py) ...

args = get_args()
args.use_openvla_features = True
args.openvla_model = vla_model
args.openvla_processor = processor
args.task_description = "pick up the red cup"

env, agent, buffer, learner, tester = experiment_setup(args)
# ... rest of training ...
```

## Key Arguments

### OpenVLA-Specific Arguments

- `--use_openvla_features True`: Enable OpenVLA features
- `--openvla_checkpoint <PATH>`: Path to OpenVLA checkpoint directory
- `--openvla_task_description <TEXT>`: Task description for LIBERO (e.g., "pick up the red cup")
- `--openvla_reward_hidden_dim <INT>`: Hidden dimension for reward network (default: 512)
- `--openvla_reward_num_layers <INT>`: Number of MLP layers (default: 3)

### Standard RRD Arguments

- `--rrd_reward_only True`: Only train reward network (recommended for initial training)
- `--r_lr <FLOAT>`: Learning rate for reward network (default: 3e-4)
- `--rrd_batch_size <INT>`: Batch size for reward training (default: 32 for LIBERO)
- `--rrd_sample_size <INT>`: Sample size for random subsequences (default: 32)

### LIBERO Arguments

- `--libero_dataset_path <PATH>`: Path to RLDS dataset
- `--libero_image_size <INT>`: Image size (256 for full resolution, 84 for smaller)

## Training Modes

### 1. Reward-Only Training (Supervised Learning)

Train only the reward network on episodic returns:

```bash
python scripts/train_with_openvla.py \
    --use_openvla_features True \
    --openvla_checkpoint <PATH> \
    --rrd_reward_only True \
    # ... other args ...
```

### 2. Full RRD Training

Train both reward network and policy:

```bash
python scripts/train_with_openvla.py \
    --use_openvla_features True \
    --openvla_checkpoint <PATH> \
    --rrd_reward_only False \
    # ... other args ...
```

## Example: Training on LIBERO-10

```bash
python scripts/train_with_openvla.py \
    --env libero-10 \
    --env_type normal \
    --alg rrd \
    --basis_alg sac \
    --use_openvla_features True \
    --openvla_checkpoint /data/kmirakho/JustImagine/openvla/openvla-7b-finetuned-libero-goal \
    --openvla_task_description "pick up the red cup" \
    --libero_dataset_path /data/kmirakho/JustImagine/modified_libero_rlds/libero_10_no_noops/1.0.0 \
    --libero_image_size 256 \
    --cuda \
    --device cuda:2 \
    --epochs 10 \
    --cycles 50 \
    --iterations 100 \
    --train_batches 50 \
    --batch_size 32 \
    --rrd_batch_size 32 \
    --rrd_sample_size 32 \
    --rrd_reward_only True \
    --r_lr 3e-4 \
    --tag openvla_reward_libero10 \
    --save_freq 2 \
    --checkpoint_dir log/checkpoints
```

## What Happens During Training

1. **Feature Extraction**: For each batch of observations:
   - Images are processed through OpenVLA's vision backbone → vision patch features
   - Task descriptions are tokenized and embedded → language features

2. **Reward Prediction**: 
   - Language and vision features are pooled and concatenated with actions
   - Reward network predicts per-step rewards
   - Predictions are averaged/scaled to match episodic returns

3. **Loss Computation**:
   - MSE loss between predicted and actual episodic returns
   - Optional variance penalty for bias correction

## Monitoring Training

- **R_loss**: Reward network loss (should decrease over time)
- **R_var**: Variance of predictions (if bias correction enabled)
- Checkpoints are saved to `log/checkpoints/<tag>/`

## Tips

1. **Start with reward-only training** (`--rrd_reward_only True`) to validate the setup
2. **Use smaller batch sizes** initially to test (e.g., `--rrd_batch_size 16`)
3. **Monitor GPU memory** - OpenVLA feature extraction can be memory-intensive
4. **Adjust learning rate** if loss doesn't decrease (`--r_lr`)
5. **Use appropriate image size** - 256x256 for full features, 84x84 for faster training

## Troubleshooting

### Out of Memory
- Reduce `--batch_size` and `--rrd_batch_size`
- Use `--libero_image_size 84` instead of 256
- Use a smaller GPU or CPU mode

### Slow Training
- Feature extraction happens on-the-fly
- Consider pre-extracting and caching features for faster training
- Reduce `--rrd_sample_size`

### Import Errors
- Make sure OpenVLA path is in `sys.path`
- Check that all dependencies are installed
- Verify checkpoint path is correct

