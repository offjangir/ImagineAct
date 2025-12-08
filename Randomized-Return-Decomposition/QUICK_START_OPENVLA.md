# Quick Start: Training with OpenVLA Features

## One-Command Training

```bash
cd Randomized-Return-Decomposition

python scripts/train_with_openvla.py \
    --env libero-10 \
    --env_type normal \
    --alg rrd \
    --basis_alg sac \
    --use_openvla_features True \
    --openvla_checkpoint /path/to/openvla/checkpoint \
    --openvla_task_description "pick up the red cup" \
    --libero_dataset_path /data/kmirakho/JustImagine/modified_libero_rlds/libero_10_no_noops/1.0.0 \
    --libero_image_size 256 \
    --cuda \
    --device cuda:2 \
    --rrd_reward_only True \
    --tag openvla_reward_training
```

## What You Need

1. **OpenVLA Checkpoint**: Path to your OpenVLA model directory
2. **LIBERO Dataset**: RLDS format dataset path
3. **Task Description**: Text description of the task (for LIBERO)

## Key Arguments Explained

| Argument | Description | Example |
|----------|-------------|---------|
| `--use_openvla_features True` | Enable OpenVLA features | Required |
| `--openvla_checkpoint <PATH>` | Path to OpenVLA checkpoint | `/path/to/openvla-7b` |
| `--openvla_task_description <TEXT>` | Task description | `"pick up the red cup"` |
| `--device cuda:2` | GPU device to use | `cuda:0`, `cuda:1`, etc. |
| `--rrd_reward_only True` | Train only reward network | Recommended for initial training |

## Training Output

- **R_loss**: Reward network loss (should decrease)
- **Checkpoints**: Saved to `log/checkpoints/<tag>/`
- **Logs**: Training progress and metrics

## Full Documentation

See `docs/HOW_TO_TRAIN_OPENVLA.md` for detailed instructions.

