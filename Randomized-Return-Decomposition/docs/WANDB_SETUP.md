# WandB Setup for Reward Network Training

## Overview

WandB (Weights & Biases) integration is now available for logging reward network losses and training metrics.

## Quick Setup

### 1. Install WandB

```bash
pip install wandb
```

### 2. Login to WandB

```bash
wandb login
```

### 3. Configure in Training Script

Edit `scripts/train_openvla.sh`:

```bash
# WandB Configuration
USE_WANDB="True"
WANDB_PROJECT="rrd-openvla-libero"  # Your project name
WANDB_ENTITY=""                      # Your username (leave empty for default)
```

### 4. Run Training

```bash
./scripts/train_openvla.sh
```

## What Gets Logged

### Reward Network Metrics

- **`R_loss`**: Reward network loss (MSE between predicted and actual episodic returns)
  - Logged every cycle
  - This is the main metric for reward-only training

- **`R_var`**: Reward variance penalty (only if `--rrd_bias_correction True`)

### Training Progress

- **`Epoch`**: Current epoch
- **`Cycle`**: Current cycle
- **`Episodes`**: Total episodes processed
- **`Timesteps`**: Total timesteps processed
- **`TimeCost(sec)/train`**: Training time per cycle
- **`TimeCost(sec)/test`**: Testing time (0 for reward-only training)

## Viewing Results

1. **WandB Dashboard**: Visit https://wandb.ai
2. **Select your project**: `rrd-openvla-libero`
3. **View run**: Click on the latest run to see plots

## Key Metrics to Monitor

### R_loss Plot
- **X-axis**: Training cycles
- **Y-axis**: Reward network loss
- **Expected**: Should decrease over time as the reward network learns

### Example Plot Names in WandB:
- `R_loss` - Main reward loss
- `Epoch` - Training progress
- `TimeCost(sec)/train` - Training speed

## Configuration Options

### In `train_openvla.sh`:

```bash
USE_WANDB="True"                    # Enable/disable WandB
WANDB_PROJECT="your-project-name"  # Project name
WANDB_ENTITY="your-username"       # Optional: your WandB username
```

### Command Line Override:

```bash
python scripts/train_with_openvla.py \
    --use_wandb True \
    --wandb_project "my-project" \
    --wandb_entity "my-username" \
    ...
```

## Troubleshooting

### WandB not logging
- Check: `pip install wandb`
- Check: `wandb login` completed
- Check: `USE_WANDB="True"` in script

### Wrong project name
- Update `WANDB_PROJECT` in script
- Or use `--wandb_project` command line argument

### Want to disable WandB
- Set `USE_WANDB="False"` in script
- Or remove `--use_wandb` flag

## Reward-Only Training

With `RRD_REWARD_ONLY="True"`:
- Only `R_loss` is logged (no policy/Q losses)
- Testing is skipped
- Focus is on reward network training

This is perfect for supervised learning of reward functions!

