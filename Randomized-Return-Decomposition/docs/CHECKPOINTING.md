# Checkpointing Guide

This document explains how to use checkpoint saving and loading in the Randomized-Return-Decomposition repository.

## Overview

The repository now supports automatic checkpoint saving during training and checkpoint loading for resuming training or evaluation. Checkpoints save all neural network weights and the observation normalizer state.

## Command-Line Arguments

### Checkpoint Saving Arguments

- `--checkpoint_dir`: Directory to save checkpoints (default: `log/checkpoints`)
- `--save_freq`: Save checkpoint every N epochs (default: `1`, set to `0` to disable periodic saves)
- `--save_final`: Whether to save final checkpoint at end of training (default: `True`)

### Checkpoint Loading Arguments

- `--load_path`: Path to checkpoint file to load (for resuming training or evaluation)

## Usage Examples

### 1. Training with Automatic Checkpointing

Save checkpoints every epoch (default behavior):
```bash
python scripts/train.py --tag='MyExperiment' --alg=rrd --basis_alg=sac --env=Ant-v2
```

This will save checkpoints to: `log/checkpoints/MyExperiment/checkpoint_epoch_1.pt`, `checkpoint_epoch_2.pt`, etc., and `checkpoint_final.pt` at the end.

### 2. Custom Checkpoint Frequency

Save checkpoints every 5 epochs:
```bash
python scripts/train.py --tag='MyExperiment' --save_freq=5 --alg=rrd --basis_alg=sac --env=Ant-v2
```

### 3. Disable Periodic Saves, Only Save Final

Only save the final checkpoint:
```bash
python scripts/train.py --tag='MyExperiment' --save_freq=0 --save_final=True --alg=rrd --basis_alg=sac --env=Ant-v2
```

### 4. Resume Training from Checkpoint

Load a checkpoint and continue training:
```bash
python scripts/train.py --tag='MyExperiment' --load_path='log/checkpoints/MyExperiment/checkpoint_epoch_3.pt' --alg=rrd --basis_alg=sac --env=Ant-v2
```

### 5. Evaluate a Trained Model

Load a checkpoint for evaluation:
```bash
python scripts/test.py --load_path='log/checkpoints/MyExperiment/checkpoint_final.pt' --env=Ant-v2
```

### 6. Custom Checkpoint Directory

Save checkpoints to a custom directory:
```bash
python scripts/train.py --tag='MyExperiment' --checkpoint_dir='my_checkpoints' --alg=rrd --basis_alg=sac --env=Ant-v2
```

## Checkpoint File Format

### PyTorch Backend (`.pt` files)
- Contains all neural network state dictionaries (policy, Q-networks, target networks, reward networks)
- Contains observation normalizer state
- Saved using `torch.save()`

### TensorFlow Backend (`.ckpt` files)
- Contains TensorFlow session variables
- Saved using `tf.train.Saver()`

## What Gets Saved

The checkpoint includes:
1. **Policy Network**: Actor/policy network weights
2. **Q-Value Networks**: Critic networks (and target networks)
3. **Reward Network**: (For RRD algorithm) Reward decomposition network
4. **Observation Normalizer**: Normalization statistics (if enabled)

## Implementation Details

### Saving Checkpoints

Checkpoints are saved:
- **Periodically**: After every N epochs (controlled by `--save_freq`)
- **Finally**: At the end of training (if `--save_final=True`)

Checkpoint paths are organized as:
```
{checkpoint_dir}/{tag}/checkpoint_epoch_{N}.pt
{checkpoint_dir}/{tag}/checkpoint_final.pt
```

If no tag is provided, the directory name is `{alg}-{env}`.

### Loading Checkpoints

When `--load_path` is specified:
1. The checkpoint file is checked for existence
2. If found, `agent.load_model()` is called to load weights
3. If loading fails, training continues with random initialization (with a warning)
4. If file doesn't exist, a warning is logged and training continues

## Error Handling

- If checkpoint saving fails, a warning is logged but training continues
- If checkpoint loading fails, a warning is logged and training starts with random weights
- Checkpoint directory is created automatically if it doesn't exist

## Best Practices

1. **Use descriptive tags**: Use `--tag` to organize checkpoints by experiment
2. **Save frequently during long training**: Set `--save_freq=1` for important experiments
3. **Keep final checkpoint**: Always save final checkpoint (`--save_final=True`)
4. **Verify checkpoints**: Test loading checkpoints before deleting old ones
5. **Organize by experiment**: Use different tags for different hyperparameter configurations

## Troubleshooting

### Checkpoint Not Found
If you get "Checkpoint path specified but file not found":
- Verify the path is correct
- Check that the file extension matches your backend (`.pt` for PyTorch, `.ckpt` for TensorFlow)
- Ensure the checkpoint was saved successfully during training

### Loading Fails
If checkpoint loading fails:
- Check that you're using the same backend (PyTorch vs TensorFlow) as when saving
- Verify the checkpoint file is not corrupted
- Ensure the model architecture matches (same algorithm, same environment)

### Out of Disk Space
If you run out of disk space:
- Increase `--save_freq` to save less frequently
- Manually delete old checkpoints
- Use `--save_final=False` if you don't need the final checkpoint

## Example Workflow

```bash
# 1. Train a model with checkpointing
python scripts/train.py --tag='RRD-Ant-v2' --alg=rrd --basis_alg=sac --env=Ant-v2 --save_freq=1

# 2. Check what checkpoints were saved
ls log/checkpoints/RRD-Ant-v2/

# 3. Evaluate the final checkpoint
python scripts/test.py --load_path='log/checkpoints/RRD-Ant-v2/checkpoint_final.pt' --env=Ant-v2

# 4. Resume training from epoch 2
python scripts/train.py --tag='RRD-Ant-v2' --load_path='log/checkpoints/RRD-Ant-v2/checkpoint_epoch_2.pt' --alg=rrd --basis_alg=sac --env=Ant-v2
```

