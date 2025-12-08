# LIBERO-10 Offline RRD Setup

This document describes how to use Randomized Reward Decomposition (RRD) for offline training on the LIBERO-10 dataset.

## Overview

The implementation supports offline reinforcement learning on LIBERO-10 demonstrations stored in RLDS format. The system:
- Loads episodes from RLDS dataset
- Uses workspace camera images only (not wrist camera)
- Trains RRD reward network to predict episodic returns from random subsequences
- Supports continuous actions (7-DoF robot control)

## Dataset Format

The dataset should be in RLDS format at:
```
/data/kmirakho/JustImagine/modified_libero_rlds/libero_10_no_noops/1.0.0
```

Each episode contains:
- `observation.image`: Workspace camera RGB image (256x256x3, uint8)
- `observation.wrist_image`: Wrist camera RGB image (not used)
- `observation.state`: Robot EEF state (8D float32)
- `action`: Robot action (7D float32)
- `reward`: Per-step reward (float32)
- `is_terminal`: Terminal flag (bool)
- `is_last`: Last step flag (bool)

## Usage

### Basic Training Command

```bash
python scripts/train.py \
    --env libero-10 \
    --env_type normal \
    --alg rrd \
    --basis_alg sac \
    --libero_dataset_path /data/kmirakho/JustImagine/modified_libero_rlds/libero_10_no_noops/1.0.0 \
    --libero_image_size 84 \
    --epochs 10 \
    --cycles 50 \
    --iterations 100 \
    --train_batches 50 \
    --batch_size 32 \
    --rrd_batch_size 32 \
    --rrd_sample_size 32
```

### Key Arguments

- `--env libero-10`: Use LIBERO-10 environment
- `--libero_dataset_path`: Path to RLDS dataset directory
- `--libero_image_size`: Image size for observations (default: 84, original: 256)
- `--alg rrd`: Use Randomized Reward Decomposition
- `--basis_alg sac`: Use SAC as the base algorithm

### Image Processing

- Images are resized from 256x256 to the specified size (default 84x84)
- Images are normalized to [0, 1] range
- Only workspace camera images are used (wrist camera is ignored)

### Offline Training

The implementation uses an offline buffer that:
- Loads all episodes from the dataset at initialization
- Samples random subsequences for RRD training
- Does not interact with the environment during training

### Reward Network Architecture

For LIBERO (continuous actions + images), the system uses `ConvRewardNetContinuous`:
- Convolutional layers to process image observations
- Concatenates image features with continuous actions
- Outputs scalar reward prediction

## Dependencies

Required packages:
- `tensorflow` (for RLDS dataset loading)
- `tensorflow_datasets` (for RLDS format support)
- `opencv-python` (for image processing)
- `numpy`
- `torch` (PyTorch backend)

## Notes

- The dataset is loaded entirely into memory at initialization
- Episodic rewards are computed as average reward per step (for numerical stability)
- The learner skips environment interaction and trains only from the buffer
- Checkpointing is supported via `--checkpoint_dir`, `--save_freq`, and `--load_path`

