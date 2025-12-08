#!/bin/bash
# Simple training script - minimal configuration
# Just modify the paths and run!

# Paths - MODIFY THESE
OPENVLA_CHECKPOINT="/data/kmirakho/JustImagine/openvla/openvla-7b-finetuned-libero-goal"
LIBERO_DATASET="/data/kmirakho/JustImagine/modified_libero_rlds/libero_10_no_noops/1.0.0"
DEVICE="cuda:2"  # Change GPU here
TAG="openvla_reward_training"

# Run training
python scripts/train_with_openvla.py \
    --env libero-10 \
    --env_type normal \
    --alg rrd \
    --basis_alg sac \
    --use_openvla_features True \
    --openvla_checkpoint "$OPENVLA_CHECKPOINT" \
    --openvla_task_description "pick up the red cup" \
    --libero_dataset_path "$LIBERO_DATASET" \
    --libero_image_size 256 \
    --cuda True \
    --device "$DEVICE" \
    --epochs 10 \
    --cycles 50 \
    --iterations 100 \
    --train_batches 50 \
    --batch_size 32 \
    --rrd_batch_size 32 \
    --rrd_sample_size 32 \
    --rrd_reward_only True \
    --r_lr 3e-4 \
    --tag "$TAG" \
    --save_freq 2 \
    --checkpoint_dir log/checkpoints \
    --save_final True

