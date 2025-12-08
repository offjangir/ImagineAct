#!/bin/bash
# Alternative: Configuration file approach for training with OpenVLA
# Source this file to set variables, then run train_with_openvla.py

# ==============================================================================
# CONFIGURATION FILE
# ==============================================================================
# Usage:
#   source scripts/train_openvla_config.sh
#   python scripts/train_with_openvla.py [additional args will override config]
# Or:
#   bash -c "source scripts/train_openvla_config.sh && python scripts/train_with_openvla.py"
# ==============================================================================

# OpenVLA Configuration
export USE_OPENVLA_FEATURES="True"
export OPENVLA_CHECKPOINT="/data/kmirakho/JustImagine/openvla/openvla-7b-finetuned-libero-goal"
export OPENVLA_TASK_DESCRIPTION="pick up the red cup"
export OPENVLA_REWARD_HIDDEN_DIM=512
export OPENVLA_REWARD_NUM_LAYERS=3

# Environment Configuration
export ENV="libero-10"
export ENV_TYPE="normal"
export LIBERO_DATASET_PATH="/data/kmirakho/JustImagine/modified_libero_rlds/libero_10_no_noops/1.0.0"
export LIBERO_IMAGE_SIZE=256

# Algorithm Configuration
export ALG="rrd"
export BASIS_ALG="sac"
export RRD_REWARD_ONLY="True"
export RRD_BIAS_CORRECTION="False"

# Training Configuration
export EPOCHS=10
export CYCLES=50
export ITERATIONS=100
export TRAIN_BATCHES=50
export BATCH_SIZE=32
export RRD_BATCH_SIZE=32
export RRD_SAMPLE_SIZE=32

# Learning Rates
export R_LR=0.0003
export PI_LR=0.0003
export Q_LR=0.0003

# Device Configuration
export USE_CUDA="True"
export DEVICE="cuda:2"

# Logging and Checkpoints
export TAG="openvla_reward_libero10"
export CHECKPOINT_DIR="log/checkpoints"
export SAVE_FREQ=2
export SAVE_FINAL="True"

# Testing Configuration
export TEST_ROLLOUTS=5
export TEST_TIMESTEPS=500

# Buffer Configuration
export BUFFER_SIZE=100000
export WARMUP=0

echo "Configuration loaded. Variables exported to environment."

