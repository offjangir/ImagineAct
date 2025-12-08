#!/bin/bash
# Training script for RRD with OpenVLA features
# Modify the variables below to customize training

set -e  # Exit on error

# ==============================================================================
# CONFIGURATION - Modify these parameters as needed
# ==============================================================================

# OpenVLA Configuration
USE_OPENVLA_FEATURES="True"
# IMPORTANT: Update this to point to your OpenVLA checkpoint
# Options:
#   1. Local path: "/path/to/openvla-7b-finetuned-libero-goal" (must exist)
#   2. HuggingFace model ID: "openvla/openvla-7b-finetuned-libero-goal" (will download automatically)
#   3. Other HF model: "username/model-name"
OPENVLA_CHECKPOINT="openvla/openvla-7b-finetuned-libero-goal"  # Using HuggingFace model ID (will download if needed)
# OPENVLA_CHECKPOINT="/path/to/openvla/model"  # Uncomment and update if you have a local checkpoint
# Task descriptions are now extracted from the dataset automatically!
# OPENVLA_TASK_DESCRIPTION is only used as fallback if dataset doesn't have task descriptions
OPENVLA_TASK_DESCRIPTION=""  # Leave empty to use dataset task descriptions, or set a fallback
OPENVLA_REWARD_HIDDEN_DIM=512
OPENVLA_REWARD_NUM_LAYERS=3

# Environment Configuration
ENV="libero-10"  # Environment name (can be libero-10, libero-goal, etc.)
ENV_TYPE="normal"
LIBERO_DATASET_PATH="/path/to/libero/dataset/libero_goal_no_noops/1.0.0"  # Update with your LIBERO dataset path
LIBERO_IMAGE_SIZE=256
LIBERO_MAX_TRAJECTORIES=""  # Maximum number of trajectories to load (empty = load all, e.g., "100" to load only 100)

# Algorithm Configuration
ALG="rrd"
BASIS_ALG="sac"
RRD_REWARD_ONLY="True"  # Set to False for full RRD training (reward + policy)
RRD_BIAS_CORRECTION="False"

# Training Configuration
EPOCHS=100               # Number of epochs (iterations per epoch = data_size / batch_size)
BATCH_SIZE=64           # Number of regular transitions for Q-network/policy training
RRD_BATCH_SIZE=64       # Total number of random subsequences for reward network training
RRD_SAMPLE_SIZE=32      # Size of each random subsequence (number of timesteps per subsequence)

# Learning Rates
R_LR=1e-4  # Reward network learning rate
PI_LR=0.0003  # Policy learning rate (if not reward-only)
Q_LR=0.0003  # Q-network learning rate (if not reward-only)

# Device Configuration
USE_CUDA="True"
# NOTE: Only a single device is supported (e.g., "cuda:0", "cuda:2", or "cpu")
# For multi-GPU training, you would need to modify the code to use DataParallel or DistributedDataParallel
DEVICE="cuda:2"  # Change to cuda:0, cuda:1, cuda:2, etc. or "cpu"

# Logging and Checkpoints
TAG="openvla_reward_libero_goal"
CHECKPOINT_DIR="log/checkpoints"
SAVE_FREQ=5  # Save checkpoint every N epochs (0 to disable)
SAVE_FINAL="True"

# WandB Configuration (for loss plotting)
USE_WANDB="True"        # Enable Weights & Biases logging
WANDB_PROJECT="rrd-openvla-libero"  # Your WandB project name
WANDB_ENTITY="interpretable_rl"    # Your WandB team/entity (username: km1369, team: interpretable_rl)
WANDB_RUN_NAME="openvla-libero-goal-full-dataset"  # Experiment/run name (leave empty to use tag, or set custom name)
# SECURITY NOTE: Consider using environment variable or wandb login instead of hardcoding API key
# You can also run: export WANDB_API_KEY="your-key" before running this script
WANDB_API_KEY="24b3f1a04c345168222fe23727e31129c3840580"  # Your WandB API key

# Export WandB API key as environment variable (WandB will automatically use it)
export WANDB_API_KEY="$WANDB_API_KEY"

# Testing Configuration
TEST_ROLLOUTS=5
TEST_TIMESTEPS=500

# Buffer Configuration
BUFFER_SIZE=100000
WARMUP=0  # No warmup for offline RL

# ==============================================================================
# SCRIPT EXECUTION - Usually no need to modify below
# ==============================================================================

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Print configuration
echo "=============================================================================="
echo "Training Configuration"
echo "=============================================================================="
echo "OpenVLA Features: $USE_OPENVLA_FEATURES"
echo "OpenVLA Checkpoint: $OPENVLA_CHECKPOINT"
if [ -z "$OPENVLA_TASK_DESCRIPTION" ]; then
    echo "Task Description: (extracted from dataset)"
else
    echo "Task Description: $OPENVLA_TASK_DESCRIPTION (fallback)"
fi
echo "Environment: $ENV"
echo "Algorithm: $ALG (basis: $BASIS_ALG)"
echo "Reward Only: $RRD_REWARD_ONLY"
echo "Device: $DEVICE"
echo "Tag: $TAG"
echo "WandB: $USE_WANDB (project: $WANDB_PROJECT, entity: $WANDB_ENTITY, run: $WANDB_RUN_NAME)"
echo "=============================================================================="
echo ""

# Build command
CMD="python scripts/train_with_openvla.py \
    --env $ENV \
    --env_type $ENV_TYPE \
    --alg $ALG \
    --basis_alg $BASIS_ALG \
    --use_openvla_features $USE_OPENVLA_FEATURES \
    --openvla_checkpoint $OPENVLA_CHECKPOINT \
    --openvla_task_description \"${OPENVLA_TASK_DESCRIPTION:-}\" \
    --openvla_reward_hidden_dim $OPENVLA_REWARD_HIDDEN_DIM \
    --openvla_reward_num_layers $OPENVLA_REWARD_NUM_LAYERS \
    --libero_dataset_path $LIBERO_DATASET_PATH \
    --libero_image_size $LIBERO_IMAGE_SIZE \
    ${LIBERO_MAX_TRAJECTORIES:+--libero_max_trajectories $LIBERO_MAX_TRAJECTORIES} \
    --cuda $USE_CUDA \
    --device $DEVICE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --rrd_batch_size $RRD_BATCH_SIZE \
    --rrd_sample_size $RRD_SAMPLE_SIZE \
    --rrd_reward_only $RRD_REWARD_ONLY \
    --rrd_bias_correction $RRD_BIAS_CORRECTION \
    --r_lr $R_LR \
    --buffer_size $BUFFER_SIZE \
    --warmup $WARMUP \
    --tag $TAG \
    --checkpoint_dir $CHECKPOINT_DIR \
    --save_freq $SAVE_FREQ \
    --save_final $SAVE_FINAL \
    --test_rollouts $TEST_ROLLOUTS \
    --test_timesteps $TEST_TIMESTEPS \
    --use_wandb $USE_WANDB \
    --wandb_project $WANDB_PROJECT"

# Add wandb_entity if specified
if [ -n "$WANDB_ENTITY" ]; then
    CMD="$CMD --wandb_entity $WANDB_ENTITY"
fi

# Add wandb_run_name if specified
if [ -n "$WANDB_RUN_NAME" ]; then
    CMD="$CMD --wandb_run_name $WANDB_RUN_NAME"
fi

# Add policy/Q learning rates if not reward-only
if [ "$RRD_REWARD_ONLY" = "False" ]; then
    CMD="$CMD --pi_lr $PI_LR --q_lr $Q_LR"
fi

# Execute command
echo "Starting training..."
echo "Command: $CMD"
echo ""
eval $CMD

echo ""
echo "Training completed!"

