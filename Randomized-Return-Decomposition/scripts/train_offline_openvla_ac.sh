#!/bin/bash
# Training script for offline actor-critic with OpenVLA features
# This script trains OpenVLA as an offline RL agent using precomputed features

# Ensure we're using bash (not sh)
if [ -z "$BASH_VERSION" ]; then
    echo "Error: This script requires bash. Please run with: bash $0"
    echo "Or make it executable and run: ./$0"
    exit 1
fi

set -e  # Exit on error

# ==============================================================================
# CONFIGURATION - Modify these parameters as needed
# ==============================================================================

# Data Paths
FEATURES_CACHE_PATH="log/feature_cache/openvla_features.pkl"  # Update with your feature cache path
RLDS_DATASET_PATH="/path/to/libero/dataset/libero_goal_no_noops/1.0.0"  # Update with your LIBERO dataset path

# Checkpoint Paths (optional - for loading pre-trained models)
REWARD_MODEL_CHECKPOINT=""  # Path to pre-trained reward model checkpoint (empty = initialize from scratch)
CRITIC_CHECKPOINT=""  # Path to pre-trained critic checkpoint (empty = initialize from scratch)

# OpenVLA Model Configuration
OPENVLA_CHECKPOINT="openvla/openvla-7b-finetuned-libero-goal"  # HuggingFace model ID or local path
# OPENVLA_CHECKPOINT="/path/to/openvla/model"  # Uncomment for local path
UNNORM_KEY=""  # Dataset key for action normalization (empty = auto-detect)

# Model Architecture
CRITIC_HIDDEN_DIMS="2048 512 128"  # Space-separated list of hidden dimensions
REWARD_HIDDEN_DIM=512
REWARD_NUM_LAYERS=3
POOL_FEATURES="True"  # Pool language/vision features to fixed-size vectors
POOL_METHOD="mean"  # "mean" or "max"

# LoRA Configuration (for decoder layers)
USE_LORA="True"  # Use LoRA instead of full fine-tuning (much more parameter-efficient)
LORA_RANK=32  # LoRA rank (typical range: 8-64, higher = more parameters but more capacity)
LORA_ALPHA=32  # LoRA alpha scaling factor (usually same as rank)

# Training Configuration
EPOCHS=121
# Note: Batch size reduced to 8 due to long sequences (~281 tokens = 25 lang + 256 vision patches)
# If you have more GPU memory, you can increase this (e.g., 16 or 32)
BATCH_SIZE=16  # Reduced for memory efficiency (use gradient_accumulation_steps to simulate larger batches)
GRADIENT_ACCUMULATION_STEPS=16  # Accumulate gradients over N steps (effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
NUM_WORKERS=4  # Data loader workers

# Quick Testing/Debugging (set to limit dataset size for faster error checking)
# Set MAX_EPISODES=20 to test with only 20 episodes (catches errors quickly)
# Set MAX_TRANSITIONS=1000 to test with only 1000 transitions (alternative to episodes)
# For production training, set both to empty string "" to use full dataset
MAX_EPISODES=""  # Maximum number of episodes to use (empty = use all, e.g., "20" for quick testing)
MAX_TRANSITIONS=""  # Maximum number of transitions to use (empty = use all, e.g., "1000" for quick testing)

# Learning Rates
VLA_LR=5e-6  # OpenVLA (actor) learning rate
CRITIC_LR=3e-4  # Critic learning rate
REWARD_LR=1e-4  # Reward network learning rate (if training reward network)
REWARD_COEF=1.0  # Reward network loss coefficient
TRAIN_REWARD_NETWORK="False"  # Set to "True" to train reward network on ground truth rewards

# RL Hyperparameters
GAMMA=0.99  # Discount factor
GAE_LAMBDA=0.95  # GAE lambda parameter (smoothing factor)
BC_COEF=1.0  # Behavior cloning coefficient (increased to match actor loss scale and ensure BC learning)
ENTROPY_COEF=0.0  # Entropy regularization coefficient (0 = disabled)
CRITIC_COEF=1.0  # Critic loss coefficient
ADVANTAGE_CLIP=10.0  # Clip advantages to [-ADVANTAGE_CLIP, ADVANTAGE_CLIP] to prevent very large actor losses (0 = disabled)
GRAD_CLIP_NORM=1.0  # Gradient clipping norm (0 = disabled, typical: 0.5-2.0)
CRITIC_WARMUP_EPOCHS=10  # Number of epochs to train only critic (0 = disabled)

# Device Configuration
DEVICE="cuda:0"  # Change to cuda:0, cuda:1, cuda:2, etc. or "cpu" (ignored if NUM_GPUS > 1)
NUM_GPUS=3  # Number of GPUs to use (1-4, uses DDP if > 1)
# GPU_IDS: Comma-separated list of GPU IDs to use (e.g., "0,2,3,7")
# If set, these specific GPUs will be used. If empty, uses sequential GPUs starting from 0.
# Example: GPU_IDS="0,2,3,7" will use GPUs 0, 2, 3, and 7
GPU_IDS="5,6,7"  # Leave empty to use sequential GPUs (0, 1, 2, 3, ...) or specify like "0,2,3,7"
SEED=42  # Random seed

# Logging and Checkpoints
TAG="offline_openvla_ac_libero_goal"
CHECKPOINT_DIR="log/checkpoints"
SAVE_FREQ=15  # Save checkpoint every N epochs (0 to disable)
SAVE_FINAL="True"

# WandB Configuration
USE_WANDB="True"  # Enable Weights & Biases logging
WANDB_PROJECT="offline-openvla-ac"
WANDB_ENTITY=""  # Your WandB team/entity (leave empty for personal account)
WANDB_RUN_NAME=""  # Experiment/run name (leave empty to use tag)

# Export WandB API key if needed (uncomment and set)
# export WANDB_API_KEY="your-api-key-here"

# ==============================================================================
# SCRIPT EXECUTION - Usually no need to modify below
# ==============================================================================

# Get script directory (compatible with both bash and sh)
if [ -n "$BASH_VERSION" ]; then
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
else
    SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
fi
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"
echo "Working directory: $(pwd)"

# Print configuration
echo "=============================================================================="
echo "Offline Actor-Critic Training Configuration"
echo "=============================================================================="
echo "Features Cache: $FEATURES_CACHE_PATH"
echo "RLDS Dataset: $RLDS_DATASET_PATH"
echo "OpenVLA Checkpoint: $OPENVLA_CHECKPOINT"
if [ -z "$UNNORM_KEY" ]; then
    UNNORM_KEY_DISPLAY="auto-detect"
else
    UNNORM_KEY_DISPLAY="$UNNORM_KEY"
fi
echo "Unnorm Key: $UNNORM_KEY_DISPLAY"
echo ""
echo "Model Checkpoints:"
if [ -n "$REWARD_MODEL_CHECKPOINT" ]; then
    echo "  Reward Model: $REWARD_MODEL_CHECKPOINT"
else
    echo "  Reward Model: (initialize from scratch)"
fi
if [ -n "$CRITIC_CHECKPOINT" ]; then
    echo "  Critic: $CRITIC_CHECKPOINT"
else
    echo "  Critic: (initialize from scratch)"
fi
echo ""
echo "Model Architecture:"
echo "  Critic Hidden Dims: $CRITIC_HIDDEN_DIMS"
echo "  Reward Hidden Dim: $REWARD_HIDDEN_DIM"
echo "  Reward Num Layers: $REWARD_NUM_LAYERS"
echo "  Pool Features: $POOL_FEATURES"
echo "  Pool Method: $POOL_METHOD"
if [ "$USE_LORA" = "True" ]; then
    echo "  LoRA: Enabled (rank=$LORA_RANK, alpha=$LORA_ALPHA)"
else
    echo "  LoRA: Disabled (full fine-tuning)"
fi
echo ""
echo "Training:"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Num Workers: $NUM_WORKERS"
if [ -n "$MAX_EPISODES" ]; then
    echo "  Max Episodes: $MAX_EPISODES (quick testing mode)"
fi
if [ -n "$MAX_TRANSITIONS" ]; then
    echo "  Max Transitions: $MAX_TRANSITIONS (quick testing mode)"
fi
echo ""
echo "Learning Rates:"
echo "  VLA (Actor): $VLA_LR"
echo "  Critic: $CRITIC_LR"
echo "  Reward: $REWARD_LR"
echo ""
echo "Reward Network Training:"
echo "  Train Reward Network: $TRAIN_REWARD_NETWORK"
echo "  Reward Loss Coefficient: $REWARD_COEF"
echo ""
echo "RL Hyperparameters:"
echo "  Gamma: $GAMMA"
echo "  GAE Lambda: $GAE_LAMBDA"
echo "  BC Coefficient: $BC_COEF"
echo "  Entropy Coefficient: $ENTROPY_COEF"
echo "  Critic Coefficient: $CRITIC_COEF"
echo ""
echo "Device: $DEVICE"
echo "Num GPUs: $NUM_GPUS"
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "  Using DistributedDataParallel (DDP) for multi-GPU training"
    if [ -n "$GPU_IDS" ]; then
        echo "  Using specific GPUs: $GPU_IDS"
    else
        echo "  Using sequential GPUs: 0 to $((NUM_GPUS - 1))"
    fi
fi
echo "Seed: $SEED"
echo "Tag: $TAG"
echo "Checkpoint Dir: $CHECKPOINT_DIR"
echo "Save Freq: $SAVE_FREQ"
echo "Save Final: $SAVE_FINAL"
echo ""
echo "WandB: $USE_WANDB"
if [ "$USE_WANDB" = "True" ]; then
    echo "  Project: $WANDB_PROJECT"
    if [ -n "$WANDB_ENTITY" ]; then
        echo "  Entity: $WANDB_ENTITY"
    fi
    if [ -z "$WANDB_RUN_NAME" ]; then
        echo "  Run Name: $TAG"
    else
        echo "  Run Name: $WANDB_RUN_NAME"
    fi
fi
echo "=============================================================================="
echo ""

# Verify we're in the right directory
if [ ! -f "scripts/train_offline_openvla_ac.py" ]; then
    echo "Error: scripts/train_offline_openvla_ac.py not found in $(pwd)"
    echo "Please run this script from the Randomized-Return-Decomposition directory"
    exit 1
fi

# Build command - use torchrun for multi-GPU, python for single GPU
if [ "$NUM_GPUS" -gt 1 ]; then
    # Set CUDA_VISIBLE_DEVICES if specific GPU IDs are provided
    if [ -n "$GPU_IDS" ]; then
        export CUDA_VISIBLE_DEVICES="$GPU_IDS"
        echo "Setting CUDA_VISIBLE_DEVICES=$GPU_IDS"
    fi
    
    # Use torchrun for DDP
    CMD="torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 scripts/train_offline_openvla_ac.py \
    --num_gpus $NUM_GPUS \
    --features_cache_path \"$FEATURES_CACHE_PATH\" \
    --rlds_dataset_path \"$RLDS_DATASET_PATH\" \
    --openvla_checkpoint \"$OPENVLA_CHECKPOINT\" \
    --critic_hidden_dims $CRITIC_HIDDEN_DIMS \
    ${REWARD_MODEL_CHECKPOINT:+--reward_model_checkpoint "$REWARD_MODEL_CHECKPOINT"} \
    ${CRITIC_CHECKPOINT:+--critic_checkpoint "$CRITIC_CHECKPOINT"} \
    --reward_hidden_dim $REWARD_HIDDEN_DIM \
    --reward_num_layers $REWARD_NUM_LAYERS \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --num_workers $NUM_WORKERS \
    ${MAX_EPISODES:+--max_episodes $MAX_EPISODES} \
    ${MAX_TRANSITIONS:+--max_transitions $MAX_TRANSITIONS} \
    --vla_lr $VLA_LR \
    --critic_lr $CRITIC_LR \
    --reward_lr $REWARD_LR \
    --reward_coef $REWARD_COEF \
    --gamma $GAMMA \
    --gae_lambda $GAE_LAMBDA \
    --bc_coef $BC_COEF \
    --entropy_coef $ENTROPY_COEF \
    --critic_coef $CRITIC_COEF \
    --advantage_clip $ADVANTAGE_CLIP \
    --grad_clip_norm $GRAD_CLIP_NORM \
    --critic_warmup_epochs $CRITIC_WARMUP_EPOCHS \
    --device $DEVICE \
    --seed $SEED \
    --tag $TAG \
    --checkpoint_dir $CHECKPOINT_DIR \
    --save_freq $SAVE_FREQ"
else
    # Use regular python for single GPU
    CMD="python scripts/train_offline_openvla_ac.py \
    --num_gpus 1 \
    --features_cache_path \"$FEATURES_CACHE_PATH\" \
    --rlds_dataset_path \"$RLDS_DATASET_PATH\" \
    --openvla_checkpoint \"$OPENVLA_CHECKPOINT\" \
    --critic_hidden_dims $CRITIC_HIDDEN_DIMS \
    ${REWARD_MODEL_CHECKPOINT:+--reward_model_checkpoint "$REWARD_MODEL_CHECKPOINT"} \
    ${CRITIC_CHECKPOINT:+--critic_checkpoint "$CRITIC_CHECKPOINT"} \
    --reward_hidden_dim $REWARD_HIDDEN_DIM \
    --reward_num_layers $REWARD_NUM_LAYERS \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --num_workers $NUM_WORKERS \
    ${MAX_EPISODES:+--max_episodes $MAX_EPISODES} \
    ${MAX_TRANSITIONS:+--max_transitions $MAX_TRANSITIONS} \
    --vla_lr $VLA_LR \
    --critic_lr $CRITIC_LR \
    --reward_lr $REWARD_LR \
    --reward_coef $REWARD_COEF \
    --gamma $GAMMA \
    --gae_lambda $GAE_LAMBDA \
    --bc_coef $BC_COEF \
    --entropy_coef $ENTROPY_COEF \
    --critic_coef $CRITIC_COEF \
    --advantage_clip $ADVANTAGE_CLIP \
    --grad_clip_norm $GRAD_CLIP_NORM \
    --critic_warmup_epochs $CRITIC_WARMUP_EPOCHS \
    --device $DEVICE \
    --seed $SEED \
    --tag $TAG \
    --checkpoint_dir $CHECKPOINT_DIR \
    --save_freq $SAVE_FREQ"
fi

# Add optional arguments (for both single and multi-GPU)
if [ -n "$UNNORM_KEY" ]; then
    CMD="$CMD --unnorm_key $UNNORM_KEY"
fi

if [ "$POOL_FEATURES" = "True" ]; then
    CMD="$CMD --pool_features"
fi

if [ -n "$POOL_METHOD" ]; then
    CMD="$CMD --pool_method $POOL_METHOD"
fi

if [ "$USE_LORA" = "True" ]; then
    CMD="$CMD --use_lora"
    CMD="$CMD --lora_rank $LORA_RANK"
    if [ -n "$LORA_ALPHA" ]; then
        CMD="$CMD --lora_alpha $LORA_ALPHA"
    fi
fi

if [ "$SAVE_FINAL" = "True" ]; then
    CMD="$CMD --save_final"
fi

if [ "$TRAIN_REWARD_NETWORK" = "True" ]; then
    CMD="$CMD --train_reward_network"
fi

if [ "$USE_WANDB" = "True" ]; then
    CMD="$CMD --use_wandb"
    CMD="$CMD --wandb_project $WANDB_PROJECT"
    if [ -n "$WANDB_ENTITY" ]; then
        CMD="$CMD --wandb_entity $WANDB_ENTITY"
    fi
    if [ -n "$WANDB_RUN_NAME" ]; then
        CMD="$CMD --wandb_run_name \"$WANDB_RUN_NAME\""
    fi
fi

# Execute command
echo "Starting training..."
if [ "$NUM_GPUS" -gt 1 ]; then
    if [ -n "$GPU_IDS" ]; then
        echo "Using torchrun with $NUM_GPUS GPUs (DDP mode) on devices: $GPU_IDS"
    else
        echo "Using torchrun with $NUM_GPUS GPUs (DDP mode)"
    fi
else
    echo "Using single GPU mode"
fi
echo "Command: $CMD"
echo ""
eval $CMD

echo ""
echo "Training completed!"




