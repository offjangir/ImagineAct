#!/bin/bash
# RL Fine-tuning of OpenVLA with World Model

# Configuration
OPENVLA_CHECKPOINT="/data/kmirakho/JustImagine/openvla/openvla-7b-finetuned-libero-goal"
WORLD_MODEL_CHECKPOINT="/data/kmirakho/JustImagine/world-model-eval/mixed_openx_9robots_20frames_0p1actiondropout_580ksteps.pt"
INITIAL_STATES_PATH="/data/kmirakho/JustImagine/world-model-eval/libero_eval_data/libero_goal"  # Directory with initial state images
DEVICE="cuda:0"

# Optional: Pre-trained components
CRITIC_CHECKPOINT=""  # Leave empty to start fresh
REWARD_MODEL_CHECKPOINT=""  # Leave empty to start fresh

# Training hyperparameters
NUM_ENVS=8
MAX_EPISODE_STEPS=512
ROLLOUT_STEPS=2048
NUM_UPDATES=1000
PPO_EPOCHS=4
BATCH_SIZE=64

# Learning rates
VLA_LR=1e-5
CRITIC_LR=3e-4

# PPO hyperparameters
GAMMA=0.99
GAE_LAMBDA=0.95
CLIP_RATIO=0.2
VALUE_COEF=0.5
ENTROPY_COEF=0.01

# Logging
TAG="openvla_rl_worldmodel"
CHECKPOINT_DIR="log/checkpoints"
SAVE_FREQ=50
USE_WANDB="True"
WANDB_PROJECT="openvla-rl-worldmodel"

# Task descriptions
TASK_DESCRIPTIONS=("pick up the red cup" "put the bowl on the plate")

# Build command
CMD="python scripts/train_openvla_rl_worldmodel.py \
    --openvla_checkpoint \"$OPENVLA_CHECKPOINT\" \
    --world_model_checkpoint \"$WORLD_MODEL_CHECKPOINT\" \
    --initial_states_path \"$INITIAL_STATES_PATH\" \
    --num_envs $NUM_ENVS \
    --max_episode_steps $MAX_EPISODE_STEPS \
    --rollout_steps $ROLLOUT_STEPS \
    --num_updates $NUM_UPDATES \
    --ppo_epochs $PPO_EPOCHS \
    --batch_size $BATCH_SIZE \
    --vla_lr $VLA_LR \
    --critic_lr $CRITIC_LR \
    --gamma $GAMMA \
    --gae_lambda $GAE_LAMBDA \
    --clip_ratio $CLIP_RATIO \
    --value_coef $VALUE_COEF \
    --entropy_coef $ENTROPY_COEF \
    --device $DEVICE \
    --tag $TAG \
    --checkpoint_dir $CHECKPOINT_DIR \
    --save_freq $SAVE_FREQ"

# Add optional checkpoints
if [ -n "$CRITIC_CHECKPOINT" ]; then
    CMD="$CMD --critic_checkpoint \"$CRITIC_CHECKPOINT\""
fi

if [ -n "$REWARD_MODEL_CHECKPOINT" ]; then
    CMD="$CMD --reward_model_checkpoint \"$REWARD_MODEL_CHECKPOINT\""
fi

# Add task descriptions
if [ ${#TASK_DESCRIPTIONS[@]} -gt 0 ]; then
    CMD="$CMD --task_descriptions ${TASK_DESCRIPTIONS[@]}"
fi

# Add WandB
if [ "$USE_WANDB" = "True" ]; then
    CMD="$CMD --use_wandb --wandb_project $WANDB_PROJECT"
fi

# Execute
echo "Starting RL training with world model..."
echo "Command: $CMD"
echo ""
eval $CMD





