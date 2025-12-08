#!/bin/bash

# Ensure the script is run with bash (not sh/dash)
if [ -z "$BASH_VERSION" ]; then
  echo "Error: This script must be run with bash. Use: bash $0 ..." >&2
  exit 1
fi

set -euo pipefail

# Default paths and settings
CHECKPOINT_PATH="/data/kmirakho/JustImagine/Randomized-Return-Decomposition/log/checkpoints/offline_openvla_ac_libero_goal_final.pt"
OPENVLA_CHECKPOINT="openvla/openvla-7b-finetuned-libero-goal"

# Default LIBERO Goal task for single-task runs (can be overridden)
TASK_NAME="put_the_bowl_on_the_plate"

# Number of eval episodes per task when evaluating multiple tasks
NUM_EVAL_PER_TASK=1

DEVICE="cuda:4"
MAX_STEPS=200

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --checkpoint_path) CHECKPOINT_PATH="$2"; shift ;;
        --openvla_checkpoint) OPENVLA_CHECKPOINT="$2"; shift ;;
        --task_name) TASK_NAME="$2"; shift ;;
        --num_eval_per_task) NUM_EVAL_PER_TASK="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        --max_steps) MAX_STEPS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

cd /data/kmirakho/JustImagine/Randomized-Return-Decomposition

echo "Running evaluation over all LIBERO Goal tasks"
echo "  Checkpoint:          $CHECKPOINT_PATH"
echo "  OpenVLA checkpoint:  $OPENVLA_CHECKPOINT"
echo "  Num eval per task:   $NUM_EVAL_PER_TASK"
echo "  Device:              $DEVICE"
echo "  Max steps per ep:    $MAX_STEPS"
echo

# List of all LIBERO Goal tasks (B DDL stems)
TASKS=(
  "open_the_middle_drawer_of_the_cabinet"
  "open_the_top_drawer_and_put_the_bowl_inside"
  "push_the_plate_to_the_front_of_the_stove"
  "put_the_bowl_on_the_plate"
  "put_the_bowl_on_the_stove"
  "put_the_bowl_on_top_of_the_cabinet"
  "put_the_cream_cheese_in_the_bowl"
  "put_the_wine_bottle_on_the_rack"
  "put_the_wine_bottle_on_top_of_the_cabinet"
)

for TASK in "${TASKS[@]}"; do
  echo "================================================================"
  echo "Evaluating task: $TASK"
  echo "================================================================"
  python scripts/evaluate_offline_openvla_ac.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --openvla_checkpoint "$OPENVLA_CHECKPOINT" \
    --task_name "$TASK" \
    --num_episodes "$NUM_EVAL_PER_TASK" \
    --device "$DEVICE" \
    --max_steps "$MAX_STEPS"
  echo
done


