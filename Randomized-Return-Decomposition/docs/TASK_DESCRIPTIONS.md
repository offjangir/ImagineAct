# Task Descriptions in LIBERO Training

## Overview

Task descriptions are now **automatically extracted from the LIBERO dataset** instead of using a fixed description. Each episode in LIBERO has its own task description (language instruction) that is used for OpenVLA feature extraction.

## How It Works

1. **Dataset Loading**: When loading LIBERO episodes, the system extracts `language_instruction` from each episode
2. **Storage**: Task descriptions are stored in `Trajectory` objects
3. **Batch Sampling**: Task descriptions are included in batches as `task_descriptions` and `rrd_task_descriptions`
4. **Feature Extraction**: OpenVLA uses the appropriate task description for each observation

## Dataset Format

LIBERO RLDS datasets store language instructions in one of these formats:

- **Episode-level**: `episode['language_instruction']` or `episode['task']['language_instruction']`
- **Step-level**: `step['language_instruction']` (less common)

The code automatically detects and extracts from both locations.

## Configuration

### Using Dataset Task Descriptions (Recommended)

Leave `--openvla_task_description` empty or unset:

```bash
python scripts/train_with_openvla.py \
    --use_openvla_features True \
    --openvla_checkpoint /path/to/checkpoint \
    # No --openvla_task_description needed!
    ...
```

### Using Fallback Task Description

If the dataset doesn't have task descriptions, you can provide a fallback:

```bash
python scripts/train_with_openvla.py \
    --use_openvla_features True \
    --openvla_checkpoint /path/to/checkpoint \
    --openvla_task_description "pick up the red cup" \
    ...
```

The fallback is only used if:
- The dataset doesn't contain `language_instruction` fields
- Task descriptions can't be extracted from episodes

## Benefits

- **Task-Specific Features**: Each task gets its own language features
- **Multi-Task Training**: Can train on multiple LIBERO tasks simultaneously
- **Automatic**: No need to manually specify task descriptions
- **Flexible**: Falls back to default if dataset doesn't have descriptions

## Example

For LIBERO-10 dataset with 10 different tasks:
- Episode 1: "pick up the red cup"
- Episode 2: "place the bowl on the plate"
- Episode 3: "open the drawer"
- ...

Each episode automatically uses its own task description for feature extraction!

