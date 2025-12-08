# Training Reward Models with OpenVLA Features

This document explains how to use OpenVLA language and vision features to train reward models in Randomized-Return-Decomposition.

## Overview

Instead of using raw observations (images or state vectors), you can use pre-extracted features from OpenVLA:
- **Language Features**: Token embeddings from the language model
- **Vision Features**: Patch features from the vision backbone (before projection)

These features provide rich semantic representations that can improve reward model learning.

## Setup

### 1. Load OpenVLA Model

First, load your OpenVLA model and processor:

```python
import sys
sys.path.append('/path/to/openvla')

from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Register OpenVLA model
AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

# Load model and processor
checkpoint_path = "path/to/openvla/checkpoint"
vla_model = AutoModelForVision2Seq.from_pretrained(
    checkpoint_path,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    local_files_only=True,
).to("cuda:0")

processor = AutoProcessor.from_pretrained(
    checkpoint_path,
    trust_remote_code=True,
    local_files_only=True
)
```

### 2. Configure Training Arguments

Add OpenVLA-related arguments to your training config:

```python
args.use_openvla_features = True
args.openvla_model = vla_model
args.openvla_processor = processor
args.task_description = "pick up the red cup"  # Task description for LIBERO
args.openvla_reward_hidden_dim = 512  # Hidden dimension for reward network
args.openvla_reward_num_layers = 3  # Number of MLP layers
```

### 3. Run Training

The training code will automatically:
1. Extract OpenVLA features from observations
2. Use `OpenVLARewardNet` instead of standard reward networks
3. Train the reward model on these features

## Architecture

### OpenVLARewardNet

The reward network architecture:

```
Input:
  - Language features: [batch, seq_len, embed_dim]
  - Vision features: [batch, num_patches, embed_dim]
  - Actions: [batch, act_dim]
  - Next language features: [batch, seq_len, embed_dim]
  - Next vision features: [batch, num_patches, embed_dim]

Processing:
  1. Pool language features → [batch, hidden_dim//2]
  2. Pool vision features → [batch, hidden_dim//2]
  3. Compute feature differences (current - next)
  4. Concatenate: [pooled_lang, pooled_vis, lang_diff, vis_diff, actions]
  5. MLP layers → reward prediction
```

### Feature Extraction

The `OpenVLAFeatureExtractor`:
1. Takes images and optional task descriptions
2. Processes through OpenVLA processor
3. Extracts:
   - Language embeddings via `get_input_embeddings()`
   - Vision patch features via `vision_backbone()`

## Usage Example

```python
from algorithm import create_agent
from envs import make_env

# Create environment
env = make_env(args)

# Set up OpenVLA
args.use_openvla_features = True
args.openvla_model = vla_model
args.openvla_processor = processor
args.task_description = env.get_task_description()  # If available

# Create agent (will use OpenVLARewardNet)
agent = create_agent(args)

# Training proceeds as normal
# The reward network will automatically use OpenVLA features
```

## Benefits

1. **Rich Semantic Features**: Language and vision features capture high-level semantics
2. **Transfer Learning**: Leverages pre-trained VLM representations
3. **Task Awareness**: Language features can encode task-specific information
4. **Efficiency**: Pre-extracted features can be cached for faster training

## Notes

- Feature extraction happens during batch preprocessing
- Features are extracted on-the-fly (not pre-computed)
- For faster training, consider pre-extracting and caching features
- Vision features use raw patch features (before projection) for more flexibility

