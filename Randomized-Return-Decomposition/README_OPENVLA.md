# Using OpenVLA Features for Reward Model Training

This integration allows you to use OpenVLA's language and vision features to train reward models in Randomized-Return-Decomposition.

## Quick Start

### 1. Load OpenVLA Model

```python
import sys
sys.path.append('/path/to/openvla')

from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Register and load
AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

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

### 2. Configure Training

```python
args.use_openvla_features = True
args.openvla_model = vla_model
args.openvla_processor = processor
args.task_description = "pick up the red cup"  # For LIBERO
```

### 3. Run Training

Training will automatically:
- Extract language and vision features from observations
- Use `OpenVLARewardNet` instead of standard reward networks
- Train on these rich semantic features

## Architecture

### OpenVLARewardNet

- **Input**: Language features, vision features, actions, and their differences
- **Processing**: 
  1. Pool language and vision features to fixed-size vectors
  2. Compute feature differences (current - next)
  3. Concatenate with actions
  4. MLP layers → reward prediction

### Feature Extraction

- **Language Features**: Token embeddings from OpenVLA's language model
- **Vision Features**: Patch features from OpenVLA's vision backbone (before projection)

## Files Added

1. `algorithm/openvla_reward_net.py` - OpenVLA reward network and feature extractor
2. `docs/OPENVLA_REWARD_TRAINING.md` - Detailed documentation
3. `examples/openvla_reward_example.py` - Usage example

## Integration Points

- `algorithm/rrd_torch.py` - Modified to support OpenVLA features
- Batch preprocessing extracts features automatically
- Reward network selection based on `use_openvla_features` flag

## Benefits

- **Rich Semantics**: Leverages pre-trained VLM representations
- **Task Awareness**: Language features encode task information
- **Transfer Learning**: Benefits from large-scale vision-language pre-training

