"""
Example script showing how to use OpenVLA features for reward model training.

This demonstrates how to integrate OpenVLA into the Randomized-Return-Decomposition
training pipeline.
"""

import sys
import os

# Add paths
sys.path.append('/path/to/openvla')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from common import get_args
from envs import make_env
from algorithm import create_agent
from learner import create_learner


def load_openvla(checkpoint_path, device="cuda:0"):
    """Load OpenVLA model and processor."""
    print(f"Loading OpenVLA from {checkpoint_path}...")
    
    # Register OpenVLA model to HF Auto Classes
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    
    # Load model
    vla_model = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device)
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        local_files_only=True
    )
    
    print("OpenVLA loaded successfully!")
    return vla_model, processor


def main():
    # Get training arguments
    args = get_args()
    
    # Load OpenVLA model
    openvla_checkpoint = args.openvla_checkpoint if hasattr(args, 'openvla_checkpoint') else "path/to/openvla/checkpoint"
    vla_model, processor = load_openvla(openvla_checkpoint, device=args.device if hasattr(args, 'device') else "cuda:0")
    
    # Configure OpenVLA features
    args.use_openvla_features = True
    args.openvla_model = vla_model
    args.openvla_processor = processor
    
    # Set task description (for LIBERO)
    if args.env_category == 'libero':
        # You can set a default or get from environment
        args.task_description = getattr(args, 'task_description', 'pick up the red cup')
    
    # Optional: Configure reward network architecture
    args.openvla_reward_hidden_dim = getattr(args, 'openvla_reward_hidden_dim', 512)
    args.openvla_reward_num_layers = getattr(args, 'openvla_reward_num_layers', 3)
    
    # Create environment
    env = make_env(args)
    
    # Create agent (will use OpenVLARewardNet if use_openvla_features=True)
    agent = create_agent(args)
    
    # Create learner
    learner = create_learner(args)
    
    # Create buffer (if needed)
    from algorithm.replay_buffer import create_buffer
    buffer = create_buffer(args)
    
    # Training loop
    print("Starting training with OpenVLA features...")
    learner.learn(args, env, agent, buffer)
    
    print("Training complete!")


if __name__ == "__main__":
    main()

