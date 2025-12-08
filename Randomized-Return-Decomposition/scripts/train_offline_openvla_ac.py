#!/usr/bin/env python3
"""
Training script for offline actor-critic with OpenVLA features.

This script trains OpenVLA as an offline RL agent using:
- Precomputed OpenVLA features (vision + language)
- OpenVLARewardNet for dense rewards
- GAE for advantage estimation
- Behavior cloning for stability
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import wandb

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from offline_openvla_ac import (
    ValueCritic,
    compute_ac_losses,
    training_step,
    OfflineOpenVLADataset,
)
from algorithm.openvla_reward_net import OpenVLARewardNet


def load_openvla_model(checkpoint_path, device="cuda:0", verbose=True):
    """Load OpenVLA model and processor.
    
    Args:
        checkpoint_path: Either a local directory path or a HuggingFace model ID
        device: Device to load model on
        verbose: If True, print loading progress (set to False for DDP non-main processes)
    """
    # Determine if it's a local path or HuggingFace model ID (before try block)
    is_local_path = os.path.exists(checkpoint_path) and os.path.isdir(checkpoint_path)
    
    # Check if we're in DDP and should suppress prints
    should_print = verbose
    if verbose:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                should_print = (dist.get_rank() == 0)
        except (ImportError, AttributeError):
            pass  # Not using DDP, print normally
    
    try:
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
        import sys
        sys.path.append(os.path.join(project_root, "../openvla_RL"))
        
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        
        # Register OpenVLA classes
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        
        if is_local_path:
            # Local checkpoint
            config_path = os.path.join(checkpoint_path, "config.json")
            if not os.path.exists(config_path):
                raise ValueError(
                    f"Checkpoint directory does not contain config.json: {checkpoint_path}\n"
                    f"This doesn't appear to be a valid OpenVLA checkpoint."
                )
            abs_checkpoint_path = os.path.abspath(checkpoint_path)
            if should_print:
                print(f"Loading OpenVLA from local path: {abs_checkpoint_path}...", flush=True)
            local_files_only = True
        else:
            # HuggingFace model ID (e.g., "openvla/openvla-7b-finetuned-libero-goal")
            if should_print:
                print(f"Loading OpenVLA from HuggingFace: {checkpoint_path}...", flush=True)
                print(f"This will use cached files if available, or download if not cached.", flush=True)
            abs_checkpoint_path = checkpoint_path
            local_files_only = False  # Allow downloading if not cached
        
        # Load model
        vla_model = AutoModelForVision2Seq.from_pretrained(
            abs_checkpoint_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=local_files_only,
        ).to(device)
        
        processor = AutoProcessor.from_pretrained(
            abs_checkpoint_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        
        return vla_model, processor
    except Exception as e:
        if should_print:
            print(f"Error loading OpenVLA model: {e}", flush=True)
            if is_local_path:
                print(f"\nPossible solutions:", flush=True)
                print(f"1. Verify the checkpoint path is correct: {checkpoint_path}", flush=True)
                print(f"2. Ensure the checkpoint directory contains config.json and model files", flush=True)
            else:
                print(f"\nPossible solutions:", flush=True)
                print(f"1. Check your internet connection to download from HuggingFace", flush=True)
                print(f"2. Use a local checkpoint path instead: --openvla_checkpoint /path/to/local/checkpoint", flush=True)
                print(f"3. The model may need to be downloaded first when online", flush=True)
        raise


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train offline actor-critic with OpenVLA")
    
    # Data paths
    parser.add_argument("--features_cache_path", type=str, required=True,
                       help="Path to precomputed OpenVLA features pickle file")
    parser.add_argument("--rlds_dataset_path", type=str, required=True,
                       help="Path to RLDS dataset directory")
    
    # Checkpoint paths (optional - for loading pre-trained models)
    parser.add_argument("--reward_model_checkpoint", type=str, default=None,
                       help="Path to pre-trained reward model checkpoint (optional, if None, initializes from scratch)")
    parser.add_argument("--critic_checkpoint", type=str, default=None,
                       help="Path to pre-trained critic checkpoint (optional, if None, initializes from scratch)")
    
    # OpenVLA model
    parser.add_argument("--openvla_checkpoint", type=str, required=True,
                       help="Path to OpenVLA checkpoint")
    parser.add_argument("--unnorm_key", type=str, default=None,
                       help="Dataset key for action normalization stats")
    
    # Model architecture
    parser.add_argument("--critic_hidden_dims", type=int, nargs="+", default=[512, 512],
                       help="Critic hidden layer dimensions")
    parser.add_argument("--reward_hidden_dim", type=int, default=512,
                       help="Reward network hidden dimension")
    parser.add_argument("--reward_num_layers", type=int, default=3,
                       help="Reward network number of layers")
    
    # LoRA configuration
    parser.add_argument("--use_lora", action="store_true",
                       help="Use LoRA (Low-Rank Adaptation) for decoder layers instead of full fine-tuning")
    parser.add_argument("--lora_rank", type=int, default=16,
                       help="LoRA rank (default: 16, typical range: 8-64)")
    parser.add_argument("--lora_alpha", type=int, default=None,
                       help="LoRA alpha scaling factor (default: same as lora_rank)")
    
    parser.add_argument("--pool_features", action="store_true", default=True,
                       help="Pool language/vision features to fixed-size vectors")
    parser.add_argument("--pool_method", type=str, default="mean", choices=["mean", "max"],
                       help="Pooling method")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Number of gradient accumulation steps (effective batch size = batch_size * gradient_accumulation_steps)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    parser.add_argument("--max_episodes", type=int, default=None,
                       help="Maximum number of episodes to use (for quick testing/debugging, None = use all)")
    parser.add_argument("--max_transitions", type=int, default=None,
                       help="Maximum number of transitions to use (for quick testing/debugging, None = use all)")
    
    # Learning rates
    parser.add_argument("--vla_lr", type=float, default=1e-5,
                       help="OpenVLA (actor) learning rate")
    parser.add_argument("--critic_lr", type=float, default=3e-4,
                       help="Critic learning rate")
    parser.add_argument("--reward_lr", type=float, default=1e-4,
                       help="Reward network learning rate (if training reward network)")
    parser.add_argument("--reward_coef", type=float, default=1.0,
                       help="Reward network loss coefficient")
    parser.add_argument("--train_reward_network", action="store_true", default=False,
                       help="Train reward network on ground truth rewards from dataset")
    
    # RL hyperparameters
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                       help="GAE lambda parameter")
    parser.add_argument("--bc_coef", type=float, default=1.0,
                       help="Behavior cloning coefficient")
    parser.add_argument("--entropy_coef", type=float, default=0.0,
                       help="Entropy regularization coefficient")
    parser.add_argument("--critic_coef", type=float, default=1.0,
                       help="Critic loss coefficient")
    parser.add_argument("--advantage_clip", type=float, default=0.0,
                       help="Clip advantages to [-advantage_clip, advantage_clip] (0 = disabled, typical: 5.0-10.0)")
    parser.add_argument("--grad_clip_norm", type=float, default=0.0,
                       help="Gradient clipping norm (0 = disabled, typical: 0.5-2.0)")
    parser.add_argument("--critic_warmup_epochs", type=int, default=0,
                       help="Number of epochs to train only critic (0 = disabled)")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use (cuda:0, cuda:1, cpu, etc.)")
    parser.add_argument("--num_gpus", type=int, default=1,
                       help="Number of GPUs to use for training (1-4, uses DDP if > 1)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Logging and checkpoints
    parser.add_argument("--tag", type=str, default="offline_openvla_ac",
                       help="Experiment tag")
    parser.add_argument("--checkpoint_dir", type=str, default="log/checkpoints",
                       help="Checkpoint directory")
    parser.add_argument("--save_freq", type=int, default=10,
                       help="Save checkpoint every N epochs (0 to disable)")
    parser.add_argument("--save_final", action="store_true", default=True,
                       help="Save final checkpoint")
    
    # WandB
    parser.add_argument("--use_wandb", action="store_true", default=False,
                       help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="offline-openvla-ac",
                       help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                       help="WandB entity/team name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="WandB run name")
    
    return parser.parse_args()


def setup_ddp(rank, world_size):
    """Initialize DDP process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    """Clean up DDP process group."""
    dist.destroy_process_group()


def main():
    args = get_args()
    
    # Multi-GPU setup
    use_ddp = args.num_gpus > 1 and torch.cuda.is_available()
    
    if use_ddp:
        # Initialize DDP
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            # Launched via torchrun or torch.distributed.launch
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ.get('LOCAL_RANK', rank))
        else:
            # Manual setup (for testing)
            rank = 0
            world_size = args.num_gpus
            local_rank = 0
            setup_ddp(rank, world_size)
        
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        if rank == 0:
            print(f"Using DDP: rank={rank}, world_size={world_size}, local_rank={local_rank}, device={device}", flush=True)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print(f"Using single GPU: device={device}", flush=True)
    
    # Set random seed (same seed for all processes)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Initialize WandB (only on rank 0)
    if args.use_wandb and rank == 0:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or args.tag,
            config=vars(args),
        )
    
    # Load OpenVLA model
    if rank == 0:
        print("Loading OpenVLA model...", flush=True)
    vla_model, processor = load_openvla_model(args.openvla_checkpoint, device=device)
    
    # Freeze vision backbone and language encoder, keep only decoder trainable
    if rank == 0:
        print("Freezing vision backbone and language encoder, keeping decoder trainable...", flush=True)
    
    # Freeze vision backbone
    if hasattr(vla_model, 'vision_backbone'):
        for param in vla_model.vision_backbone.parameters():
            param.requires_grad = False
        if rank == 0:
            print("  ✓ Vision backbone frozen", flush=True)
    
    # Apply LoRA to decoder layers for parameter-efficient training
    use_lora = args.use_lora
    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else lora_rank
    
    if rank == 0:
        print(f"LoRA configuration: use_lora={use_lora}, rank={lora_rank}, alpha={lora_alpha}", flush=True)
        print(f"Model has llm_backbone: {hasattr(vla_model, 'llm_backbone')}", flush=True)
        print(f"Model has language_model: {hasattr(vla_model, 'language_model')}", flush=True)
    
    # OpenVLA loaded from HuggingFace uses 'language_model' instead of 'llm_backbone'
    if use_lora:
        llm_module = None
        if hasattr(vla_model, 'llm_backbone'):
            llm_module = vla_model.llm_backbone
        elif hasattr(vla_model, 'language_model'):
            llm_module = vla_model.language_model
        
        if llm_module is not None:
            if rank == 0:
                print(f"Applying LoRA to decoder layers (rank={lora_rank}, alpha={lora_alpha})...", flush=True)
            try:
                from peft import LoraConfig, get_peft_model, TaskType
                
                # Get the actual LLM model (might be nested)
                llm = llm_module.llm if hasattr(llm_module, 'llm') else llm_module
                
                # Debug: Print model structure to find correct module names
                if rank == 0:
                    print(f"  Debug: LLM type: {type(llm).__name__}", flush=True)
                    if hasattr(llm, 'model'):
                        print(f"  Debug: LLM has 'model' attribute: {type(llm.model).__name__}", flush=True)
                        if hasattr(llm.model, 'layers') and len(llm.model.layers) > 0:
                            first_layer = llm.model.layers[0]
                            print(f"  Debug: First layer modules: {list(first_layer.named_modules())[:10]}", flush=True)
                
                # LoRA configuration for decoder layers
                # Target modules for LLaMA-style transformers - try multiple naming conventions
                target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention projections
                    "gate_proj", "up_proj", "down_proj",     # Feed-forward projections
                ]
                
                lora_config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=0.0,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                )
                
                # Apply LoRA to the LLM module
                if hasattr(vla_model, 'llm_backbone'):
                    vla_model.llm_backbone.llm = get_peft_model(llm, lora_config)
                    lora_llm = vla_model.llm_backbone.llm
                elif hasattr(vla_model, 'language_model'):
                    vla_model.language_model = get_peft_model(llm, lora_config)
                    lora_llm = vla_model.language_model
                else:
                    raise ValueError("Could not find llm_backbone or language_model to apply LoRA")
                
                if rank == 0:
                    print(f"  ✓ LoRA applied to decoder layers", flush=True)
                
                # Count LoRA parameters
                lora_params = sum(p.numel() for p in lora_llm.parameters() if p.requires_grad)
                total_llm_params = sum(p.numel() for p in lora_llm.parameters())
                if rank == 0:
                    print(f"  ✓ LoRA trainable parameters: {lora_params:,} ({100*lora_params/total_llm_params:.4f}% of LLM)", flush=True)
                
                # Enable gradient checkpointing to save memory (if supported)
                try:
                    if hasattr(lora_llm, 'gradient_checkpointing_enable'):
                        lora_llm.gradient_checkpointing_enable()
                        if rank == 0:
                            print("  ✓ Gradient checkpointing enabled (saves memory)", flush=True)
                    elif hasattr(lora_llm, 'model') and hasattr(lora_llm.model, 'gradient_checkpointing_enable'):
                        lora_llm.model.gradient_checkpointing_enable()
                        if rank == 0:
                            print("  ✓ Gradient checkpointing enabled (saves memory)", flush=True)
                except Exception as e:
                    if rank == 0:
                        print(f"  ⚠ Could not enable gradient checkpointing: {e}", flush=True)
                
            except ImportError as e:
                if rank == 0:
                    print(f"  ⚠ Warning: peft library not found. Install with: pip install peft", flush=True)
                    print(f"  Error: {e}", flush=True)
                    print("  Falling back to full fine-tuning of decoder layers...", flush=True)
                use_lora = False
            except Exception as e:
                if rank == 0:
                    print(f"  ⚠ Warning: Failed to apply LoRA: {type(e).__name__}: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    print("  Falling back to full fine-tuning of decoder layers...", flush=True)
                use_lora = False
        else:
            if rank == 0:
                print("  ⚠ Warning: Could not find llm_backbone or language_model in model", flush=True)
                print("  Falling back to full fine-tuning of decoder layers...", flush=True)
            use_lora = False
    
    # If not using LoRA, use full fine-tuning of decoder layers
    if not use_lora:
        llm_module = None
        if hasattr(vla_model, 'llm_backbone'):
            llm_module = vla_model.llm_backbone
        elif hasattr(vla_model, 'language_model'):
            llm_module = vla_model.language_model
        
        if llm_module is not None:
            llm = llm_module.llm if hasattr(llm_module, 'llm') else llm_module
            
            # Get the actual transformer model (might be nested as llm.model)
            model = llm.model if hasattr(llm, 'model') else llm
            
            # Strategy: Freeze everything in llm_module first, then unfreeze only decoder layers
            # First, freeze all LLM parameters
            for param in llm_module.parameters():
                param.requires_grad = False
            
            # Now unfreeze only the decoder layers (transformer blocks)
            decoder_layers = None
            if hasattr(model, 'layers'):
                decoder_layers = model.layers
            elif hasattr(llm, 'layers'):
                decoder_layers = llm.layers
            
            if decoder_layers is not None:
                # Unfreeze decoder layers only
                for param in decoder_layers.parameters():
                    param.requires_grad = True
                trainable_params = sum(p.numel() for p in decoder_layers.parameters())
                if rank == 0:
                    print(f"  ✓ Decoder layers trainable ({trainable_params:,} parameters)", flush=True)
                
                # Count what's frozen
                frozen_embeds = 0
                if hasattr(model, 'embed_tokens'):
                    frozen_embeds += sum(p.numel() for p in model.embed_tokens.parameters())
                if hasattr(llm, 'lm_head'):
                    frozen_embeds += sum(p.numel() for p in llm.lm_head.parameters())
                if hasattr(model, 'norm'):
                    frozen_embeds += sum(p.numel() for p in model.norm.parameters())
                if rank == 0:
                    print(f"  ✓ Language encoder frozen (embeddings, norm, lm_head: {frozen_embeds:,} parameters)", flush=True)
            else:
                if rank == 0:
                    print("  ⚠ Warning: Could not find decoder layers, all LLM parameters will be trainable", flush=True)
                # If we can't find layers, unfreeze everything (fallback)
                for param in llm_module.parameters():
                    param.requires_grad = True
        else:
            if rank == 0:
                print("  ⚠ Warning: Could not find llm_backbone or language_model in model", flush=True)
    
    # Freeze projector if exists (vision-language adapter)
    if hasattr(vla_model, 'projector'):
        for param in vla_model.projector.parameters():
            param.requires_grad = False
        if rank == 0:
            print("  ✓ Projector frozen", flush=True)
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in vla_model.parameters())
    trainable_params = sum(p.numel() for p in vla_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    if rank == 0:
        print(f"\nParameter Summary:", flush=True)
        print(f"  Total parameters: {total_params:,}", flush=True)
        print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)", flush=True)
        print(f"  Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)", flush=True)
    
    vla_model.train()  # Set to training mode
    
    # Get action dimension
    action_dim = vla_model.get_action_dim(args.unnorm_key)
    if rank == 0:
        print(f"Action dimension: {action_dim}", flush=True)
    
    # Load dataset
    if rank == 0:
        print("Loading dataset...", flush=True)
    dataset = OfflineOpenVLADataset(
        features_cache_path=args.features_cache_path,
        rlds_dataset_path=args.rlds_dataset_path,
        pool_features=args.pool_features,
        pool_method=args.pool_method,
        max_episodes=args.max_episodes,
        max_transitions=args.max_transitions,
    )
    
    # Get state dimension from first sample
    sample = dataset[0]
    state_dim = sample["state"].shape[0]
    if rank == 0:
        print(f"State dimension: {state_dim}", flush=True)
    
    # Get feature dimensions for reward network
    lang_feat_dim = sample["language_features"].shape[-1]
    vis_feat_dim = sample["vision_features"].shape[-1]
    if rank == 0:
        print(f"Language feature dim: {lang_feat_dim}, Vision feature dim: {vis_feat_dim}", flush=True)
    
    # Create models
    if rank == 0:
        print("Creating models...", flush=True)
    critic = ValueCritic(state_dim=state_dim, hidden_dims=tuple(args.critic_hidden_dims)).to(device)
    
    reward_model = OpenVLARewardNet(
        language_feature_dim=lang_feat_dim,
        vision_feature_dim=vis_feat_dim,
        act_dim=action_dim,
        hidden_dim=args.reward_hidden_dim,
        num_layers=args.reward_num_layers,
    ).to(device)
    
    # Load checkpoints if provided
    if args.reward_model_checkpoint and os.path.exists(args.reward_model_checkpoint):
        if rank == 0:
            print(f"Loading reward model from checkpoint: {args.reward_model_checkpoint}", flush=True)
        checkpoint = torch.load(args.reward_model_checkpoint, map_location=device)
        
        # Handle different checkpoint formats
        reward_state_dict = None
        if 'reward_model_state_dict' in checkpoint:
            reward_state_dict = checkpoint['reward_model_state_dict']
        elif 'state_dict' in checkpoint:
            reward_state_dict = checkpoint['state_dict']
        elif 'networks' in checkpoint and 'reward_net' in checkpoint['networks']:
            # Checkpoint from RRD training (nested structure)
            reward_state_dict = checkpoint['networks']['reward_net']
        elif isinstance(checkpoint, dict) and any(k.startswith('language_pool') or k.startswith('mlp') for k in checkpoint.keys()):
            # Direct state dict
            reward_state_dict = checkpoint
        else:
            # Try to find reward_net in nested structures
            if 'networks' in checkpoint:
                if 'reward_net' in checkpoint['networks']:
                    reward_state_dict = checkpoint['networks']['reward_net']
                else:
                    # Try first key in networks
                    first_key = list(checkpoint['networks'].keys())[0]
                    reward_state_dict = checkpoint['networks'][first_key]
            else:
                reward_state_dict = checkpoint
        
        if reward_state_dict is not None:
            reward_model.load_state_dict(reward_state_dict)
            if rank == 0:
                print("✓ Reward model loaded from checkpoint", flush=True)
        else:
            if rank == 0:
                print("⚠ Warning: Could not find reward model state dict in checkpoint, initializing from scratch", flush=True)
    else:
        if rank == 0:
            if args.reward_model_checkpoint:
                print(f"Warning: Reward model checkpoint not found: {args.reward_model_checkpoint}, initializing from scratch", flush=True)
            else:
                print("Initializing reward model from scratch (no checkpoint provided)", flush=True)
    
    if args.critic_checkpoint and os.path.exists(args.critic_checkpoint):
        if rank == 0:
            print(f"Loading critic from checkpoint: {args.critic_checkpoint}", flush=True)
        checkpoint = torch.load(args.critic_checkpoint, map_location=device)
        if 'critic_state_dict' in checkpoint:
            critic.load_state_dict(checkpoint['critic_state_dict'])
            if rank == 0:
                print("✓ Critic loaded from checkpoint", flush=True)
        elif 'state_dict' in checkpoint:
            critic.load_state_dict(checkpoint['state_dict'])
            if rank == 0:
                print("✓ Critic loaded from checkpoint", flush=True)
        else:
            critic.load_state_dict(checkpoint)
            if rank == 0:
                print("✓ Critic loaded from checkpoint", flush=True)
    else:
        if rank == 0:
            if args.critic_checkpoint:
                print(f"Warning: Critic checkpoint not found: {args.critic_checkpoint}, initializing from scratch", flush=True)
            else:
                print("Initializing critic from scratch (no checkpoint provided)", flush=True)
    
    # Create optimizers
    vla_optimizer = torch.optim.AdamW(vla_model.parameters(), lr=args.vla_lr)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr)
    reward_optimizer = torch.optim.AdamW(reward_model.parameters(), lr=args.reward_lr)
    
    # Create data loader with DistributedSampler if using DDP
    if use_ddp:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
        shuffle = False  # Sampler handles shuffling
    else:
        sampler = None
        shuffle = True
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Training loop
    if rank == 0:
        print(f"\nStarting training for {args.epochs} epochs...", flush=True)
        print(f"Dataset size: {len(dataset)}, Batches per epoch: {len(dataloader)}", flush=True)
        if args.save_freq > 0:
            print(f"Checkpoints will be saved every {args.save_freq} epochs (at epochs: {', '.join([str(i) for i in range(args.save_freq, args.epochs + 1, args.save_freq)])})", flush=True)
        else:
            print("Checkpoint saving is disabled (save_freq=0)", flush=True)
        print("=" * 80, flush=True)
    
    global_step = 0
    for epoch in range(args.epochs):
        # Check if we're in critic warmup period
        skip_actor_update = args.critic_warmup_epochs > 0 and epoch < args.critic_warmup_epochs
        if rank == 0 and skip_actor_update:
            print(f"\n[Warmup] Epoch {epoch+1}/{args.epochs}: Training critic only (warmup for {args.critic_warmup_epochs} epochs)", flush=True)
        
        # Set epoch for DistributedSampler
        if use_ddp:
            sampler.set_epoch(epoch)
        epoch_losses = {
            "loss_total": 0.0,
            "loss_actor": 0.0,
            "loss_bc": 0.0,
            "loss_critic": 0.0,
            "entropy": 0.0,
        }
        if args.train_reward_network:
            epoch_losses["loss_reward"] = 0.0
        
        # Only show progress bar on rank 0 to avoid duplicate prints in DDP
        if rank == 0:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        else:
            pbar = dataloader  # Use dataloader directly without tqdm on non-main processes
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Add ground truth rewards to batch if training reward network
            if args.train_reward_network:
                # Use dataset rewards as ground truth
                batch["reward_gt"] = batch["reward"]
            
            # Determine if this is the first step in accumulation or last step
            is_accumulation_start = (batch_idx % args.gradient_accumulation_steps == 0)
            is_accumulation_end = ((batch_idx + 1) % args.gradient_accumulation_steps == 0) or (batch_idx == len(dataloader) - 1)
            
            # Training step with gradient accumulation
            losses = training_step(
                vla_model=vla_model,
                critic=critic,
                reward_model=reward_model,
                batch=batch,
                vla_optimizer=vla_optimizer,
                critic_optimizer=critic_optimizer,
                reward_optimizer=reward_optimizer if args.train_reward_network else None,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                bc_coef=args.bc_coef,
                entropy_coef=args.entropy_coef,
                critic_coef=args.critic_coef,
                reward_coef=args.reward_coef,
                train_reward_network=args.train_reward_network,
                unnorm_key=args.unnorm_key,
                use_dense_rewards=True,
                skip_actor_update=skip_actor_update,
                grad_clip_norm=args.grad_clip_norm,
                advantage_clip=args.advantage_clip,
                zero_grad=is_accumulation_start,  # Only zero at start of accumulation
                step_optimizers=is_accumulation_end,  # Only step at end of accumulation
                accumulation_steps=args.gradient_accumulation_steps,
            )
            
            # Accumulate losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key]
            
            # Update progress bar (only on rank 0)
            if rank == 0 and hasattr(pbar, 'set_postfix'):  # Check if pbar is tqdm
                pbar.set_postfix({
                    "loss": f"{losses['loss_total']:.4f}",
                    "actor": f"{losses['loss_actor']:.4f}",
                    "critic": f"{losses['loss_critic']:.4f}",
                })
            
            # Log to WandB (only on rank 0) - log every batch
            if args.use_wandb and rank == 0:
                log_dict = {f"train/{k}": v for k, v in losses.items()}
                log_dict["train/global_step"] = global_step
                log_dict["train/epoch"] = epoch + 1
                log_dict["train/batch"] = batch_idx + 1
                wandb.log(log_dict, step=global_step)
            
            global_step += 1
        
        # Average losses over epoch (gather from all processes if using DDP)
        if use_ddp:
            # Gather losses from all processes
            gathered_losses = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_losses, epoch_losses)
            
            # Average across all processes
            all_losses = {}
            for proc_losses in gathered_losses:
                for k, v in proc_losses.items():
                    if k not in all_losses:
                        all_losses[k] = []
                    all_losses[k].append(v)
            # Average across processes
            for key in epoch_losses:
                epoch_losses[key] = sum(all_losses[key]) / len(all_losses[key])
        else:
            num_batches = len(dataloader)
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
        
        # Print epoch summary (only on rank 0)
        if rank == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs} Summary:", flush=True)
            for key, value in epoch_losses.items():
                print(f"  {key}: {value:.6f}", flush=True)
        
        # Log epoch averages to WandB (only on rank 0) - log after every epoch
        if args.use_wandb and rank == 0:
            log_dict = {f"epoch/{k}": v for k, v in epoch_losses.items()}
            log_dict["epoch/epoch"] = epoch + 1
            log_dict["epoch/global_step"] = global_step
            wandb.log(log_dict, step=global_step)
        
        # Save checkpoint (only on rank 0)
        if rank == 0 and args.save_freq > 0 and (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.tag}_epoch_{epoch+1}.pt")
            # Get state dicts (unwrap DDP if needed)
            if use_ddp:
                vla_state_dict = vla_model.module.state_dict() if hasattr(vla_model, 'module') else vla_model.state_dict()
                critic_state_dict = critic.module.state_dict() if hasattr(critic, 'module') else critic.state_dict()
                reward_state_dict = reward_model.module.state_dict() if hasattr(reward_model, 'module') else reward_model.state_dict()
            else:
                vla_state_dict = vla_model.state_dict()
                critic_state_dict = critic.state_dict()
                reward_state_dict = reward_model.state_dict()
            
            torch.save({
                'epoch': epoch + 1,
                'vla_model_state_dict': vla_state_dict,
                'critic_state_dict': critic_state_dict,
                'reward_model_state_dict': reward_state_dict,
                'vla_optimizer_state_dict': vla_optimizer.state_dict(),
                'critic_optimizer_state_dict': critic_optimizer.state_dict(),
                'reward_optimizer_state_dict': reward_optimizer.state_dict(),
                'args': vars(args),
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}", flush=True)
    
    # Save final checkpoint (only on rank 0)
    if rank == 0 and args.save_final:
        checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.tag}_final.pt")
        # Get state dicts (unwrap DDP if needed)
        if use_ddp:
            vla_state_dict = vla_model.module.state_dict() if hasattr(vla_model, 'module') else vla_model.state_dict()
            critic_state_dict = critic.module.state_dict() if hasattr(critic, 'module') else critic.state_dict()
            reward_state_dict = reward_model.module.state_dict() if hasattr(reward_model, 'module') else reward_model.state_dict()
        else:
            vla_state_dict = vla_model.state_dict()
            critic_state_dict = critic.state_dict()
            reward_state_dict = reward_model.state_dict()
        
        torch.save({
            'epoch': args.epochs,
            'vla_model_state_dict': vla_state_dict,
            'critic_state_dict': critic_state_dict,
            'reward_model_state_dict': reward_state_dict,
            'vla_optimizer_state_dict': vla_optimizer.state_dict(),
            'critic_optimizer_state_dict': critic_optimizer.state_dict(),
            'reward_optimizer_state_dict': reward_optimizer.state_dict(),
            'args': vars(args),
        }, checkpoint_path)
        print(f"Saved final checkpoint: {checkpoint_path}", flush=True)
    
    # Cleanup DDP
    if use_ddp and 'RANK' not in os.environ:
        cleanup_ddp()
    
    if rank == 0:
        print("\nTraining completed!", flush=True)
        if args.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()




