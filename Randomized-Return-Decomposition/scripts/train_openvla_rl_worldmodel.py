#!/usr/bin/env python3
"""
RL Fine-tuning of OpenVLA using World Model as Environment

This script does online RL training where:
- World Model acts as the environment (generates next states)
- OpenVLA acts as the actor (generates actions)
- Critic estimates values
- Reward model computes rewards
- PPO updates OpenVLA policy
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from collections import deque

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Add world model eval path
world_model_path = os.path.join(os.path.dirname(project_root), "world-model-eval-babel")
if os.path.exists(world_model_path) and world_model_path not in sys.path:
    sys.path.insert(0, world_model_path)

from offline_openvla_ac import ValueCritic, compute_gae_advantages_and_returns
from algorithm.openvla_reward_net import OpenVLAFeatureExtractor, OpenVLARewardNet


def load_openvla_model(checkpoint_path, device="cuda:0"):
    """Load OpenVLA model and processor."""
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
        
        # Load model
        vla_model = AutoModelForVision2Seq.from_pretrained(
            checkpoint_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
        ).to(device)
        
        processor = AutoProcessor.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        
        return vla_model, processor
    except Exception as e:
        print(f"Error loading OpenVLA model: {e}")
        raise


def load_world_model(checkpoint_path, device="cuda:0"):
    """Load world model."""
    from world_model import WorldModel
    
    world_model = WorldModel(
        checkpoint_path=checkpoint_path,
        use_pixel_rope=False,
        default_cfg=1.0
    )
    world_model.device = device
    return world_model


def pad_actions_to_10dim(actions, source_dim=7):
    """Pad 7-dim actions to 10-dim for world model."""
    if isinstance(actions, torch.Tensor):
        padding = torch.zeros(*actions.shape[:-1], 10 - source_dim, 
                             device=actions.device, dtype=actions.dtype)
        return torch.cat([actions, padding], dim=-1)
    else:
        padding = np.zeros((*actions.shape[:-1], 10 - source_dim), dtype=actions.dtype)
        return np.concatenate([actions, padding], axis=-1)


class WorldModelEnv:
    """Wrapper for world model as RL environment."""
    
    def __init__(self, world_model, initial_states, task_descriptions, device="cuda:0"):
        self.world_model = world_model
        self.initial_states = initial_states
        self.task_descriptions = task_descriptions
        self.device = device
        self.num_envs = len(initial_states)
        
        # Current states for each env
        self.current_frames = [None] * self.num_envs
        self.current_tasks = [""] * self.num_envs
        self.dones = np.ones(self.num_envs, dtype=bool)
        
    def reset(self, env_ids=None):
        """Reset environments."""
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        
        observations = []
        for env_id in env_ids:
            # Sample random initial state
            state_idx = np.random.randint(0, len(self.initial_states))
            initial_frame = self.initial_states[state_idx]
            
            # Convert to tensor format expected by world model
            if isinstance(initial_frame, np.ndarray):
                initial_frame_tensor = torch.from_numpy(initial_frame).float() / 255.0
            else:
                initial_frame_tensor = initial_frame
            
            # Reset world model
            self.world_model.reset(initial_frame_tensor)
            
            # Store current state
            self.current_frames[env_id] = initial_frame
            self.current_tasks[env_id] = self.task_descriptions[state_idx] if state_idx < len(self.task_descriptions) else ""
            self.dones[env_id] = False
            
            observations.append(initial_frame)
        
        return observations, self.current_tasks
    
    def step(self, actions):
        """
        Step environment with actions.
        
        Args:
            actions: List of actions, each [7] or tensor
            
        Returns:
            next_obs: List of next observations
            rewards: List of rewards
            dones: Array of done flags
            infos: List of info dicts
        """
        next_obs = []
        rewards = []
        dones = []
        infos = []
        
        for env_id, action in enumerate(actions):
            if self.dones[env_id]:
                # Already done, return same state
                next_obs.append(self.current_frames[env_id])
                rewards.append(0.0)
                dones.append(True)
                infos.append({})
                continue
            
            # Convert action to tensor if needed
            if isinstance(action, np.ndarray):
                action_tensor = torch.from_numpy(action).float().to(self.device)
            else:
                action_tensor = action.to(self.device)
            
            # Pad to 10-dim
            action_10d = pad_actions_to_10dim(action_tensor.unsqueeze(0), source_dim=7)
            
            # Generate next frame
            try:
                for idx, next_frame_tensor in self.world_model.generate_chunk(action_10d):
                    # Convert tensor to numpy
                    next_frame = (next_frame_tensor[0].cpu().numpy() * 255.0).astype(np.uint8)
                    next_obs.append(next_frame)
                    
                    # Update current frame
                    self.current_frames[env_id] = next_frame
                    
                    # Compute reward (placeholder - will use reward model)
                    rewards.append(0.0)  # Will compute with reward model
                    
                    # Check termination (placeholder)
                    done = False
                    self.dones[env_id] = done
                    dones.append(done)
                    
                    infos.append({})
                    break
            except Exception as e:
                print(f"Error generating frame for env {env_id}: {e}")
                # Fallback: return current state
                next_obs.append(self.current_frames[env_id])
                rewards.append(0.0)
                dones.append(True)
                self.dones[env_id] = True
                infos.append({"error": str(e)})
        
        return next_obs, rewards, np.array(dones), infos


def compute_ppo_loss(logprobs, old_logprobs, advantages, clip_ratio=0.2):
    """Compute PPO clipped loss."""
    ratio = torch.exp(logprobs - old_logprobs)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
    return loss.mean()


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RL fine-tuning of OpenVLA with world model")
    
    # Model paths
    parser.add_argument("--openvla_checkpoint", type=str, required=True,
                       help="Path to OpenVLA checkpoint")
    parser.add_argument("--world_model_checkpoint", type=str, required=True,
                       help="Path to world model checkpoint")
    parser.add_argument("--critic_checkpoint", type=str, default=None,
                       help="Path to critic checkpoint (optional)")
    parser.add_argument("--reward_model_checkpoint", type=str, default=None,
                       help="Path to reward model checkpoint (optional)")
    
    # Initial states
    parser.add_argument("--initial_states_path", type=str, required=True,
                       help="Path to initial states (images or dataset)")
    parser.add_argument("--task_descriptions", type=str, nargs="+", default=["pick up the red cup"],
                       help="Task descriptions for each environment")
    
    # Training hyperparameters
    parser.add_argument("--num_envs", type=int, default=8,
                       help="Number of parallel environments")
    parser.add_argument("--max_episode_steps", type=int, default=512,
                       help="Maximum episode length")
    parser.add_argument("--rollout_steps", type=int, default=2048,
                       help="Number of steps to collect before update")
    parser.add_argument("--num_updates", type=int, default=1000,
                       help="Number of PPO update iterations")
    parser.add_argument("--ppo_epochs", type=int, default=4,
                       help="Number of PPO epochs per update")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for PPO updates")
    
    # Learning rates
    parser.add_argument("--vla_lr", type=float, default=1e-5,
                       help="OpenVLA learning rate")
    parser.add_argument("--critic_lr", type=float, default=3e-4,
                       help="Critic learning rate")
    
    # PPO hyperparameters
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                       help="GAE lambda")
    parser.add_argument("--clip_ratio", type=float, default=0.2,
                       help="PPO clip ratio")
    parser.add_argument("--value_coef", type=float, default=0.5,
                       help="Value loss coefficient")
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                       help="Entropy coefficient")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Logging
    parser.add_argument("--tag", type=str, default="openvla_rl_worldmodel",
                       help="Experiment tag")
    parser.add_argument("--checkpoint_dir", type=str, default="log/checkpoints",
                       help="Checkpoint directory")
    parser.add_argument("--save_freq", type=int, default=50,
                       help="Save checkpoint every N updates")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="openvla-rl-worldmodel",
                       help="WandB project name")
    
    return parser.parse_args()


def load_initial_states(path):
    """Load initial states from path."""
    import cv2
    from pathlib import Path
    
    path = Path(path)
    initial_states = []
    
    if path.is_dir():
        # Load all images from directory
        for img_file in sorted(path.glob("*.jpg")) + sorted(path.glob("*.png")):
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            initial_states.append(img)
    elif path.suffix == ".npy":
        # Load numpy array
        states = np.load(path)
        initial_states = [states[i] for i in range(len(states))]
    else:
        raise ValueError(f"Unknown initial states format: {path}")
    
    print(f"Loaded {len(initial_states)} initial states")
    return initial_states


def main():
    args = get_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize WandB
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.tag,
            config=vars(args),
        )
    
    # Load models
    print("Loading models...")
    vla_model, processor = load_openvla_model(args.openvla_checkpoint, device=device)
    vla_model.train()
    
    world_model = load_world_model(args.world_model_checkpoint, device=device)
    
    # Load initial states
    initial_states = load_initial_states(args.initial_states_path)
    task_descriptions = args.task_descriptions * (len(initial_states) // len(args.task_descriptions) + 1)
    task_descriptions = task_descriptions[:len(initial_states)]
    
    # Create environment
    env = WorldModelEnv(world_model, initial_states, task_descriptions, device=device)
    
    # Create feature extractor for reward model
    feature_extractor = OpenVLAFeatureExtractor(vla_model, processor, device=device)
    
    # Get feature dimensions
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    lang_feat, vis_feat = feature_extractor.extract_features([dummy_img], ["test"])
    lang_feat_dim = lang_feat.shape[-1]
    vis_feat_dim = vis_feat.shape[-1]
    action_dim = 7
    
    # Create critic
    state_dim = lang_feat_dim + vis_feat_dim  # Assuming mean pooling
    critic = ValueCritic(state_dim=state_dim, hidden_dims=(512, 512)).to(device)
    
    # Create reward model
    reward_model = OpenVLARewardNet(
        language_feature_dim=lang_feat_dim,
        vision_feature_dim=vis_feat_dim,
        act_dim=action_dim,
        hidden_dim=512,
        num_layers=3,
    ).to(device)
    
    # Load checkpoints if provided
    if args.critic_checkpoint and os.path.exists(args.critic_checkpoint):
        critic.load_state_dict(torch.load(args.critic_checkpoint))
        print(f"Loaded critic from {args.critic_checkpoint}")
    
    if args.reward_model_checkpoint and os.path.exists(args.reward_model_checkpoint):
        reward_model.load_state_dict(torch.load(args.reward_model_checkpoint))
        print(f"Loaded reward model from {args.reward_model_checkpoint}")
    
    # Create optimizers
    vla_optimizer = torch.optim.AdamW(vla_model.parameters(), lr=args.vla_lr)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Training loop
    print(f"\nStarting RL training for {args.num_updates} updates...")
    
    global_step = 0
    
    for update in range(args.num_updates):
        # Collect rollouts
        print(f"\nUpdate {update+1}/{args.num_updates}: Collecting rollouts...")
        
        observations = []
        actions = []
        rewards = []
        dones = []
        logprobs = []
        values = []
        task_descs = []
        
        # Reset environments
        obs_batch, task_batch = env.reset()
        
        for step in range(args.rollout_steps):
            # Get actions from OpenVLA
            obs_tensor = torch.from_numpy(np.stack(obs_batch)).float() / 255.0
            obs_tensor = obs_tensor.permute(0, 3, 1, 2).to(device)  # [B, C, H, W]
            
            # Prepare inputs for OpenVLA
            inputs = processor(images=[obs for obs in obs_batch], 
                             text=task_batch,
                             return_tensors="pt",
                             padding=True).to(device)
            
            # Get action predictions
            with torch.no_grad():
                outputs = vla_model(**inputs)
                # Extract action from model output
                # This depends on your OpenVLA model structure
                # Placeholder: assume model returns actions
                if hasattr(outputs, 'predicted_actions'):
                    action_tensor = outputs.predicted_actions
                else:
                    # Fallback: sample from output distribution
                    # This is a placeholder - adjust based on your model
                    action_tensor = torch.randn(len(obs_batch), 7, device=device)
            
            # Clip actions to valid range
            action_tensor = torch.clamp(action_tensor, -1.0, 1.0)
            actions_np = action_tensor.cpu().numpy()
            
            # Get log probabilities (for PPO)
            with torch.no_grad():
                # Compute logprob of actions
                # This is simplified - adjust based on your model
                logprob = torch.randn(len(obs_batch), device=device)  # Placeholder
            
            # Extract features for critic and reward
            with torch.no_grad():
                lang_feat, vis_feat = feature_extractor.extract_features(obs_batch, task_batch)
                state_features = torch.cat([lang_feat.mean(dim=1), vis_feat.mean(dim=1)], dim=1)
                value = critic(state_features).squeeze(-1)
            
            # Store data
            observations.extend(obs_batch)
            actions.append(actions_np)
            task_descs.extend(task_batch)
            logprobs.append(logprob.cpu())
            values.append(value.cpu())
            
            # Step environment
            next_obs, step_rewards, step_dones, infos = env.step(actions_np.tolist())
            
            # Compute rewards using reward model
            with torch.no_grad():
                next_lang_feat, next_vis_feat = feature_extractor.extract_features(
                    next_obs, task_batch
                )
                # Compute dense rewards
                reward_tensor = reward_model(
                    language_features=lang_feat,
                    vision_features=vis_feat,
                    acts=action_tensor,
                    language_features_next=next_lang_feat,
                    vision_features_next=next_vis_feat,
                ).squeeze(-1)
                step_rewards = reward_tensor.cpu().numpy()
            
            rewards.append(step_rewards)
            dones.append(step_dones)
            
            obs_batch = next_obs
            
            global_step += args.num_envs
        
        # Convert to tensors
        observations = np.array(observations)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        dones = np.concatenate(dones, axis=0)
        logprobs = torch.cat(logprobs, dim=0)
        values = torch.cat(values, dim=0)
        
        # Compute advantages and returns
        print("Computing advantages...")
        returns, advantages = compute_gae_advantages_and_returns(
            rewards=torch.from_numpy(rewards).float(),
            values=values,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            dones=torch.from_numpy(dones),
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO updates
        print(f"Performing {args.ppo_epochs} PPO epochs...")
        indices = np.arange(len(observations))
        
        for epoch in range(args.ppo_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, len(indices), args.batch_size):
                end = start + args.batch_size
                batch_idx = indices[start:end]
                
                # Get batch data
                batch_obs = observations[batch_idx]
                batch_actions = torch.from_numpy(actions[batch_idx]).float().to(device)
                batch_old_logprobs = logprobs[batch_idx].to(device)
                batch_advantages = advantages[batch_idx].to(device)
                batch_returns = returns[batch_idx].to(device)
                batch_values = values[batch_idx].to(device)
                batch_tasks = [task_descs[i] for i in batch_idx]
                
                # Get new logprobs and values
                inputs = processor(images=[batch_obs[i] for i in range(len(batch_obs))],
                                 text=batch_tasks,
                                 return_tensors="pt",
                                 padding=True).to(device)
                
                outputs = vla_model(**inputs)
                # Get new logprob (adjust based on your model)
                new_logprob = torch.randn(len(batch_idx), device=device)  # Placeholder
                
                # Extract features for critic
                lang_feat, vis_feat = feature_extractor.extract_features(
                    [batch_obs[i] for i in range(len(batch_obs))], batch_tasks
                )
                state_features = torch.cat([lang_feat.mean(dim=1), vis_feat.mean(dim=1)], dim=1)
                new_values = critic(state_features).squeeze(-1)
                
                # PPO loss
                actor_loss = compute_ppo_loss(new_logprob, batch_old_logprobs, batch_advantages, args.clip_ratio)
                
                # Critic loss
                value_loss = F.mse_loss(new_values, batch_returns)
                
                # Entropy (placeholder)
                entropy = torch.tensor(0.0, device=device)
                
                # Total loss
                total_loss = actor_loss + args.value_coef * value_loss - args.entropy_coef * entropy
                
                # Update
                vla_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(vla_model.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                vla_optimizer.step()
                critic_optimizer.step()
                
                # Log
                if args.use_wandb:
                    wandb.log({
                        "train/actor_loss": actor_loss.item(),
                        "train/value_loss": value_loss.item(),
                        "train/total_loss": total_loss.item(),
                        "train/mean_reward": rewards.mean(),
                        "train/mean_return": returns.mean(),
                        "train/mean_advantage": advantages.mean(),
                        "train/global_step": global_step,
                    })
        
        # Save checkpoint
        if (update + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.tag}_update_{update+1}.pt")
            torch.save({
                'update': update + 1,
                'vla_model_state_dict': vla_model.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'reward_model_state_dict': reward_model.state_dict(),
                'vla_optimizer_state_dict': vla_optimizer.state_dict(),
                'critic_optimizer_state_dict': critic_optimizer.state_dict(),
                'args': vars(args),
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    print("\nTraining completed!")
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()





