"""
Evaluation function to plot GT vs predicted rewards per step
"""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import io
import os
import cv2
from PIL import Image


def evaluate_reward_predictions(agent, buffer, num_trajectories=5, device='cuda', epoch=None):
    """
    Evaluate reward network by plotting GT vs predicted rewards for trajectories.
    
    Args:
        agent: The RRD agent with reward network
        buffer: The replay buffer containing trajectories
        num_trajectories: Number of trajectories to evaluate
        device: Device to run inference on
        epoch: Current epoch number (for video naming)
    
    Returns:
        Tuple of (list of matplotlib figures, list of video paths) - one per trajectory
    """
    agent.reward_net.eval()
    
    # Get the actual device the reward network is on
    # This ensures we use the same device as the model
    model_device = next(agent.reward_net.parameters()).device
    if device != str(model_device):
        # Use the model's device instead of the passed device
        device = model_device
    
    figures = []
    video_paths = []
    
    # Sample random trajectories
    if len(buffer.ep) < num_trajectories:
        traj_indices = list(range(len(buffer.ep)))
    else:
        traj_indices = np.random.choice(len(buffer.ep), num_trajectories, replace=False)
    
    with torch.no_grad():
        for traj_idx in traj_indices:
            trajectory = buffer.ep[traj_idx]
            
            if trajectory.length == 0:
                continue
            
            # Get GT rewards from trajectory
            gt_rewards = trajectory.ep['rews'].copy()  # Shape: (length,)
            if isinstance(gt_rewards, np.ndarray):
                gt_rewards = gt_rewards.flatten()
            else:
                gt_rewards = np.array(gt_rewards).flatten()
            
            # Check if rewards are episodic (sparse: mostly zeros, reward only at end)
            # Or if they're per-step (dense rewards throughout)
            unique_rewards = np.unique(gt_rewards)
            is_episodic = len(unique_rewards) == 1 or (len(unique_rewards) == 2 and 0.0 in unique_rewards and np.sum(gt_rewards > 0) <= 1)
            
            # Calculate cumulative GT rewards if per-step
            if is_episodic:
                # For episodic: use the episodic reward value for all steps
                episodic_reward = gt_rewards[-1] if len(gt_rewards) > 0 else 0.0
                gt_rewards_plot = np.full(len(gt_rewards), episodic_reward)
            else:
                # For per-step: use actual per-step rewards
                gt_rewards_plot = gt_rewards.copy()
            
            # Calculate cumulative GT rewards
            gt_cumulative = np.cumsum(gt_rewards)
            
            # Get observations, actions, and next observations
            obs_list = []
            obs_next_list = []
            acts_list = []
            
            for i in range(trajectory.length):
                obs_list.append(trajectory.ep['obs'][i])
                if i + 1 < trajectory.length:
                    obs_next_list.append(trajectory.ep['obs'][i + 1])
                else:
                    obs_next_list.append(trajectory.ep['obs'][i])  # Terminal state
                acts_list.append(trajectory.ep['acts'][i])
            
            # Convert to numpy arrays
            obs = np.array(obs_list)
            obs_next = np.array(obs_next_list)
            acts = np.array(acts_list)
            
            # Get features if available
            if hasattr(agent.args, 'use_openvla_features') and agent.args.use_openvla_features:
                if trajectory.features_extracted and trajectory.lang_feat is not None:
                    # Use pre-extracted features
                    lang_feat = [trajectory.lang_feat[i] for i in range(trajectory.length)]
                    vis_feat = [trajectory.vis_feat[i] for i in range(trajectory.length)]
                    lang_feat_next = [trajectory.lang_feat[min(i+1, trajectory.length-1)] for i in range(trajectory.length)]
                    vis_feat_next = [trajectory.vis_feat[min(i+1, trajectory.length-1)] for i in range(trajectory.length)]
                    
                    # Stack features - convert to tensors if needed and move to correct device
                    # Use model_device to ensure consistency with reward_net
                    def to_tensor_on_device(f, target_device):
                        """Convert to tensor and move to target device"""
                        if isinstance(f, torch.Tensor):
                            # If already a tensor, move to device
                            return f.to(target_device).float()
                        elif isinstance(f, np.ndarray):
                            # Convert numpy to tensor and move to device
                            return torch.from_numpy(f).to(target_device).float()
                        else:
                            # Convert to tensor and move to device
                            return torch.tensor(f, device=target_device).float()
                    
                    # Convert to tensors and move to model_device, then stack
                    lang_feat_tensor = torch.stack([to_tensor_on_device(f, model_device) for f in lang_feat])
                    vis_feat_tensor = torch.stack([to_tensor_on_device(f, model_device) for f in vis_feat])
                    lang_feat_next_tensor = torch.stack([to_tensor_on_device(f, model_device) for f in lang_feat_next])
                    vis_feat_next_tensor = torch.stack([to_tensor_on_device(f, model_device) for f in vis_feat_next])
                    acts_tensor = torch.from_numpy(acts).float().to(model_device)
                    
                    # Predict rewards
                    pred_rewards = agent.reward_net(
                        lang_feat_tensor,
                        vis_feat_tensor,
                        acts_tensor,
                        lang_feat_next_tensor,
                        vis_feat_next_tensor,
                    )
                else:
                    # Extract features on-the-fly (fallback)
                    # This is slower but works if features aren't pre-extracted
                    pred_rewards_list = []
                    for i in range(trajectory.length):
                        # Extract features for this step
                        lang_feat, vis_feat = agent.openvla_extractor.extract_features(
                            obs[i:i+1],
                            [trajectory.task_description]
                        )
                        lang_feat_next, vis_feat_next = agent.openvla_extractor.extract_features(
                            obs_next[i:i+1],
                            [trajectory.task_description]
                        )
                        
                        # Convert to tensors and move to model_device
                        lang_feat = lang_feat[0].to(model_device).float()
                        vis_feat = vis_feat[0].to(model_device).float()
                        lang_feat_next = lang_feat_next[0].to(model_device).float()
                        vis_feat_next = vis_feat_next[0].to(model_device).float()
                        acts_tensor = torch.from_numpy(acts[i:i+1]).float().to(model_device)
                        
                        # Predict reward
                        pred = agent.reward_net(
                            lang_feat.unsqueeze(0),
                            vis_feat.unsqueeze(0),
                            acts_tensor,
                            lang_feat_next.unsqueeze(0),
                            vis_feat_next.unsqueeze(0),
                        )
                        pred_rewards_list.append(pred.cpu().numpy())
                    
                    pred_rewards = np.array(pred_rewards_list)
                    pred_rewards = torch.from_numpy(pred_rewards).to(device)
            else:
                # Use standard observations
                obs_tensor = torch.from_numpy(obs).float().to(model_device)
                obs_next_tensor = torch.from_numpy(obs_next).float().to(model_device)
                acts_tensor = torch.from_numpy(acts).float().to(model_device)
                
                # Normalize observations
                obs_tensor = agent.normalize_obs(obs_tensor)
                obs_next_tensor = agent.normalize_obs(obs_next_tensor)
                
                # Predict rewards
                pred_rewards = agent.reward_net(obs_tensor, acts_tensor, obs_next_tensor)
            
            # Convert predictions to numpy
            pred_rewards_np = pred_rewards.cpu().numpy().flatten()
            
            # Calculate cumulative predicted rewards
            pred_cumulative = np.cumsum(pred_rewards_np)
            
            # Create plot with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            steps = np.arange(len(gt_rewards))
            
            # Top plot: Per-step rewards
            ax1.set_title(f'Per-Step Rewards - Trajectory {traj_idx+1} (Epoch Evaluation)', fontsize=14)
            
            # Plot GT per-step rewards
            if is_episodic:
                # If episodic, plot constant line for all steps
                ax1.plot(steps, gt_rewards_plot, 'b-', label='GT Reward (Episodic)', linewidth=2, alpha=0.7)
            else:
                # Plot per-step rewards
                ax1.plot(steps, gt_rewards_plot, 'b-', label='GT Reward (Per-Step)', linewidth=2, alpha=0.7, marker='o', markersize=4)
            
            # Plot predicted per-step rewards
            ax1.plot(steps, pred_rewards_np, 'r--', label='Predicted Reward (Per-Step)', linewidth=2, alpha=0.7, marker='s', markersize=4)
            
            ax1.set_ylabel('Reward', fontsize=12)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Bottom plot: Cumulative rewards
            ax2.set_title('Cumulative Rewards', fontsize=14)
            
            # Plot cumulative GT rewards
            ax2.plot(steps, gt_cumulative, 'b-', label='GT Cumulative Reward', linewidth=2, alpha=0.7, marker='o', markersize=4)
            
            # If episodic, also show the episodic reward value as a constant line
            if is_episodic:
                episodic_reward = gt_rewards[-1] if len(gt_rewards) > 0 else 0.0
                ax2.axhline(y=episodic_reward, color='b', linestyle=':', linewidth=2, alpha=0.5, label=f'GT Episodic Reward ({episodic_reward:.2f})')
            
            # Plot cumulative predicted rewards
            ax2.plot(steps, pred_cumulative, 'r--', label='Predicted Cumulative Reward', linewidth=2, alpha=0.7, marker='s', markersize=4)
            
            ax2.set_xlabel('Step', fontsize=12)
            ax2.set_ylabel('Cumulative Reward', fontsize=12)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            # Add task description if available (on top subplot)
            if hasattr(trajectory, 'task_description') and trajectory.task_description:
                ax1.text(0.02, 0.98, f'Task: {trajectory.task_description}', 
                        transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            figures.append(fig)
            
            # Create video from observations
            video_path = create_trajectory_video(trajectory, traj_idx, epoch=epoch, output_dir='log/eval_videos')
            video_paths.append(video_path)
    
    agent.reward_net.train()  # Set back to training mode
    return figures, video_paths


def create_trajectory_video(trajectory, traj_idx, fps=10, output_dir='log/eval_videos', epoch=None):
    """
    Create a video from trajectory observations.
    
    Args:
        trajectory: Trajectory object with observations
        traj_idx: Trajectory index (for naming)
        fps: Frames per second for video
        output_dir: Directory to save video
        epoch: Current epoch number (for filename)
    
    Returns:
        Path to saved video file
    """
    os.makedirs(output_dir, exist_ok=True)
    if epoch is not None:
        video_path = os.path.join(output_dir, f'epoch_{epoch}_traj_{traj_idx+1}.mp4')
    else:
        video_path = os.path.join(output_dir, f'traj_{traj_idx+1}.mp4')
    
    if trajectory.length == 0:
        return None
    
    # Get all observations
    obs_list = []
    for i in range(trajectory.length + 1):  # +1 for initial observation
        if i == 0:
            # Initial observation (first in ep['obs'])
            obs = trajectory.ep['obs'][0]
        else:
            # Subsequent observations
            obs = trajectory.ep['obs'][i]
        
        # Convert to numpy if needed
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        elif not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        
        # Handle different observation formats
        # Observations are normalized [0, 1], need to convert to [0, 255]
        if obs.dtype != np.uint8:
            if obs.max() <= 1.0:
                obs = (obs * 255).astype(np.uint8)
            else:
                obs = obs.astype(np.uint8)
        
        # Ensure shape is (H, W, C)
        if len(obs.shape) == 2:
            # Grayscale, convert to RGB
            obs = cv2.cvtColor(obs, cv2.COLOR_GRAY2RGB)
        elif len(obs.shape) == 3 and obs.shape[2] == 1:
            # Single channel, convert to RGB
            obs = np.repeat(obs, 3, axis=2)
        elif len(obs.shape) == 4:
            # Batch dimension, take first
            obs = obs[0]
        
        # Resize to standard size if needed (for consistent video)
        target_size = 256
        if obs.shape[0] != target_size or obs.shape[1] != target_size:
            obs = cv2.resize(obs, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
        # Ensure RGB format (not BGR)
        if obs.shape[2] == 3:
            # Check if it's BGR (OpenCV default) or RGB
            # For LIBERO, images should already be RGB, but let's be safe
            # We'll assume RGB since that's what the environment uses
            pass
        
        obs_list.append(obs)
    
    if len(obs_list) == 0:
        return None
    
    # Get video dimensions from first frame
    height, width = obs_list[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # Write frames
    for obs in obs_list:
        # Convert RGB to BGR for OpenCV VideoWriter
        obs_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        out.write(obs_bgr)
    
    out.release()
    return video_path


def log_figures_and_videos_to_wandb(figures, video_paths, summary_writer, epoch, step):
    """
    Log matplotlib figures and videos to WandB.
    
    Args:
        figures: List of matplotlib figures
        video_paths: List of video file paths
        summary_writer: SummaryWriterPyTorch object with wandb attribute
        epoch: Current epoch number
        step: Current step number
    """
    if summary_writer is None:
        return
    
    # Check if WandB is available
    if not hasattr(summary_writer, 'wandb') or summary_writer.wandb is None:
        return
    
    wandb = summary_writer.wandb
    if not hasattr(wandb, 'log'):
        return
    
    try:
        for i, fig in enumerate(figures):
            # Convert figure to image
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            
            # Prepare log dict
            log_dict = {
                f'eval/gt_vs_pred_rewards_traj_{i+1}': wandb.Image(img),
            }
            
            # Add video if available
            if i < len(video_paths) and video_paths[i] is not None and os.path.exists(video_paths[i]):
                log_dict[f'eval/video_traj_{i+1}'] = wandb.Video(video_paths[i])
            
            # Log to WandB
            wandb.log(log_dict, step=step)
            
            # Close figure to free memory
            plt.close(fig)
            buf.close()
    except Exception as e:
        print(f"Warning: Failed to log figures/videos to WandB: {e}")
        import traceback
        traceback.print_exc()
        # Close figures anyway
        for fig in figures:
            plt.close(fig)

