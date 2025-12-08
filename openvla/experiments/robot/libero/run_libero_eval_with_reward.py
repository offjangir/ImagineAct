"""
run_libero_eval_with_reward.py

Runs a model in a LIBERO simulation environment and evaluates reward predictions
using an OpenVLA reward network.

Usage:
    # OpenVLA with reward evaluation:
    python experiments/robot/libero/run_libero_eval_with_reward.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --reward_net_checkpoint <REWARD_NET_CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import torch
import tqdm
from libero.libero import benchmark
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

import wandb

# Ensure local openvla experiments module takes precedence over site-packages
openvla_path = str(Path(__file__).parent.parent.parent.parent.absolute())
print(f"openvla_path: {openvla_path}")
print(f"sys.path: {sys.path}")
if openvla_path in sys.path:
    sys.path.remove(openvla_path)
sys.path.insert(0, openvla_path)

# Add Randomized-Return-Decomposition to path
rrd_path = str(Path(__file__).parent.parent.parent.parent.parent / "Randomized-Return-Decomposition")
if rrd_path not in sys.path:
    sys.path.insert(0, rrd_path)

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

# Import reward network components
from algorithm.openvla_reward_net import OpenVLARewardNet, OpenVLAFeatureExtractor


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    use_film: bool = False                           # Whether to use FiLM conditioning
    num_images_in_input: int = 1                     # Number of images in model input
    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    use_proprio: bool = False                        # Whether to use proprioceptive information

    #################################################################################################################
    # Reward network parameters
    #################################################################################################################
    reward_net_checkpoint: Optional[str] = None       # Path to reward network checkpoint
    reward_hidden_dim: int = 512                      # Reward network hidden dimension
    reward_num_layers: int = 3                        # Reward network number of layers
    evaluate_rewards: bool = True                     # Whether to evaluate rewards during rollout

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 3                     # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    device: str = "cuda:2"                           # CUDA device to use (e.g., "cuda:0", "cuda:1", "cpu")
    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


def load_reward_network(cfg: GenerateConfig, vla_model, processor, device):
    """Load reward network from checkpoint."""
    if cfg.reward_net_checkpoint is None:
        print("[!] Warning: No reward network checkpoint provided. Reward evaluation will be skipped.")
        return None, None
    
    print(f"[*] Loading reward network from: {cfg.reward_net_checkpoint}")
    
    # Create feature extractor
    feature_extractor = OpenVLAFeatureExtractor(
        vla_model,
        processor,
        task_description="",  # Will be set per observation
        device=device,
    )
    
    # Get feature dimensions by extracting features from a dummy image
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    lang_feat, vis_feat = feature_extractor.extract_features([dummy_img], ["test"])
    lang_feat_dim = lang_feat.shape[-1]
    vis_feat_dim = vis_feat.shape[-1]
    action_dim = 7  # LIBERO action dimension
    
    print(f"[*] Language feature dim: {lang_feat_dim}, Vision feature dim: {vis_feat_dim}")
    
    # Create reward network
    reward_net = OpenVLARewardNet(
        language_feature_dim=lang_feat_dim,
        vision_feature_dim=vis_feat_dim,
        act_dim=action_dim,
        hidden_dim=cfg.reward_hidden_dim,
        num_layers=cfg.reward_num_layers,
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(cfg.reward_net_checkpoint, map_location=device)
    
    # Handle different checkpoint formats
    reward_state_dict = None
    if 'reward_model_state_dict' in checkpoint:
        reward_state_dict = checkpoint['reward_model_state_dict']
    elif 'state_dict' in checkpoint:
        reward_state_dict = checkpoint['state_dict']
    elif 'networks' in checkpoint and 'reward_net' in checkpoint['networks']:
        reward_state_dict = checkpoint['networks']['reward_net']
    elif isinstance(checkpoint, dict) and any(k.startswith('language_pool') or k.startswith('mlp') for k in checkpoint.keys()):
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
        reward_net.load_state_dict(reward_state_dict)
        print(f"[*] Successfully loaded reward network from checkpoint")
    else:
        raise ValueError(f"Could not find reward network state dict in checkpoint: {cfg.reward_net_checkpoint}")
    
    reward_net.eval()
    return reward_net, feature_extractor


def plot_reward_comparison(
    env_step_rewards,
    reward_net_step_rewards,
    env_episodic_reward,
    reward_net_cumulative,
    episode_num,
    task_description,
    success,
    output_dir,
):
    """
    Create and save reward comparison plots for a single rollout.
    
    Args:
        env_step_rewards: List of environment step-wise rewards
        reward_net_step_rewards: List of reward network step-wise rewards
        env_episodic_reward: Environment episodic reward (1.0 if success, 0.0 otherwise)
        reward_net_cumulative: Cumulative reward from reward network
        episode_num: Episode number
        task_description: Task description string
        success: Whether episode was successful
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    steps = np.arange(len(env_step_rewards))
    
    # Top plot: Step-wise rewards comparison
    ax1.set_title(
        f'Step-wise Rewards - Episode {episode_num} ({task_description})\n'
        f'Success: {success}, Env Episodic: {env_episodic_reward:.2f}, Reward Net Cumulative: {reward_net_cumulative:.4f}',
        fontsize=12
    )
    
    # Plot environment step-wise rewards
    ax1.plot(steps, env_step_rewards, 'b-', label='Environment Step Reward', 
             linewidth=2, alpha=0.7, marker='o', markersize=4)
    
    # Plot reward network step-wise rewards
    if len(reward_net_step_rewards) > 0:
        # Align reward net rewards with env rewards (they should be the same length)
        reward_net_aligned = reward_net_step_rewards[:len(steps)]
        if len(reward_net_aligned) < len(steps):
            # Pad with zeros if needed
            reward_net_aligned = np.pad(reward_net_aligned, (0, len(steps) - len(reward_net_aligned)), 'constant')
        ax1.plot(steps, reward_net_aligned, 'r--', label='Reward Net Step Reward', 
                 linewidth=2, alpha=0.7, marker='s', markersize=4)
    
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Cumulative rewards comparison
    ax2.set_title('Cumulative Rewards', fontsize=12)
    
    # Calculate cumulative environment rewards
    env_cumulative = np.cumsum(env_step_rewards)
    ax2.plot(steps, env_cumulative, 'b-', label=f'Environment Cumulative (Total: {env_cumulative[-1]:.2f})', 
             linewidth=2, alpha=0.7, marker='o', markersize=4)
    
    # Plot episodic reward as horizontal line
    ax2.axhline(y=env_episodic_reward, color='b', linestyle=':', linewidth=2, 
               alpha=0.5, label=f'Environment Episodic Reward ({env_episodic_reward:.2f})')
    
    # Calculate cumulative reward network rewards
    if len(reward_net_step_rewards) > 0:
        reward_net_aligned = reward_net_step_rewards[:len(steps)]
        if len(reward_net_aligned) < len(steps):
            reward_net_aligned = np.pad(reward_net_aligned, (0, len(steps) - len(reward_net_aligned)), 'constant')
        reward_net_cumulative_plot = np.cumsum(reward_net_aligned)
        ax2.plot(steps, reward_net_cumulative_plot, 'r--', 
                 label=f'Reward Net Cumulative (Total: {reward_net_cumulative:.4f})', 
                 linewidth=2, alpha=0.7, marker='s', markersize=4)
    
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Cumulative Reward', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"episode_{episode_num}_reward_comparison.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return plot_path


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Load reward network if checkpoint is provided
    device = torch.device(cfg.device)
    reward_net = None
    feature_extractor = None
    if cfg.evaluate_rewards and cfg.reward_net_checkpoint is not None:
        reward_net, feature_extractor = load_reward_network(cfg, model, processor, device)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    if cfg.evaluate_rewards and reward_net is not None:
        run_id += "-with-reward"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")
    
    # Create directory for reward plots
    reward_plots_dir = os.path.join(cfg.local_log_dir, run_id + "_reward_plots")
    os.makedirs(reward_plots_dir, exist_ok=True)
    print(f"Saving reward plots to: {reward_plots_dir}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    print(f"config: {cfg}")

    # Start evaluation
    total_episodes, total_successes = 0, 0
    all_reward_predictions = []  # Store reward predictions for analysis
    
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
        # Use the actual task description from the environment (not hardcoded)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            replay_images = []
            episode_reward_net_rewards = []  # Store reward network predictions for this episode
            episode_env_rewards = []  # Store environment step-wise rewards
            actual_step_count = 0  # Track actual environment steps (excluding wait steps)
            success_hold_count = 0  # Track consecutive success steps (need 10 for confirmation)
            
            # Note: max_steps is the maximum number of environment steps (not including wait steps)
            # The loop goes through max_steps + num_steps_wait, but we break early if done=True
            # So the actual number of steps varies per episode and is tracked by actual_step_count
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}... (max_steps={max_steps}, num_steps_wait={cfg.num_steps_wait})")
            log_file.write(f"Starting episode {task_episodes+1}... (max_steps={max_steps}, num_steps_wait={cfg.num_steps_wait})\n")
            
            # Loop through max_steps + num_steps_wait, but break early if done=True
            # The actual number of environment steps will be tracked by actual_step_count
            for t in tqdm.tqdm(range(max_steps + cfg.num_steps_wait)):
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }
                    
                    # Query model to get action
                    action = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                    )
                    #if action is list, select first element
                    if isinstance(action, list):
                        action = action[0]
                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    
                    # Track environment step-wise reward
                    episode_env_rewards.append(float(reward))
                    actual_step_count += 1
                    
                    # CRITICAL: Check for success using env._check_success() or env.check_success() method
                    # LIBERO's step() method sets done = self._check_success(), but it might not work correctly
                    # So we check it directly and break early when success is detected
                    task_success = False
                    if hasattr(env, 'check_success'):
                        try:
                            task_success = env.check_success()
                        except:
                            pass
                    elif hasattr(env, '_check_success'):
                        try:
                            task_success = env._check_success()
                        except:
                            pass
                    
                    # Use hold count to avoid false positives (need 10 consecutive success checks)
                    if task_success:
                        if success_hold_count > 0:
                            success_hold_count -= 1  # Decrement count
                            if success_hold_count == 0:
                                # Success confirmed after 10 consecutive steps
                                print(f"  SUCCESS confirmed via check_success() at step {actual_step_count} (held for 10 steps)")
                                log_file.write(f"  SUCCESS confirmed via check_success() at step {actual_step_count} (held for 10 steps)\n")
                                log_file.flush()
                                task_successes += 1
                                total_successes += 1
                                done = True  # Mark as done so we break
                                break
                        else:
                            success_hold_count = 10  # Start hold count on first success
                    else:
                        success_hold_count = 0  # Reset if success check fails
                    
                    # Check done flag (original code behavior)
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    
                    # Evaluate reward if reward network is loaded (only if not done)
                    # Reward network takes (obs, action, obs_next) to predict reward for transition
                    reward_pred = None
                    if reward_net is not None and feature_extractor is not None:
                        with torch.no_grad():
                            # Get next image for reward computation (use current obs after step)
                            img_next = get_libero_image(obs, resize_size)
                            
                            # Extract features for current and next observations
                            lang_feat_curr, vis_feat_curr = feature_extractor.extract_features(
                                [img], [task_description]
                            )
                            lang_feat_next, vis_feat_next = feature_extractor.extract_features(
                                [img_next], [task_description]
                            )
                            
                            # Convert to tensors and move to device
                            lang_feat_curr = lang_feat_curr.to(device).float()
                            vis_feat_curr = vis_feat_curr.to(device).float()
                            lang_feat_next = lang_feat_next.to(device).float()
                            vis_feat_next = vis_feat_next.to(device).float()
                            action_tensor = torch.from_numpy(action).float().to(device).unsqueeze(0)
                            
                            # Predict reward: reward for taking action from img to img_next
                            reward_pred = reward_net(
                                lang_feat_curr,
                                vis_feat_curr,
                                action_tensor,
                                lang_feat_next,
                                vis_feat_next,
                            )
                            reward_pred = reward_pred.item()
                            episode_reward_net_rewards.append(reward_pred)
                            
                            if actual_step_count % 10 == 0:  # Log every 10 steps
                                print(f"  Step {actual_step_count}: Env reward = {reward:.4f}, Predicted reward = {reward_pred:.4f}")
                                log_file.write(f"  Step {actual_step_count}: Env reward = {reward:.4f}, Predicted reward = {reward_pred:.4f}\n")

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            # Calculate environment episodic reward (1.0 if success, 0.0 otherwise)
            # Match original code: done=True means success
            env_episodic_reward = 1.0 if done else 0.0
            is_success = done  # For use in reward plots and statistics (match original: done=True = success)
            
            # Calculate reward network cumulative reward
            reward_net_cumulative = sum(episode_reward_net_rewards) if episode_reward_net_rewards else 0.0
            
            # Create and save reward comparison plot
            if episode_reward_net_rewards or episode_env_rewards:
                plot_path = plot_reward_comparison(
                    env_step_rewards=episode_env_rewards,
                    reward_net_step_rewards=episode_reward_net_rewards,
                    env_episodic_reward=env_episodic_reward,
                    reward_net_cumulative=reward_net_cumulative,
                    episode_num=total_episodes,
                    task_description=task_description,
                    success=done,
                    output_dir=reward_plots_dir,
                )
                print(f"Saved reward plot to: {plot_path}")
                log_file.write(f"Saved reward plot to: {plot_path}\n")

            # Log reward statistics for this episode
            if episode_reward_net_rewards:
                episode_total_reward = reward_net_cumulative
                episode_mean_reward = np.mean(episode_reward_net_rewards)
                episode_max_reward = np.max(episode_reward_net_rewards)
                episode_min_reward = np.min(episode_reward_net_rewards)
                
                env_total_reward = sum(episode_env_rewards)
                env_mean_reward = np.mean(episode_env_rewards) if episode_env_rewards else 0.0
                
                print(f"Episode {total_episodes} reward stats:")
                print(f"  Environment - Step-wise total: {env_total_reward:.4f}, Mean: {env_mean_reward:.4f}, "
                      f"Episodic: {env_episodic_reward:.2f}, Steps: {actual_step_count}")
                print(f"  Reward Net - Total: {episode_total_reward:.4f}, Mean: {episode_mean_reward:.4f}, "
                      f"Max: {episode_max_reward:.4f}, Min: {episode_min_reward:.4f}")
                log_file.write(f"Episode {total_episodes} reward stats:\n")
                log_file.write(f"  Environment - Step-wise total: {env_total_reward:.4f}, Mean: {env_mean_reward:.4f}, "
                              f"Episodic: {env_episodic_reward:.2f}, Steps: {actual_step_count}\n")
                log_file.write(f"  Reward Net - Total: {episode_total_reward:.4f}, Mean: {episode_mean_reward:.4f}, "
                              f"Max: {episode_max_reward:.4f}, Min: {episode_min_reward:.4f}\n")
                
                all_reward_predictions.append({
                    'episode': total_episodes,
                    'task': task_description,
                    'success': done,
                    'actual_steps': actual_step_count,
                    'env_step_rewards': episode_env_rewards,
                    'env_episodic_reward': env_episodic_reward,
                    'reward_net_rewards': episode_reward_net_rewards,
                    'reward_net_cumulative': reward_net_cumulative,
                    'total_reward': episode_total_reward,
                    'mean_reward': episode_mean_reward,
                })
                
                if cfg.use_wandb:
                    wandb.log({
                        f"reward/episode_total": episode_total_reward,
                        f"reward/episode_mean": episode_mean_reward,
                        f"reward/episode_max": episode_max_reward,
                        f"reward/episode_min": episode_min_reward,
                        f"reward/env_episodic": env_episodic_reward,
                        f"reward/env_step_total": env_total_reward,
                        f"reward/actual_steps": actual_step_count,
                        "episode": total_episodes,
                    })

            # Save a replay video of the episode
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
            )

            # Log current results - match original code exactly
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    # Log overall reward statistics
    if all_reward_predictions:
        all_total_rewards = [r['reward_net_cumulative'] for r in all_reward_predictions]
        all_mean_rewards = [r['mean_reward'] for r in all_reward_predictions]
        all_env_episodic = [r['env_episodic_reward'] for r in all_reward_predictions]
        all_actual_steps = [r['actual_steps'] for r in all_reward_predictions]
        
        print(f"\n=== Overall Reward Statistics ===")
        print(f"Reward Network:")
        print(f"  Mean episode total reward: {np.mean(all_total_rewards):.4f} ± {np.std(all_total_rewards):.4f}")
        print(f"  Mean episode mean reward: {np.mean(all_mean_rewards):.4f} ± {np.std(all_mean_rewards):.4f}")
        print(f"Environment:")
        print(f"  Mean episodic reward: {np.mean(all_env_episodic):.4f} ± {np.std(all_env_episodic):.4f}")
        print(f"  Mean actual steps per episode: {np.mean(all_actual_steps):.2f} ± {np.std(all_actual_steps):.2f}")
        log_file.write(f"\n=== Overall Reward Statistics ===\n")
        log_file.write(f"Reward Network:\n")
        log_file.write(f"  Mean episode total reward: {np.mean(all_total_rewards):.4f} ± {np.std(all_total_rewards):.4f}\n")
        log_file.write(f"  Mean episode mean reward: {np.mean(all_mean_rewards):.4f} ± {np.std(all_mean_rewards):.4f}\n")
        log_file.write(f"Environment:\n")
        log_file.write(f"  Mean episodic reward: {np.mean(all_env_episodic):.4f} ± {np.std(all_env_episodic):.4f}\n")
        log_file.write(f"  Mean actual steps per episode: {np.mean(all_actual_steps):.2f} ± {np.std(all_actual_steps):.2f}\n")
        
        if cfg.use_wandb:
            wandb.log({
                "reward/overall_mean_total": np.mean(all_total_rewards),
                "reward/overall_std_total": np.std(all_total_rewards),
                "reward/overall_mean_mean": np.mean(all_mean_rewards),
                "reward/overall_std_mean": np.std(all_mean_rewards),
                "reward/overall_env_episodic_mean": np.mean(all_env_episodic),
                "reward/overall_env_episodic_std": np.std(all_env_episodic),
                "reward/overall_mean_steps": np.mean(all_actual_steps),
                "reward/overall_std_steps": np.std(all_actual_steps),
            })

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()

