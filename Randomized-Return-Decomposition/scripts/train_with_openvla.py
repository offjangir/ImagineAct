#!/usr/bin/env python3
"""
Training script with OpenVLA features for reward model training.

This script loads OpenVLA model and uses its language and vision features
to train reward models in Randomized-Return-Decomposition.
"""

import os
import sys
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for plotting
import matplotlib.pyplot as plt

# Set environment variable to use PyTorch backend (required for OpenVLA)
os.environ['USE_PYTORCH'] = '1'

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Add OpenVLA path if needed
openvla_path = os.path.join(os.path.dirname(project_root), 'openvla')
if os.path.exists(openvla_path) and openvla_path not in sys.path:
    sys.path.insert(0, openvla_path)

import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from common import get_args, experiment_setup
from utils.os_utils import make_dir


def load_openvla_model(checkpoint_path, device="cuda:0"):
    """Load OpenVLA model and processor.
    
    Args:
        checkpoint_path: Either a local directory path or a HuggingFace model ID
        device: Device to load model on
    """
    import os
    
    # Register OpenVLA model to HF Auto Classes
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    
    # Determine if it's a local path or HuggingFace model ID
    is_local_path = os.path.exists(checkpoint_path) and os.path.isdir(checkpoint_path)
    
    if is_local_path:
        # Local checkpoint
        config_path = os.path.join(checkpoint_path, "config.json")
        if not os.path.exists(config_path):
            raise ValueError(
                f"Checkpoint directory does not contain config.json: {checkpoint_path}\n"
                f"This doesn't appear to be a valid OpenVLA checkpoint."
            )
        abs_checkpoint_path = os.path.abspath(checkpoint_path)
        print(f"[*] Loading OpenVLA from local path: {abs_checkpoint_path}...")
        local_files_only = True
    else:
        # HuggingFace model ID (e.g., "prismatical/openvla-7b")
        print(f"[*] Loading OpenVLA from HuggingFace: {checkpoint_path}...")
        print(f"[*] This will download the model if not already cached.")
        abs_checkpoint_path = checkpoint_path
        local_files_only = False
    
    # Load model
    try:
        vla_model = AutoModelForVision2Seq.from_pretrained(
            abs_checkpoint_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=local_files_only,
        ).to(device)
    except Exception as e:
        if is_local_path:
            error_msg = (
                f"Failed to load OpenVLA model from local path: {abs_checkpoint_path}\n"
                f"Error: {e}\n"
                f"\nPossible solutions:\n"
                f"1. Verify the checkpoint path is correct\n"
                f"2. Check that all required files are present (config.json, model files, etc.)\n"
                f"3. Try using a HuggingFace model ID instead (e.g., 'prismatical/openvla-7b')"
            )
        else:
            error_msg = (
                f"Failed to load OpenVLA model from HuggingFace: {abs_checkpoint_path}\n"
                f"Error: {e}\n"
                f"\nPossible solutions:\n"
                f"1. Check your internet connection\n"
                f"2. Verify the model ID is correct\n"
                f"3. Try downloading the model locally first"
            )
        raise RuntimeError(error_msg) from e
    
    # Load processor
    try:
        processor = AutoProcessor.from_pretrained(
            abs_checkpoint_path,
            trust_remote_code=True,
            local_files_only=local_files_only
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load OpenVLA processor from {abs_checkpoint_path}\n"
            f"Error: {e}"
        ) from e
    
    print(f"[*] OpenVLA loaded successfully on {device}!")
    return vla_model, processor


def main():
    # Get command line arguments
    args = get_args()
    
    # Check if OpenVLA features are enabled
    if hasattr(args, 'use_openvla_features') and args.use_openvla_features:
        if not hasattr(args, 'openvla_checkpoint') or args.openvla_checkpoint is None:
            raise ValueError(
                "use_openvla_features=True requires --openvla_checkpoint to be specified. "
                "Example: --openvla_checkpoint /path/to/openvla/checkpoint"
            )
        
        # Determine device
        use_cuda = getattr(args, 'cuda', torch.cuda.is_available())
        if use_cuda and torch.cuda.is_available():
            # Check if device is specified in args
            if hasattr(args, 'device') and args.device:
                device = args.device if isinstance(args.device, str) else f"cuda:{args.device}"
            else:
                device = "cuda:0"
        else:
            device = "cpu"
        
        # Load OpenVLA model
        vla_model, processor = load_openvla_model(args.openvla_checkpoint, device=device)
        
        # Set OpenVLA model and processor in args
        args.openvla_model = vla_model
        args.openvla_processor = processor
        
        # Set task description if provided
        if hasattr(args, 'openvla_task_description') and args.openvla_task_description:
            args.task_description = args.openvla_task_description
        
        print(f"[*] OpenVLA features enabled!")
        print(f"[*] Task description: {getattr(args, 'task_description', 'N/A')}")
        print(f"[*] Reward network hidden dim: {getattr(args, 'openvla_reward_hidden_dim', 512)}")
        print(f"[*] Reward network num layers: {getattr(args, 'openvla_reward_num_layers', 3)}")
    
    # Setup experiment
    env, agent, buffer, learner, tester = experiment_setup(args)
    
    # Pre-extract OpenVLA features if enabled
    # NOTE: Pre-extraction can be slow. If training with on-the-fly extraction is already fast enough,
    # you can skip pre-extraction by setting SKIP_PRE_EXTRACTION=True or commenting out this section
    SKIP_PRE_EXTRACTION = False  # Set to True to skip pre-extraction and use on-the-fly extraction
    
    if hasattr(args, 'use_openvla_features') and args.use_openvla_features and not SKIP_PRE_EXTRACTION:
        if hasattr(agent, 'openvla_extractor') and agent.openvla_extractor is not None:
            # Check if buffer supports feature pre-extraction
            if hasattr(buffer, 'pre_extract_openvla_features'):
                # Use larger batch size for faster feature extraction (adjust based on GPU memory)
                # 128 should work on most GPUs, increase to 256 if you have more memory
                feature_batch_size = 128
                num_workers = 256  # Number of parallel workers for processing episodes
                args.logger.info("Starting feature pre-extraction...")
                args.logger.info(f"Using {num_workers} parallel workers, batch size {feature_batch_size} per episode")
                args.logger.info("Note: If this is too slow, you can skip pre-extraction and use on-the-fly extraction during training")
                buffer.pre_extract_openvla_features(
                    agent.openvla_extractor, 
                    batch_size=feature_batch_size,
                    num_workers=num_workers
                )
            else:
                args.logger.warning("Buffer does not support feature pre-extraction. Features will be extracted on-the-fly.")
        else:
            args.logger.warning("OpenVLA feature extractor not found in agent. Features will be extracted on-the-fly.")
    elif hasattr(args, 'use_openvla_features') and args.use_openvla_features and SKIP_PRE_EXTRACTION:
        args.logger.info("Skipping feature pre-extraction. Features will be extracted on-the-fly during training.")
    
    # Setup logging (PyTorch doesn't use graph/sess, but we keep for compatibility)
    use_wandb = getattr(args, 'use_wandb', False)
    # Use custom run name if provided, otherwise use tag
    run_name = getattr(args, 'wandb_run_name', None) or getattr(args, 'tag', '')
    wandb_config = {
        'name': run_name,
        'env': getattr(args, 'env', ''),
        'alg': getattr(args, 'alg', ''),
        'basis_alg': getattr(args, 'basis_alg', ''),
        'use_openvla_features': getattr(args, 'use_openvla_features', False),
        'rrd_reward_only': getattr(args, 'rrd_reward_only', False),
        'epochs': getattr(args, 'epochs', 0),
        'cycles': getattr(args, 'cycles', 0),
        'iterations': getattr(args, 'iterations', 0),
        'train_batches': getattr(args, 'train_batches', 0),
        'batch_size': getattr(args, 'batch_size', 0),
        'rrd_batch_size': getattr(args, 'rrd_batch_size', 0),
        'rrd_sample_size': getattr(args, 'rrd_sample_size', 0),
        'r_lr': getattr(args, 'r_lr', 0),
    }
    
    if hasattr(agent, 'graph') and agent.graph is not None:
        args.logger.summary_init(
            agent.graph, agent.sess,
            use_wandb=use_wandb,
            wandb_project=getattr(args, 'wandb_project', None),
            wandb_entity=getattr(args, 'wandb_entity', None),
            wandb_config=wandb_config
        )
    else:
        # For PyTorch, initialize logger without graph/sess
        args.logger.summary_init(
            None, None,
            use_wandb=use_wandb,
            wandb_project=getattr(args, 'wandb_project', None),
            wandb_entity=getattr(args, 'wandb_entity', None),
            wandb_config=wandb_config
        )
    
    # Progress info
    args.logger.add_item('Epoch')
    args.logger.add_item('Cycle')
    args.logger.add_item('Episodes@green')
    args.logger.add_item('Timesteps')
    args.logger.add_item('TimeCost(sec)/train')
    args.logger.add_item('TimeCost(sec)/test')
    
    # Algorithm info
    for key in agent.train_info.keys():
        args.logger.add_item(key, 'scalar')
    for key in learner.learner_info:
        args.logger.add_item(key, 'scalar')
    
    # Test info (only log if not reward-only training, as there's no environment interaction)
    if not (hasattr(args, 'rrd_reward_only') and args.rrd_reward_only):
        for key in agent.step_info.keys():
            args.logger.add_item(key, 'scalar')
        for key in env.env_info.keys():
            args.logger.add_item(key, 'scalar')
        for key in tester.info:
            args.logger.add_item(key, 'scalar')
    
    args.logger.summary_setup()
    
    # Setup checkpoint directory
    checkpoint_dir = None
    if args.save_freq > 0 or args.save_final:
        checkpoint_base_dir = args.checkpoint_dir
        if args.tag != '':
            checkpoint_dir = os.path.join(checkpoint_base_dir, args.tag)
        else:
            checkpoint_dir = os.path.join(checkpoint_base_dir, args.alg + '-' + args.env)
        make_dir(checkpoint_dir, clear=False)
        args.logger.info(f'Checkpoints will be saved to: {checkpoint_dir}')
    
    # Training loop
    episodes_cnt = 0
    global_step = 0
    
    # Track best loss for checkpoint saving
    best_epoch_loss = None
    best_epoch = None
    
    # Calculate iterations per epoch based on data size
    if hasattr(args, 'rrd_reward_only') and args.rrd_reward_only:
        batch_size = getattr(args, 'rrd_batch_size', args.batch_size)
    else:
        batch_size = args.batch_size
    
    iterations_per_epoch = max(1, buffer.length // batch_size)
    
    args.logger.info("\n" + "="*80)
    args.logger.info("Starting Training")
    args.logger.info("="*80)
    args.logger.info(f"Total epochs: {args.epochs}")
    args.logger.info(f"Data size: {buffer.length} steps, Batch size: {batch_size}")
    args.logger.info(f"Iterations per epoch: {iterations_per_epoch} (data_size / batch_size)")
    args.logger.info("="*80 + "\n")
    print(f"Starting Training: {args.epochs} epochs, {iterations_per_epoch} iterations/epoch")
    
    for epoch in range(args.epochs):
        # Track R_loss for this epoch
        epoch_r_loss_sum = 0.0
        epoch_r_loss_count = 0
        epoch_header = f"\n{'='*80}\nEPOCH {epoch+1}/{args.epochs}\n{'='*80}"
        args.logger.info(epoch_header)
        
        # Create tqdm progress bar for iterations in this epoch
        # Use stderr to avoid conflicts with logger stdout output
        import sys
        iteration_pbar = tqdm(
            range(iterations_per_epoch),
            desc=f"Epoch {epoch+1}/{args.epochs}",
            unit="iter",
            ncols=120,
            leave=True,
            file=sys.stderr,  # Write to stderr to avoid conflicts with logger
            dynamic_ncols=True,  # Adjust width dynamically
            mininterval=0.1,  # Update at most every 0.1 seconds
            maxinterval=1.0,  # Force update every 1 second
        )
        
        args.logger.tabular_clear()
        args.logger.summary_clear()
        
        import time
        start_time = time.time()
        
        # Track moving average of R_loss (for postfix display)
        moving_avg_window = 10  # Number of recent batches to average
        recent_r_losses = []
        prev_r_loss_total = 0.0
        prev_r_loss_count = 0
        
        for iteration in iteration_pbar:
            learner.learn(args, env, agent, buffer)
            
            # Extract R_loss from this batch (compute incremental value)
            batch_r_loss = None
            if 'R_loss' in args.logger.values and 'R_loss' in args.logger.counts:
                current_total = args.logger.values['R_loss']
                current_count = args.logger.counts['R_loss']
                
                if current_count > prev_r_loss_count:
                    # Calculate the new batch loss: (new_total - prev_total) / (new_count - prev_count)
                    new_batch_total = current_total - prev_r_loss_total
                    new_batch_count = current_count - prev_r_loss_count
                    batch_r_loss = new_batch_total / new_batch_count if new_batch_count > 0 else None
                    
                    # Update moving average
                    if batch_r_loss is not None:
                        recent_r_losses.append(batch_r_loss)
                        if len(recent_r_losses) > moving_avg_window:
                            recent_r_losses.pop(0)
                    
                    # Update previous values for next iteration
                    prev_r_loss_total = current_total
                    prev_r_loss_count = current_count
                    
                    # Accumulate for epoch average
                    epoch_r_loss_sum = current_total
                    epoch_r_loss_count = current_count
            
            # Calculate moving average
            moving_avg_r_loss = None
            if len(recent_r_losses) > 0:
                moving_avg_r_loss = sum(recent_r_losses) / len(recent_r_losses)
            
            # Update tqdm postfix with moving average
            postfix_dict = {}
            if moving_avg_r_loss is not None:
                postfix_dict['R_loss (MA)'] = f"{moving_avg_r_loss:.6f}"
            elif batch_r_loss is not None:
                postfix_dict['R_loss'] = f"{batch_r_loss:.6f}"
            else:
                postfix_dict['R_loss'] = "N/A"
            iteration_pbar.set_postfix(postfix_dict, refresh=True)
        
        args.logger.add_record('TimeCost(sec)/train', time.time()-start_time)
        args.logger.add_record('Epoch', str(epoch)+'/'+str(args.epochs))
        args.logger.add_record('Episodes', learner.ep_counter)
        args.logger.add_record('Timesteps', learner.step_counter)
        
        # Close the progress bar for this epoch
        iteration_pbar.close()
        
        # Compute average R_loss for this epoch
        epoch_avg_r_loss = None
        if epoch_r_loss_count > 0:
            epoch_avg_r_loss = epoch_r_loss_sum / epoch_r_loss_count
        
        # Log epoch completion with average loss
        epoch_complete_msg = f"\n{'='*80}\nEpoch {epoch+1}/{args.epochs} completed!"
        if epoch_avg_r_loss is not None:
            epoch_complete_msg += f"\nAverage R_loss: {epoch_avg_r_loss:.6f}"
        epoch_complete_msg += f"\n{'='*80}\n"
        args.logger.info(epoch_complete_msg)
        if epoch_avg_r_loss is not None:
            print(f"Epoch {epoch+1}/{args.epochs} completed - Avg R_loss: {epoch_avg_r_loss:.6f}")
        
        args.logger.tabular_show(args.tag)
        args.logger.summary_show(global_step)
        global_step += 1
        
        tester.epoch_summary()
        
        # Evaluate reward predictions every 10 epochs
        if (epoch + 1) % 10 == 0:
            args.logger.info(f"\n{'='*80}\nEvaluating reward predictions (epoch {epoch+1})...\n{'='*80}")
            try:
                # Import evaluation function (same directory)
                import importlib.util
                eval_module_path = os.path.join(os.path.dirname(__file__), 'evaluate_rewards.py')
                spec = importlib.util.spec_from_file_location("evaluate_rewards", eval_module_path)
                eval_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(eval_module)
                evaluate_reward_predictions = eval_module.evaluate_reward_predictions
                log_figures_and_videos_to_wandb = eval_module.log_figures_and_videos_to_wandb
                
                # Evaluate on 5 trajectories
                eval_figures, eval_video_paths = evaluate_reward_predictions(agent, buffer, num_trajectories=5, device=args.device, epoch=epoch+1)
                
                # Log to WandB if available
                if hasattr(args.logger, 'summary_writer') and args.logger.summary_writer is not None:
                    log_figures_and_videos_to_wandb(eval_figures, eval_video_paths, args.logger.summary_writer, epoch+1, global_step)
                    args.logger.info(f"✓ Evaluation plots and videos logged to WandB")
                
                # Also save figures locally
                eval_plots_dir = os.path.join('log', 'eval_plots', args.tag)
                os.makedirs(eval_plots_dir, exist_ok=True)
                for i, fig in enumerate(eval_figures):
                    fig.savefig(os.path.join(eval_plots_dir, f'epoch_{epoch+1}_traj_{i+1}.png'), dpi=100, bbox_inches='tight')
                    plt.close(fig)
                args.logger.info(f"✓ Evaluation plots saved to {eval_plots_dir}")
                
                # Videos are already saved by create_trajectory_video, just log the directory
                eval_videos_dir = os.path.join('log', 'eval_videos')
                if eval_video_paths and any(p is not None for p in eval_video_paths):
                    args.logger.info(f"✓ Evaluation videos saved to {eval_videos_dir}")
                
                args.logger.info(f"✓ Evaluation complete for epoch {epoch+1}\n")
            except Exception as e:
                args.logger.warning(f"✗ Evaluation failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Check if this is the best epoch (minimum loss)
        is_best_epoch = False
        if epoch_avg_r_loss is not None:
            if best_epoch_loss is None or epoch_avg_r_loss < best_epoch_loss:
                    best_epoch_loss = epoch_avg_r_loss
                    best_epoch = epoch + 1
                    is_best_epoch = True
                    best_msg = f"\n{'='*80}\n🎉 NEW BEST EPOCH! Epoch {epoch+1} has minimum loss: {epoch_avg_r_loss:.6f}\n{'='*80}\n"
                    args.logger.info(best_msg)
                    print(f"🎉 NEW BEST EPOCH! Epoch {epoch+1} - Loss: {epoch_avg_r_loss:.6f}")
        
            # Save checkpoint if this is the best epoch
            if checkpoint_dir is not None and is_best_epoch:
                ext = '.pt'  # PyTorch checkpoint
                best_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_best_epoch_{epoch+1}_loss_{epoch_avg_r_loss:.6f}{ext}')
                checkpoint_msg = f"\n{'='*80}\nSaving BEST checkpoint: {best_checkpoint_path}\n{'='*80}"
                args.logger.info(checkpoint_msg)
                try:
                    agent.save_model(best_checkpoint_path)
                    # Also save as checkpoint_best.pt for easy access
                    best_checkpoint_simple = os.path.join(checkpoint_dir, f'checkpoint_best{ext}')
                    agent.save_model(best_checkpoint_simple)
                    # Verify file was created
                    if os.path.exists(best_checkpoint_path):
                        file_size = os.path.getsize(best_checkpoint_path) / (1024 * 1024)  # Size in MB
                        success_msg = f"✓ Best checkpoint saved! Size: {file_size:.2f} MB"
                        args.logger.info(success_msg)
                        print(f"✓ Saved best checkpoint ({file_size:.2f} MB)")
                    else:
                        error_msg = "✗ Warning: Best checkpoint file not found after save operation!"
                        args.logger.warning(error_msg)
                        print(error_msg)
                except Exception as e:
                    error_msg = f"✗ Error saving best checkpoint: {e}"
                    args.logger.warning(error_msg)
                    print(error_msg)
                    import traceback
                    traceback.print_exc()
        
        # Save checkpoint periodically (in addition to best checkpoint)
        if checkpoint_dir is not None and args.save_freq > 0 and (epoch + 1) % args.save_freq == 0:
            ext = '.pt'  # PyTorch checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}{ext}')
            checkpoint_msg = f"\n{'='*80}\nSaving checkpoint: {checkpoint_path}\n{'='*80}"
            args.logger.info(checkpoint_msg)
            try:
                agent.save_model(checkpoint_path)
                # Verify file was created
                if os.path.exists(checkpoint_path):
                    file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # Size in MB
                    success_msg = f"✓ Checkpoint saved successfully! Size: {file_size:.2f} MB"
                    args.logger.info(success_msg)
                    print(f"✓ Saved checkpoint epoch {epoch+1} ({file_size:.2f} MB)")
                else:
                    error_msg = "✗ Warning: Checkpoint file not found after save operation!"
                    args.logger.warning(error_msg)
                    print(error_msg)
            except Exception as e:
                error_msg = f"✗ Error saving checkpoint: {e}"
                args.logger.warning(error_msg)
                print(error_msg)
                import traceback
                traceback.print_exc()
    
    tester.final_summary()
    
    # Save final checkpoint
    if checkpoint_dir is not None and args.save_final:
        ext = '.pt'
        final_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_final{ext}')
        final_msg = f"\n{'='*80}\nSaving FINAL checkpoint: {final_checkpoint_path}\n{'='*80}"
        args.logger.info(final_msg)
        print(final_msg)
        try:
            agent.save_model(final_checkpoint_path)
            # Verify file was created
            if os.path.exists(final_checkpoint_path):
                file_size = os.path.getsize(final_checkpoint_path) / (1024 * 1024)  # Size in MB
                success_msg = f"✓ Final checkpoint saved successfully! Size: {file_size:.2f} MB"
                args.logger.info(success_msg)
                print(f"✓ Saved final checkpoint ({file_size:.2f} MB)")
            else:
                error_msg = "✗ Warning: Final checkpoint file not found after save operation!"
                args.logger.warning(error_msg)
                print(error_msg)
        except Exception as e:
            error_msg = f"✗ Error saving final checkpoint: {e}"
            args.logger.warning(error_msg)
            print(error_msg)
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()

