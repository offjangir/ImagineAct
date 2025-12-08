import numpy as np
import time
import os
import sys

# Set protobuf implementation to avoid version conflicts with TensorBoard
# This allows TensorFlow/tensorflow_datasets to work with newer protobuf versions
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

# Add parent directory to path so we can import from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import get_args,experiment_setup
from utils.os_utils import make_dir

if __name__=='__main__':
    args = get_args()
    env, agent, buffer, learner, tester = experiment_setup(args)

    args.logger.summary_init(agent.graph, agent.sess)

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

    # Test info
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

    episodes_cnt = 0
    global_step = 0  # Track global training step for TensorBoard
    for epoch in range(args.epochs):
        for cycle in range(args.cycles):
            args.logger.tabular_clear()
            args.logger.summary_clear()

            start_time = time.time()
            learner.learn(args, env, agent, buffer)
            args.logger.add_record('TimeCost(sec)/train', time.time()-start_time)

            start_time = time.time()
            # Skip testing if we're only training reward network (offline/supervised learning)
            if not (hasattr(args, 'rrd_reward_only') and args.rrd_reward_only):
                tester.cycle_summary()
            else:
                # For reward-only training, log that testing is skipped
                args.logger.info("Skipping test rollouts (reward-only training mode)")
            args.logger.add_record('TimeCost(sec)/test', time.time()-start_time)

            args.logger.add_record('Epoch', str(epoch)+'/'+str(args.epochs))
            args.logger.add_record('Cycle', str(cycle)+'/'+str(args.cycles))
            args.logger.add_record('Episodes', learner.ep_counter)
            args.logger.add_record('Timesteps', learner.step_counter)

            args.logger.tabular_show(args.tag)
            # Use global_step instead of learner.step_counter for TensorBoard
            # This ensures each cycle gets a unique step number instead of overwriting
            args.logger.summary_show(global_step)
            global_step += 1

        tester.epoch_summary()
        
        # Save checkpoint periodically
        if checkpoint_dir is not None and args.save_freq > 0 and (epoch + 1) % args.save_freq == 0:
            # Use appropriate extension based on backend
            USE_PYTORCH = os.environ.get('USE_PYTORCH', '0') == '1'
            ext = '.pt' if USE_PYTORCH else '.ckpt'
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}{ext}')
            try:
                agent.save_model(checkpoint_path)
                args.logger.info(f'Saved checkpoint to {checkpoint_path}')
            except Exception as e:
                args.logger.warning(f'Failed to save checkpoint: {e}')

    tester.final_summary()
    
    # Save final checkpoint
    if checkpoint_dir is not None and args.save_final:
        # Use appropriate extension based on backend
        USE_PYTORCH = os.environ.get('USE_PYTORCH', '0') == '1'
        ext = '.pt' if USE_PYTORCH else '.ckpt'
        final_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_final{ext}')
        try:
            agent.save_model(final_checkpoint_path)
            args.logger.info(f'Saved final checkpoint to {final_checkpoint_path}')
        except Exception as e:
            args.logger.warning(f'Failed to save final checkpoint: {e}')
