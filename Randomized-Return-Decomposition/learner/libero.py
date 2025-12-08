import numpy as np

class LiberoLearner:
    """Learner for offline LIBERO training (no environment interaction)"""
    
    def __init__(self, args):
        self.ep_counter = 0
        self.step_counter = 0
        self.target_count = 0
        self.learner_info = []
        
        # For offline RL, step_counter is based on buffer size
        # We'll update it based on training batches
        if hasattr(args, 'buffer') and args.buffer is not None:
            self.step_counter = args.buffer.length
    
    def learn(self, args, env, agent, buffer):
        """Offline learning: train on one batch from buffer"""
        # Skip data collection - all data is already in buffer
        
        # Update step counter from buffer
        self.step_counter = buffer.length
        
        self.target_count += 1
        
        # Update normalizer if needed (once per epoch, checked externally)
        if hasattr(agent, 'obs_normalizer') and agent.obs_normalizer is not None:
            # Only update normalizer occasionally to avoid overhead
            if self.target_count % 100 == 0:
                agent.normalizer_update(buffer.sample_batch())
        
        if 'pi_delay_freq' in args.__dict__.keys():
            # TD3-style delayed policy updates
            batch = buffer.sample_batch()
            info = agent.train_q(batch)
            args.logger.add_dict(info)
            if self.target_count % args.pi_delay_freq == 0:
                batch = buffer.sample_batch()
                info = agent.train_pi(batch)
                args.logger.add_dict(info)
        else:
            # Check if we're doing reward-only training (supervised learning)
            if hasattr(args, 'rrd_reward_only') and args.rrd_reward_only:
                # Only train reward network, skip policy learning
                batch = buffer.sample_batch()
                if hasattr(agent, 'train_r'):
                    info = agent.train_r(batch)
                else:
                    raise ValueError("Reward-only training requires RRD algorithm")
                args.logger.add_dict(info)
            else:
                # Standard training (SAC, DDPG, RRD) - trains both policy and reward
                batch = buffer.sample_batch()
                info = agent.train(batch)
                args.logger.add_dict(info)
        
        # Update target networks
        if self.target_count % args.train_target == 0:
            agent.target_update()
        
        # Update episode counter (approximate based on buffer)
        # For offline RL, we estimate based on average episode length
        if buffer.length > 0 and len(buffer.ep) > 0:
            avg_ep_length = buffer.length / len(buffer.ep)
            self.ep_counter = int(buffer.length / avg_ep_length) if avg_ep_length > 0 else 0

