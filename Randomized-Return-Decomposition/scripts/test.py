import numpy as np
import os
import sys

# Add parent directory to path so we can import from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import make_env
from utils.os_utils import make_dir

class Tester:
    def __init__(self, args):
        self.args = args
        # Reuse existing environment if available (to avoid loading dataset multiple times)
        if hasattr(args, 'env_instance') and args.env_instance is not None:
            self.env = args.env_instance
        else:
            self.env = make_env(args)
        self.info = []

        if args.save_rews:
            make_dir('log/rews', clear=False)
            self.rews_record = {}
            self.rews_record[args.env] = []

    def test_rollouts(self):
        rewards_sum = 0.0
        rews_List, V_pred_List = [], []
        for _ in range(self.args.test_rollouts):
            rewards = 0.0
            obs = self.env.reset()
            for timestep in range(self.args.test_timesteps):
                action, info = self.args.agent.step(obs, explore=False, test_info=True)
                self.args.logger.add_dict(info)
                if 'test_eps' in self.args.__dict__.keys():
                    # the default testing scheme of Atari games
                    if np.random.uniform(0.0, 1.0)<=self.args.test_eps:
                        action = self.env.action_space.sample()
                obs, reward, done, info = self.env.step(action)
                rewards += reward
                if done: break
            rewards_sum += rewards
            self.args.logger.add_dict(info)

        if self.args.save_rews:
            step = self.args.learner.step_counter
            rews = rewards_sum/self.args.test_rollouts
            self.rews_record[self.args.env].append((step, rews))

    def cycle_summary(self):
        # Skip test rollouts if we're only training reward network (offline/supervised learning)
        if hasattr(self.args, 'rrd_reward_only') and self.args.rrd_reward_only:
            # For reward-only training, we don't have a policy to test
            # Just log that we're skipping testing
            return
        self.test_rollouts()

    def epoch_summary(self):
        if self.args.save_rews:
            for key, acc_info in self.rews_record.items():
                log_folder = 'rews'
                if self.args.tag!='': log_folder = log_folder+'/'+self.args.tag
                self.args.logger.save_npz(acc_info, key, log_folder)

    def final_summary(self):
        pass
