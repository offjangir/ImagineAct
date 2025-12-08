import gym
import numpy as np
from utils.os_utils import remove_color

class MuJoCoNormalEnv():
    def __init__(self, args):
        self.args = args
        self.env = gym.make(args.env).env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.render = self.env.render
        # Use public API instead of private _get_obs
        # For MuJoCo, observations come from step()/reset(), so we'll use last_obs
        self.last_obs = None

        self.acts_dims = list(self.action_space.shape)
        self.obs_dims = list(self.observation_space.shape)

        self.action_scale = np.array(self.action_space.high)
        for value_low, value_high in zip(list(self.action_space.low), list(self.action_space.high)):
            assert abs(value_low+value_high)<1e-3, (value_low, value_high)

        self.reset()
        self.env_info = {
            'Steps': self.process_info_steps, # episode steps
            'Rewards@green': self.process_info_rewards # episode cumulative rewards
        }

    def process_info_steps(self, obs, reward, info):
        self.steps += 1
        return self.steps

    def process_info_rewards(self, obs, reward, info):
        self.rewards += reward
        return self.rewards

    def process_info(self, obs, reward, info):
        return {
            remove_color(key): value_func(obs, reward, info)
            for key, value_func in self.env_info.items()
        }

    def env_step(self, action):
        step_result = self.env.step(action*self.action_scale)
        # Handle both old (obs, reward, done, info) and new (obs, reward, terminated, truncated, info) gym API
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result
        
        info = self.process_info(obs, reward, info)
        self.last_obs = obs.copy() if hasattr(obs, 'copy') else np.array(obs)
        if self.steps==self.args.test_timesteps: done = True
        return obs, reward, done, info

    def step(self, action):
        obs, reward, done, info = self.env_step(action)
        return obs, reward, done, info

    def get_obs(self):
        """Get current observation"""
        if self.last_obs is not None:
            return self.last_obs.copy()
        else:
            reset_result = self.env.reset()
            # Handle both old (obs) and new (obs, info) gym API
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            return obs.copy() if hasattr(obs, 'copy') else obs

    def reset_ep(self):
        self.steps = 0
        self.rewards = 0.0
        reset_result = self.env.reset()
        # Handle both old (obs) and new (obs, info) gym API
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        self.last_obs = obs.copy() if hasattr(obs, 'copy') else np.array(obs)

    def reset(self):
        self.reset_ep()
        return self.last_obs.copy()
