import cv2
import gym
import numpy as np
from utils.os_utils import remove_color

class AtariNormalEnv():
    def __init__(self, args):
        self.args = args
        if args.sticky:
            # frameskip is deterministic
            self.env = gym.make(args.env+'Deterministic-v0').env
        else:
            self.env = gym.make(args.env+'Deterministic-v4').env

        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(84,84,args.frames), dtype=np.float32)
        assert type(self.action_space) is gym.spaces.discrete.Discrete
        self.acts_dims = [self.action_space.n]
        self.obs_dims = list(self.observation_space.shape)

        self.render = self.env.render

        self.reset()
        self.env_info = {
            'Steps': self.process_info_steps, # episode steps
            'Rewards@green': self.process_info_rewards # episode cumulative rewards
        }

    def get_new_frame(self):
        # standard wrapper for atari
        # Use render() instead of private _get_obs() for newer gym versions
        try:
            # Try to get screen from ALE directly (faster, if available)
            if hasattr(self.env, 'ale'):
                frame = self.env.ale.getScreenRGB()
            else:
                # Use render() public API
                frame = self.env.render(mode='rgb_array')
                if frame is None:
                    # Fallback if render doesn't work
                    raise AttributeError("render failed")
        except (AttributeError, TypeError, ValueError):
            # Last resort: try render with different mode
            try:
                frame = self.env.render()
                if not isinstance(frame, np.ndarray):
                    raise TypeError("render returned wrong type")
            except:
                # If all else fails, return a black frame (shouldn't happen)
                frame = np.zeros((210, 160, 3), dtype=np.uint8)
        
        frame = frame.astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84,84), interpolation=cv2.INTER_AREA)
        self.last_frame = frame.copy()
        return frame

    def get_obs(self):
        return self.last_obs.copy()

    def get_frame(self):
        return self.last_frame.copy()

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
        step_result = self.env.step(action)
        # Handle both old (obs, reward, done, info) and new (obs, reward, terminated, truncated, info) gym API
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result
        
        info = self.process_info(obs, reward, info)
        self.frames_stack = self.frames_stack[1:]+[self.get_new_frame()]
        self.last_obs = np.stack(self.frames_stack, axis=-1)
        if self.steps==self.args.test_timesteps: done = True
        return self.last_obs.copy(), reward, done, info

    def step(self, action):
        obs, reward, done, info = self.env_step(action)
        return obs, reward, done, info

    def reset_ep(self):
        self.steps = 0
        self.rewards = 0.0

    def reset(self):
        self.reset_ep()
        while True:
            flag = True
            reset_result = self.env.reset()
            # Handle both old (obs) and new (obs, info) gym API
            # We don't need the return value here, just reset the env
            for _ in range(max(self.args.noop-self.args.frames,0)):
                step_result = self.env.step(0)
                # Handle both old and new step() API
                if len(step_result) == 5:
                    _, _, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    _, _, done, _ = step_result
                if done:
                    flag = False
                    break
            if flag: break

        self.frames_stack = []
        for _ in range(self.args.frames):
            self.env.step(0)  # We don't use the return value here
            self.frames_stack.append(self.get_new_frame())

        self.last_obs = np.stack(self.frames_stack, axis=-1)
        return self.last_obs.copy()
