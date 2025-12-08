# Adapting RRD for LIBERO-10 Dataset

This document outlines the changes needed to train RRD on the LIBERO-10 dataset and addresses whether RRD requires sparse rewards.

## Does RRD Require Sparse Rewards?

### Short Answer: **No, but it's designed for episodic rewards**

RRD does **not strictly require sparse rewards**, but it is designed to work with **episodic reward structure**:

1. **RRD works with any reward structure** as long as you can compute episodic returns
2. **It's most beneficial when rewards are sparse or delayed** (long-horizon tasks)
3. **It can work with dense rewards** - it will learn to redistribute them

### How RRD Handles Rewards

RRD uses the **episodic return** (sum of all rewards in an episode) as the training target:

```python
# From rrd_buffer.py
def rrd_sample(self, sample_size, store_coef=False):
    # Episodic return = sum of all rewards in episode
    'rrd_rews': [self.sum_rews/self.length]  # Average reward per step
```

**Note**: The implementation uses `sum_rews/length` (average reward per step), but the training objective compares the **sum of predicted rewards** to the **episodic return**. This is mathematically equivalent.

### Episodic Rewards Wrapper

The repository includes `envs/ep_rews.py` which converts dense rewards to sparse episodic rewards:

```python
# Accumulates rewards during episode
# Only gives reward at episode end (sum of all rewards)
# All intermediate steps get reward = 0.0
```

**For LIBERO-10**: You can use this wrapper if you want to test RRD's sparse reward handling, but it's not required.

## Changes Needed for LIBERO-10

### 1. Create LIBERO Environment Wrapper

**Location**: `envs/normal_libero.py` (new file)

```python
import numpy as np
import rlds
from utils.os_utils import remove_color

class LiberoNormalEnv:
    def __init__(self, args):
        self.args = args
        
        # Load LIBERO-10 dataset
        dataset_path = args.libero_dataset_path  # e.g., 'modified_libero_rlds/libero_10_no_noops/1.0.0'
        self.dataset = rlds.builder_from_directory(dataset_path)
        
        # Extract observation and action dimensions from dataset
        # LIBERO provides: RGB images, proprioception, language instructions
        # You'll need to decide on observation format:
        # Option 1: Use images only (need to handle multi-camera)
        # Option 2: Use proprioception only
        # Option 3: Concatenate image features + proprioception
        
        # For now, assuming you'll use a processed observation format
        self.observation_space = ...  # Define based on your choice
        self.action_space = ...       # LIBERO uses 7-DoF actions
        
        self.acts_dims = list(self.action_space.shape)  # [7] for 7-DoF
        self.obs_dims = list(self.observation_space.shape)
        
        # Track current episode
        self.current_episode = None
        self.current_step = 0
        self.episode_iterator = None
        
        self.reset()
        
        self.env_info = {
            'Steps': self.process_info_steps,
            'Rewards@green': self.process_info_rewards
        }
    
    def reset(self):
        """Reset to start of a new episode from dataset"""
        # Sample a random episode from dataset
        # LIBERO-10 has multiple episodes per task
        self.current_episode = self._sample_episode()
        self.current_step = 0
        self.rewards = 0.0
        self.steps = 0
        
        # Get initial observation
        obs = self._get_obs_at_step(0)
        self.last_obs = obs
        return obs.copy()
    
    def step(self, action):
        """Step through dataset episode"""
        # For offline RL, we're replaying demonstrations
        # Action is ignored (or used for learning, but doesn't affect trajectory)
        
        self.current_step += 1
        
        if self.current_step >= len(self.current_episode['steps']):
            # Episode done
            done = True
            obs_next = self.last_obs  # Terminal state
            reward = 0.0  # Reward already accumulated
        else:
            done = False
            obs_next = self._get_obs_at_step(self.current_step)
            reward = self._get_reward_at_step(self.current_step)
        
        info = self.process_info(obs_next, reward, {})
        self.last_obs = obs_next
        self.rewards += reward
        self.steps += 1
        
        return obs_next.copy(), reward, done, info
    
    def _sample_episode(self):
        """Sample a random episode from LIBERO-10 dataset"""
        # LIBERO-10 is in RLDS format
        # Each episode has: steps, language_instruction, etc.
        episodes = list(self.dataset['train'].as_numpy_iterator())
        return np.random.choice(episodes)
    
    def _get_obs_at_step(self, step_idx):
        """Extract observation at given step"""
        step = self.current_episode['steps'][step_idx]
        
        # Extract observation based on your choice:
        # Option 1: Images only
        # obs = step['observation']['image']  # Shape: (H, W, C) or multi-camera
        
        # Option 2: Proprioception only
        # obs = step['observation']['state']  # Shape: (state_dim,)
        
        # Option 3: Concatenate features
        # image_features = process_image(step['observation']['image'])
        # obs = np.concatenate([image_features, step['observation']['state']])
        
        # For now, placeholder - you'll need to implement based on your needs
        obs = step['observation']['state']  # Assuming proprioception
        return obs
    
    def _get_reward_at_step(self, step_idx):
        """Extract reward at given step"""
        step = self.current_episode['steps'][step_idx]
        return float(step['reward'])
    
    def get_obs(self):
        return self.last_obs.copy()
    
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
```

### 2. Update Environment Factory

**File**: `envs/__init__.py`

```python
from .normal_libero import LiberoNormalEnv

# Add to envs_collection
envs_collection = {
    # ... existing envs ...
    'libero-10': 'libero',
    # Add more LIBERO tasks as needed
}

def make_env(args):
    normal_env = {
        'atari': AtariNormalEnv,
        'mujoco': MuJoCoNormalEnv,
        'libero': LiberoNormalEnv  # Add this
    }[envs_collection[args.env]]
    
    return {
        'normal': normal_env,
        'ep_rews': create_EpisodicRewardsEnv(normal_env)
    }[args.env_type](args)
```

### 3. Add LIBERO Arguments

**File**: `common.py`

```python
def get_args():
    # ... existing code ...
    
    def libero_args():
        parser.add_argument('--libero_dataset_path', 
                           help='Path to LIBERO-10 RLDS dataset', 
                           type=str, 
                           default='modified_libero_rlds/libero_10_no_noops/1.0.0')
        parser.add_argument('--libero_obs_mode', 
                           help='Observation mode: state, image, or combined', 
                           type=str, 
                           default='state',
                           choices=['state', 'image', 'combined'])
        parser.add_argument('--libero_image_size', 
                           help='Image size if using image observations', 
                           type=int, 
                           default=224)
        # Set defaults for LIBERO
        parser.set_defaults(
            epochs=10,
            cycles=50,
            iterations=100,
            timesteps=200,  # LIBERO episodes are typically 200-500 steps
            test_rollouts=5,
            test_timesteps=500,
            batch_size=128,
            warmup=5000,
            buffer_size=500000
        )
    
    env_args_collection = {
        'atari': atari_args,
        'mujoco': mujoco_args,
        'libero': libero_args  # Add this
    }
```

### 4. Handle Image Observations (if using)

If you want to use image observations from LIBERO:

**Option A: Use existing ConvRewardNet**
- LIBERO provides RGB images from workspace and wrist cameras
- You'll need to decide: single camera or multi-camera
- May need to modify `ConvRewardNet` to handle different image sizes

**Option B: Extract image features first**
- Use a pretrained vision encoder (e.g., ResNet, ViT)
- Extract features and use `MLPRewardNet` on features
- More efficient but requires feature extraction step

### 5. Offline RL Considerations

**Important**: RRD is designed for **online RL** (agent interacts with environment). For **offline RL** (training on fixed dataset):

#### Option 1: Offline Reward Model Training Only

If you only want to train the reward model (as mentioned in your notes):

```python
# Create a simplified training script
# Only train reward network, don't train policy

def train_reward_model_only(args):
    # Load LIBERO dataset
    env = LiberoNormalEnv(args)
    
    # Create reward network
    agent = create_agent(args)  # RRD agent
    
    # Create offline buffer from dataset
    buffer = create_offline_buffer_from_libero(args, env)
    
    # Only train reward network
    for epoch in range(args.epochs):
        for cycle in range(args.cycles):
            batch = buffer.sample_batch()
            # Only call train_r, not train_q or train_pi
            r_info = agent.train_r(batch)
            # Log reward loss
```

#### Option 2: Offline RL with RRD

For full offline RL, you'll need to:
- Disable environment interaction
- Sample only from dataset
- Use offline RL techniques (e.g., conservative Q-learning, CQL)

### 6. Create Offline Buffer

**File**: `algorithm/replay_buffer/libero_buffer/offline_libero_buffer.py` (new)

```python
import numpy as np
import rlds
from .rrd_buffer import ReplayBuffer_RRD

class OfflineLiberoBuffer(ReplayBuffer_RRD):
    """Offline buffer that loads LIBERO-10 dataset"""
    
    def __init__(self, args):
        super().__init__(args)
        
        # Load LIBERO dataset
        dataset_path = args.libero_dataset_path
        self.dataset = rlds.builder_from_directory(dataset_path)
        
        # Preload all episodes into memory
        self._load_episodes()
    
    def _load_episodes(self):
        """Load all episodes from LIBERO dataset"""
        self.ep = []
        
        for episode in self.dataset['train'].as_numpy_iterator():
            # Convert RLDS episode to Trajectory format
            trajectory = self._episode_to_trajectory(episode)
            self.ep.append(trajectory)
            self.ep_counter += 1
            self.length += trajectory.length
        
        print(f"Loaded {len(self.ep)} episodes, {self.length} total steps")
    
    def _episode_to_trajectory(self, episode):
        """Convert RLDS episode to Trajectory object"""
        from .rrd_buffer import Trajectory
        
        # Get initial observation
        init_obs = self._extract_obs(episode['steps'][0])
        trajectory = Trajectory(init_obs)
        
        # Add all steps
        for i, step in enumerate(episode['steps']):
            obs = self._extract_obs(step['observation'])
            obs_next = self._extract_obs(episode['steps'][i+1]['observation']) if i+1 < len(episode['steps']) else obs
            action = step['action']
            reward = float(step['reward'])
            done = bool(step['is_terminal']) if 'is_terminal' in step else (i == len(episode['steps']) - 1)
            
            info = {
                'obs': obs,
                'obs_next': obs_next,
                'acts': action,
                'rews': reward,
                'done': done,
                'real_done': done
            }
            trajectory.store_transition(info)
        
        return trajectory
    
    def _extract_obs(self, observation):
        """Extract observation based on args.libero_obs_mode"""
        if self.args.libero_obs_mode == 'state':
            return observation['state']
        elif self.args.libero_obs_mode == 'image':
            # Process image (resize, normalize, etc.)
            return self._process_image(observation['image'])
        elif self.args.libero_obs_mode == 'combined':
            # Concatenate image features and state
            image_features = self._process_image(observation['image'])
            return np.concatenate([image_features, observation['state']])
    
    def _process_image(self, image):
        """Process image for use in network"""
        # Resize, normalize, etc.
        # Return flattened features or use ConvRewardNet
        pass
    
    def store_transition(self, info):
        """Override: Don't store new transitions (offline dataset)"""
        # For offline RL, we don't add new transitions
        # All data comes from preloaded dataset
        pass
```

### 7. Update Buffer Factory

**File**: `algorithm/replay_buffer/__init__.py`

```python
from .libero_buffer.offline_libero_buffer import OfflineLiberoBuffer

def create_buffer(args):
    if args.env_category == 'libero' and hasattr(args, 'offline') and args.offline:
        return OfflineLiberoBuffer(args)
    elif args.env_category == 'atari':
        return ReplayBuffer_FrameStack(args)
    else:
        return ReplayBuffer_RRD(args)
```

## Reward Structure for LIBERO-10

### LIBERO Reward Format

LIBERO-10 provides:
- **Per-step rewards**: Typically sparse (0.0 most steps, 1.0 on task completion)
- **Episodic structure**: Each episode is a complete task demonstration

### RRD Compatibility

**RRD works with LIBERO rewards as-is**:
- LIBERO rewards are already episodic (sparse, task completion = 1.0)
- RRD will learn to redistribute these sparse rewards
- No need for `ep_rews` wrapper (but you can use it to test)

### If Rewards are Dense

If LIBERO provides dense rewards (e.g., distance to goal at each step):
- RRD will still work
- It will learn to decompose the dense rewards
- May be less beneficial than with sparse rewards, but still functional

## Summary of Required Changes

1. ✅ **Create `envs/normal_libero.py`**: LIBERO environment wrapper
2. ✅ **Update `envs/__init__.py`**: Add LIBERO to environment factory
3. ✅ **Update `common.py`**: Add LIBERO-specific arguments
4. ✅ **Create offline buffer** (if doing offline RL): `algorithm/replay_buffer/libero_buffer/offline_libero_buffer.py`
5. ✅ **Update buffer factory**: Handle LIBERO buffer creation
6. ⚠️ **Handle observations**: Decide on image vs. state vs. combined
7. ⚠️ **Handle multi-camera**: If using images, decide how to combine cameras

## Testing

```bash
# Test with state observations
python scripts/train.py \
    --env=libero-10 \
    --alg=rrd \
    --basis_alg=sac \
    --libero_dataset_path='modified_libero_rlds/libero_10_no_noops/1.0.0' \
    --libero_obs_mode=state \
    --tag='RRD-LIBERO-State'

# Test with image observations
python scripts/train.py \
    --env=libero-10 \
    --alg=rrd \
    --basis_alg=sac \
    --libero_obs_mode=image \
    --tag='RRD-LIBERO-Image'
```

## Key Points

1. **RRD does NOT require sparse rewards** - it works with any episodic reward structure
2. **RRD is most beneficial with sparse/delayed rewards** - but will work with dense rewards too
3. **For offline RL**, you'll need to modify the buffer to load from dataset instead of environment
4. **LIBERO-10 rewards are already sparse** - perfect for RRD's intended use case
5. **Observation format** is the main decision - state, image, or combined features

