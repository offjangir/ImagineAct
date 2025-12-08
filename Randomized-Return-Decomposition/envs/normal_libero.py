import numpy as np
import cv2
import os
from utils.os_utils import remove_color

try:
    import tensorflow_datasets as tfds
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: tensorflow_datasets not available. LIBERO environment will not work.")

class LiberoNormalEnv:
    """LIBERO environment wrapper for offline RLDS dataset"""
    
    def __init__(self, args):
        self.args = args
        
        if not TF_AVAILABLE:
            raise ImportError("tensorflow_datasets is required for LIBERO environment")
        
        # Load RLDS dataset
        dataset_path = getattr(args, 'libero_dataset_path', 
                              '/data/kmirakho/JustImagine/modified_libero_rlds/libero_10_no_noops/1.0.0')
        
        # Print dataset information
        print("=" * 80)
        print(f"LIBERO Dataset Configuration:")
        print(f"  Dataset Path: {dataset_path}")
        print(f"  Image Size: {getattr(args, 'libero_image_size', 256)}x{getattr(args, 'libero_image_size', 256)}")
        print(f"  Using workspace camera images only (wrist camera ignored)")
        print("=" * 80)
        
        # Configure TensorFlow to not use GPU (to avoid conflicts with PyTorch)
        tf.config.set_visible_devices([], 'GPU')
        # Disable TensorFlow eager execution optimizations that can cause segfaults
        tf.config.run_functions_eagerly(False)
        
        # Load dataset using tensorflow_datasets
        print(f"\nLoading LIBERO-10 dataset from: {dataset_path}")
        
        # Check if path exists
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        
        # List available files in dataset directory
        dataset_files = [f for f in os.listdir(dataset_path) if f.endswith('.tfrecord')]
        print(f"  Found {len(dataset_files)} TFRecord file(s) in dataset directory")
        if len(dataset_files) > 0:
            print(f"  Example files: {dataset_files[:3]}{'...' if len(dataset_files) > 3 else ''}")
        
        builder = tfds.builder_from_directory(dataset_path)
        dataset = builder.as_dataset(split='train')
        
        # Convert to list for easier iteration (for offline RL)
        # Use as_numpy_iterator to avoid TensorFlow graph issues
        print("\nConverting RLDS dataset to numpy format...")
        self.episodes = []
        episode_count = 0
        
        # Use as_numpy_iterator for safer conversion
        try:
            dataset_iter = dataset.as_numpy_iterator()
        except:
            # Fallback to regular iteration
            dataset_iter = iter(dataset)
        
        for episode in dataset_iter:
            episode_dict = {}
            episode_dict['steps'] = []
            
            # Extract language instruction/task description from episode
            # LIBERO stores language instructions at episode level or step level
            language_instruction = None
            if 'language_instruction' in episode:
                lang_inst = episode['language_instruction']
                if hasattr(lang_inst, 'numpy'):
                    language_instruction = lang_inst.numpy()
                elif isinstance(lang_inst, bytes):
                    language_instruction = lang_inst.decode('utf-8')
                else:
                    language_instruction = str(lang_inst)
            elif 'task' in episode and isinstance(episode['task'], dict):
                if 'language_instruction' in episode['task']:
                    lang_inst = episode['task']['language_instruction']
                    if hasattr(lang_inst, 'numpy'):
                        language_instruction = lang_inst.numpy()
                    elif isinstance(lang_inst, bytes):
                        language_instruction = lang_inst.decode('utf-8')
                    else:
                        language_instruction = str(lang_inst)
            
            # Store task description at episode level
            episode_dict['task_description'] = language_instruction if language_instruction else ""
            
            # Convert episode steps to numpy
            # Handle both tf.data.Dataset (sequence) and numpy array formats
            steps_data = episode['steps']
            if hasattr(steps_data, '__iter__') and not isinstance(steps_data, (list, np.ndarray)):
                # It's a tf.data.Dataset sequence, convert to list first
                try:
                    steps_list = list(steps_data.as_numpy_iterator())
                except:
                    steps_list = list(steps_data)
            else:
                steps_list = steps_data
            
            for step in steps_list:
                # Handle image (may be encoded JPEG bytes)
                image = step['observation']['image']
                if isinstance(image, bytes):
                    # Decode JPEG
                    image_np = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
                    if image_np is not None:
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                    else:
                        # Fallback: create dummy image
                        image_np = np.zeros((256, 256, 3), dtype=np.uint8)
                elif hasattr(image, 'numpy'):
                    image_np = image.numpy()
                else:
                    image_np = np.array(image)
                
                # Handle state
                state = step['observation']['state']
                if hasattr(state, 'numpy'):
                    state_np = state.numpy()
                else:
                    state_np = np.array(state, dtype=np.float32)
                
                # Handle action
                action = step['action']
                if hasattr(action, 'numpy'):
                    action_np = action.numpy()
                else:
                    action_np = np.array(action, dtype=np.float32)
                
                # Handle reward
                reward = step['reward']
                if hasattr(reward, 'numpy'):
                    reward_val = float(reward.numpy())
                else:
                    reward_val = float(reward)
                
                # Handle flags
                is_terminal = step.get('is_terminal', False)
                if hasattr(is_terminal, 'numpy'):
                    is_terminal = bool(is_terminal.numpy())
                else:
                    is_terminal = bool(is_terminal)
                
                is_last = step.get('is_last', False)
                if hasattr(is_last, 'numpy'):
                    is_last = bool(is_last.numpy())
                else:
                    is_last = bool(is_last)
                
                # Check for language instruction at step level (some datasets store it per step)
                step_language = None
                if 'language_instruction' in step:
                    lang_inst = step['language_instruction']
                    if hasattr(lang_inst, 'numpy'):
                        step_language = lang_inst.numpy()
                    elif isinstance(lang_inst, bytes):
                        step_language = lang_inst.decode('utf-8')
                    else:
                        step_language = str(lang_inst)
                
                # Use step-level language if available, otherwise use episode-level
                if step_language:
                    if not episode_dict.get('task_description'):
                        episode_dict['task_description'] = step_language
                
                step_dict = {
                    'observation': {
                        'image': image_np,
                        'state': state_np,
                    },
                    'action': action_np,
                    'reward': reward_val,
                    'is_terminal': is_terminal,
                    'is_last': is_last,
                }
                episode_dict['steps'].append(step_dict)
            
            self.episodes.append(episode_dict)
            episode_count += 1
            if episode_count % 100 == 0:
                print(f"  Loaded {episode_count} episodes...")
        
        print(f"\n✓ Successfully loaded {len(self.episodes)} episodes from LIBERO-10 dataset")
        print(f"  Total steps across all episodes: {sum(len(ep['steps']) for ep in self.episodes)}")
        print("=" * 80)
        
        # Determine observation and action dimensions from first episode
        first_obs = self._extract_obs(self.episodes[0]['steps'][0])
        first_action = self.episodes[0]['steps'][0]['action']
        
        # Observation space: workspace image (256, 256, 3) -> will be resized/processed
        # For RRD, we'll use the image directly
        self.observation_space = type('Space', (), {
            'shape': first_obs.shape,
            'dtype': first_obs.dtype
        })()
        
        # Action space: 7-DoF continuous actions
        self.action_space = type('Space', (), {
            'shape': first_action.shape if hasattr(first_action, 'shape') else (len(first_action),),
            'dtype': first_action.dtype if hasattr(first_action, 'dtype') else type(first_action),
            'low': np.array([-1.0] * 7),
            'high': np.array([1.0] * 7)
        })()
        
        self.acts_dims = list(self.action_space.shape)
        self.obs_dims = list(self.observation_space.shape)
        
        # Current episode tracking (for offline RL, we iterate through dataset)
        self.current_episode_idx = 0
        self.current_step_idx = 0
        self.current_episode = None
        
        # Episode statistics
        self.steps = 0
        self.rewards = 0.0
        
        self.env_info = {
            'Steps': self.process_info_steps,
            'Rewards@green': self.process_info_rewards
        }
        
        self.reset()
    
    def _extract_obs(self, step):
        """Extract workspace image observation"""
        image = step['observation']['image']
        
        # Handle different image formats
        if isinstance(image, bytes):
            # Decode JPEG if needed
            image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif hasattr(image, 'numpy'):
            image = image.numpy()
        
        # Ensure image is uint8 and in correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Resize if needed (LIBERO images are 256x256, but we might want smaller for efficiency)
        target_size = getattr(self.args, 'libero_image_size', 84)  # Default to 84x84 like Atari
        if target_size != image.shape[0] or target_size != image.shape[1]:
            image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1] for neural networks
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def reset(self):
        """Reset to a random episode from dataset"""
        # Sample random episode for offline training
        self.current_episode_idx = np.random.randint(len(self.episodes))
        self.current_episode = self.episodes[self.current_episode_idx]
        self.current_step_idx = 0
        self.steps = 0
        self.rewards = 0.0
        
        # Get initial observation
        obs = self._extract_obs(self.current_episode['steps'][0])
        self.last_obs = obs
        return obs.copy()
    
    def step(self, action):
        """Step through dataset episode (offline - action is ignored)"""
        # In offline RL, we're replaying demonstrations
        # The action doesn't affect the trajectory, but we still need to return next state
        
        self.current_step_idx += 1
        
        if self.current_step_idx >= len(self.current_episode['steps']):
            # Episode done
            done = True
            obs_next = self.last_obs  # Terminal state
            reward = 0.0
        else:
            step_data = self.current_episode['steps'][self.current_step_idx]
            obs_next = self._extract_obs(step_data)
            reward = step_data['reward']
            done = step_data['is_terminal'] or step_data['is_last']
        
        info = self.process_info(obs_next, reward, {})
        self.last_obs = obs_next
        self.rewards += reward
        self.steps += 1
        
        return obs_next.copy(), reward, done, info
    
    def get_obs(self):
        """Get current observation"""
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
    
    def render(self, mode='human'):
        """Render current observation (for visualization)"""
        if mode == 'rgb_array':
            return (self.last_obs * 255).astype(np.uint8)
        return None

