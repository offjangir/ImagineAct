import copy
import numpy as np
from ..mujoco_buffer.rrd_buffer import Trajectory

class OfflineLiberoBuffer:
    """Offline replay buffer for LIBERO-10 RLDS dataset"""
    
    def __init__(self, args):
        self.args = args
        self.ep_counter = 0
        self.step_counter = 0
        self.buffer_size = self.args.buffer_size
        
        # Load episodes from LIBERO environment
        # The environment has already loaded all episodes
        self.ep = []
        self.ram_idx = []
        self.length = 0
        self.head_idx = 0
        
        # Load episodes from dataset
        self._load_episodes_from_env(args)
        
        # Update step_counter to reflect buffer size
        self.step_counter = self.length
        
        print(f"Offline buffer loaded: {len(self.ep)} episodes, {self.length} total steps")
        
        # Pre-extract OpenVLA features if enabled (will be called after agent initialization)
        self.features_pre_extracted = False
    
    def _load_episodes_from_env(self, args):
        """Load all episodes from LIBERO environment"""
        # Try to reuse existing environment if available (to avoid loading dataset twice)
        env = None
        if hasattr(args, 'env_instance') and args.env_instance is not None:
            env = args.env_instance
        else:
            # Fallback: create new environment if not available
            from envs import make_env
            env = make_env(args)
        
        # Get max trajectories limit (if specified)
        max_trajectories = getattr(args, 'libero_max_trajectories', None)
        if max_trajectories is not None and max_trajectories > 0:
            print(f"Limiting to {max_trajectories} trajectories (out of {len(env.episodes) if hasattr(env, 'episodes') else 'unknown'} total)")
        
        # Access episodes from environment
        if hasattr(env, 'episodes'):
            episode_count = 0
            for episode_data in env.episodes:
                # Limit number of trajectories if specified
                if max_trajectories is not None and episode_count >= max_trajectories:
                    break
                
                # Convert RLDS episode to Trajectory format
                trajectory = self._episode_to_trajectory(episode_data, env)
                self.ep.append(trajectory)
                self.ep_counter += 1
                self.length += trajectory.length
                # Add episode index for each step
                for _ in range(trajectory.length):
                    self.ram_idx.append(self.ep_counter - 1)
                episode_count += 1
        else:
            raise ValueError("LIBERO environment does not have episodes attribute")
    
    def _episode_to_trajectory(self, episode_data, env):
        """Convert RLDS episode to Trajectory object"""
        # Get initial observation
        init_obs = env._extract_obs(episode_data['steps'][0])
        trajectory = Trajectory(init_obs)
        
        # Store task description from episode
        task_description = episode_data.get('task_description', '')
        trajectory.task_description = task_description
        
        # Add all steps
        for i in range(len(episode_data['steps'])):
            step = episode_data['steps'][i]
            
            # Extract observation
            obs = env._extract_obs(step)
            
            # Get next observation
            if i + 1 < len(episode_data['steps']):
                obs_next = env._extract_obs(episode_data['steps'][i + 1])
            else:
                obs_next = obs  # Terminal state
            
            # Extract action and reward
            action = step['action']
            if isinstance(action, (list, tuple)):
                action = np.array(action, dtype=np.float32)
            elif hasattr(action, 'numpy'):
                action = action.numpy().astype(np.float32)
            else:
                action = np.array(action, dtype=np.float32)
            
            reward = float(step['reward'])
            done = bool(step['is_terminal'] or step['is_last'])
            
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
    
    def store_transition(self, info):
        """Override: Don't store new transitions (offline dataset)"""
        # For offline RL, we don't add new transitions
        # All data comes from preloaded dataset
        pass
    
    def sample_batch(self, batch_size=-1, rrd_batch_size=-1, rrd_sample_size=-1):
        """Sample batch from offline dataset"""
        if batch_size == -1:
            batch_size = self.args.batch_size
        if rrd_batch_size == -1:
            rrd_batch_size = self.args.rrd_batch_size
        if rrd_sample_size == -1:
            rrd_sample_size = self.args.rrd_sample_size
        
        batch = dict(
            obs=[], obs_next=[], acts=[], rews=[], done=[],
            rrd_obs=[], rrd_obs_next=[], rrd_acts=[], rrd_rews=[], rrd_ep_length=[]
        )
        if self.args.rrd_bias_correction:
            batch['rrd_var_coef'] = []
        
        # Store task descriptions for each sample
        batch['task_descriptions'] = []
        batch['rrd_task_descriptions'] = []
        
        # Initialize feature lists if features are pre-extracted
        if self.features_pre_extracted:
            batch['lang_feat'] = []
            batch['vis_feat'] = []
            batch['lang_feat_next'] = []
            batch['vis_feat_next'] = []
            batch['rrd_lang_feat'] = []
            batch['rrd_vis_feat'] = []
            batch['rrd_lang_feat_next'] = []
            batch['rrd_vis_feat_next'] = []
        
        # Sample regular transitions
        for i in range(batch_size):
            idx = self.ram_idx[np.random.randint(self.length)] - self.head_idx
            info = self.ep[idx].sample()
            for key in info.keys():
                if key not in batch:
                    batch[key] = []
                batch[key].append(info[key])
            # Add task description from trajectory
            task_desc = getattr(self.ep[idx], 'task_description', '')
            batch['task_descriptions'].append(task_desc)
        
        # Sample RRD batches (random subsequences)
        for i in range(rrd_batch_size // rrd_sample_size):
            idx = self.ram_idx[np.random.randint(self.length)] - self.head_idx
            info = self.ep[idx].rrd_sample(rrd_sample_size, store_coef=self.args.rrd_bias_correction)
            for key in info.keys():
                if key not in batch:
                    batch[key] = []
                batch[key].append(info[key])
            # Add task description from trajectory (same for all samples in subsequence)
            task_desc = getattr(self.ep[idx], 'task_description', '')
            batch['rrd_task_descriptions'].append(task_desc)
        
        return batch
    
    def _get_features_cache_path(self):
        """Get path to features cache file"""
        import os
        # Create cache directory based on dataset path and model checkpoint
        dataset_path = getattr(self.args, 'libero_dataset_path', 'unknown')
        checkpoint = getattr(self.args, 'openvla_checkpoint', 'unknown')
        max_traj = getattr(self.args, 'libero_max_trajectories', None)
        
        # Create a hash from dataset path and checkpoint
        import hashlib
        cache_key = f"{dataset_path}_{checkpoint}_{max_traj}_{len(self.ep)}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        
        cache_dir = os.path.join("log", "feature_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"openvla_features_{cache_hash}.pkl")
        return cache_file
    
    def _load_features_from_cache(self):
        """Load pre-extracted features from pickle file"""
        import os
        import pickle
        import torch
        
        cache_file = self._get_features_cache_path()
        
        if not os.path.exists(cache_file):
            return False
        
        print(f"\n{'='*80}")
        print(f"Loading pre-extracted features from cache: {cache_file}")
        print(f"{'='*80}\n")
        
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Verify cache matches current buffer
            if cached_data.get('num_trajectories') != len(self.ep):
                print(f"Warning: Cache has {cached_data.get('num_trajectories')} trajectories, but buffer has {len(self.ep)}. Re-extracting...")
                return False
            
            # Load features into trajectories
            features_dict = cached_data['features']
            loaded_count = 0
            for traj_idx, trajectory in enumerate(self.ep):
                if traj_idx in features_dict:
                    traj_features = features_dict[traj_idx]
                    # Convert numpy arrays to torch tensors
                    trajectory.lang_feat = []
                    trajectory.vis_feat = []
                    for f in traj_features['lang_feat']:
                        if isinstance(f, np.ndarray):
                            trajectory.lang_feat.append(torch.from_numpy(f))
                        elif isinstance(f, torch.Tensor):
                            trajectory.lang_feat.append(f)
                        else:
                            trajectory.lang_feat.append(torch.tensor(f))
                    for f in traj_features['vis_feat']:
                        if isinstance(f, np.ndarray):
                            trajectory.vis_feat.append(torch.from_numpy(f))
                        elif isinstance(f, torch.Tensor):
                            trajectory.vis_feat.append(f)
                        else:
                            trajectory.vis_feat.append(torch.tensor(f))
                    trajectory.features_extracted = True
                    loaded_count += 1
            
            # Check if features are already globally padded (from cache metadata)
            needs_global_padding = False
            if 'global_max_lang_len' in cached_data and cached_data['global_max_lang_len'] is not None:
                self.global_max_lang_len = cached_data['global_max_lang_len']
                self.global_max_vis_len = cached_data.get('global_max_vis_len', None)
                print(f"✓ Cache indicates global padding: lang_seq_len={self.global_max_lang_len}, vis_seq_len={self.global_max_vis_len}")
                # Verify features are actually padded to this length
                needs_global_padding = True  # Will verify and pad if needed
            else:
                # Old cache file - need to find global max and pad
                needs_global_padding = True
                print("⚠ Cache file doesn't have global padding metadata. Will pad features now...")
            
            # Apply global padding if needed
            if needs_global_padding:
                print(f"\n{'='*80}")
                print("Applying global padding to cached features...")
                print(f"{'='*80}\n")
                
                import torch
                from tqdm import tqdm
                
                # Find global max sequence lengths across ALL features
                global_max_lang_len = 0
                global_max_vis_len = 0
                
                print("Finding global max sequence lengths...")
                for trajectory in tqdm(self.ep, desc="Scanning features"):
                    if trajectory.features_extracted:
                        for lang_feat in trajectory.lang_feat:
                            if lang_feat is not None:
                                global_max_lang_len = max(global_max_lang_len, lang_feat.shape[0])
                        for vis_feat in trajectory.vis_feat:
                            if vis_feat is not None:
                                global_max_vis_len = max(global_max_vis_len, vis_feat.shape[0])
                
                print(f"Global max lengths: lang={global_max_lang_len}, vis={global_max_vis_len}")
                print("Padding all features to global max lengths...")
                
                # Pad all features to global max lengths
                for trajectory in tqdm(self.ep, desc="Padding features"):
                    if trajectory.features_extracted:
                        # Pad language features
                        padded_lang_feat = []
                        for lang_feat in trajectory.lang_feat:
                            if lang_feat is not None:
                                current_len = lang_feat.shape[0]
                                if current_len < global_max_lang_len:
                                    pad_size = global_max_lang_len - current_len
                                    padding = torch.zeros(pad_size, *lang_feat.shape[1:], dtype=lang_feat.dtype)
                                    padded_feat = torch.cat([lang_feat, padding], dim=0)
                                else:
                                    padded_feat = lang_feat
                                padded_lang_feat.append(padded_feat)
                            else:
                                padded_lang_feat.append(None)
                        
                        # Pad vision features
                        padded_vis_feat = []
                        for vis_feat in trajectory.vis_feat:
                            if vis_feat is not None:
                                current_len = vis_feat.shape[0]
                                if current_len < global_max_vis_len:
                                    pad_size = global_max_vis_len - current_len
                                    padding = torch.zeros(pad_size, *vis_feat.shape[1:], dtype=vis_feat.dtype)
                                    padded_feat = torch.cat([vis_feat, padding], dim=0)
                                else:
                                    padded_feat = vis_feat
                                padded_vis_feat.append(padded_feat)
                            else:
                                padded_vis_feat.append(None)
                        
                        # Replace with padded features
                        trajectory.lang_feat = padded_lang_feat
                        trajectory.vis_feat = padded_vis_feat
                
                # Store global max lengths
                self.global_max_lang_len = global_max_lang_len
                self.global_max_vis_len = global_max_vis_len
                
                print(f"✓ All features padded to: lang_seq_len={global_max_lang_len}, vis_seq_len={global_max_vis_len}")
                print(f"{'='*80}\n")
            
            self.features_pre_extracted = True
            print(f"✓ Loaded features for {loaded_count} trajectories from cache!")
            print(f"{'='*80}\n")
            return True
        except Exception as e:
            print(f"Error loading cache: {e}")
            print("Re-extracting features...")
            return False
    
    def _save_features_to_cache(self):
        """Save pre-extracted features to pickle file"""
        import os
        import pickle
        import torch
        
        cache_file = self._get_features_cache_path()
        
        print(f"\n{'='*80}")
        print(f"Saving features to cache: {cache_file}")
        print(f"{'='*80}\n")
        
        try:
            # Collect features from all trajectories
            features_dict = {}
            for traj_idx, trajectory in enumerate(self.ep):
                if trajectory.features_extracted:
                    # Convert tensors to numpy for serialization
                    lang_feat_np = [f.numpy() if isinstance(f, torch.Tensor) else f for f in trajectory.lang_feat]
                    vis_feat_np = [f.numpy() if isinstance(f, torch.Tensor) else f for f in trajectory.vis_feat]
                    features_dict[traj_idx] = {
                        'lang_feat': lang_feat_np,
                        'vis_feat': vis_feat_np
                    }
            
            cached_data = {
                'num_trajectories': len(self.ep),
                'features': features_dict,
                'dataset_path': getattr(self.args, 'libero_dataset_path', 'unknown'),
                'checkpoint': getattr(self.args, 'openvla_checkpoint', 'unknown'),
                'global_max_lang_len': getattr(self, 'global_max_lang_len', None),
                'global_max_vis_len': getattr(self, 'global_max_vis_len', None),
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
            
            file_size_mb = os.path.getsize(cache_file) / (1024 * 1024)
            print(f"✓ Features saved to cache! File size: {file_size_mb:.2f} MB")
            print(f"{'='*80}\n")
            return True
        except Exception as e:
            print(f"Error saving cache: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def pre_extract_openvla_features(self, openvla_extractor, batch_size=64, num_workers=256):
        """
        Pre-extract OpenVLA features for all observations in the buffer.
        This should be called once after agent initialization to cache features.
        
        Optimized with parallel episode processing and batched step processing.
        Features are saved to a pickle file and loaded on subsequent runs.
        
        Args:
            openvla_extractor: OpenVLAFeatureExtractor instance
            batch_size: Batch size for feature extraction within each episode (to avoid OOM)
            num_workers: Number of parallel workers for processing episodes (default: 256)
        """
        if self.features_pre_extracted:
            print("Features already pre-extracted, skipping...")
            return
        
        if not (hasattr(self.args, 'use_openvla_features') and self.args.use_openvla_features):
            print("OpenVLA features not enabled, skipping pre-extraction...")
            return
        
        # Try to load from cache first
        if self._load_features_from_cache():
            return
        
        print(f"\n{'='*80}")
        print("Pre-extracting OpenVLA features for all observations...")
        print(f"This may take a while for {self.length} total steps...")
        print(f"Using {num_workers} parallel workers, batch size {batch_size} per episode")
        print(f"Note: If this is too slow, consider skipping pre-extraction and using on-the-fly extraction during training")
        print(f"{'='*80}\n")
        
        import torch
        from tqdm import tqdm
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        # Initialize feature storage for all trajectories
        trajectories_to_process = []
        for traj_idx, trajectory in enumerate(self.ep):
            if not trajectory.features_extracted:
                trajectory.lang_feat = [None] * len(trajectory.ep['obs'])
                trajectory.vis_feat = [None] * len(trajectory.ep['obs'])
                trajectories_to_process.append((traj_idx, trajectory))
        
        print(f"Processing {len(trajectories_to_process)} trajectories with {num_workers} workers...")
        
        # Lock for thread-safe access to extractor (GPU model can be shared across threads)
        extractor_lock = threading.Lock()
        
        def process_trajectory(traj_data):
            """Process a single trajectory: extract features for all its observations in batches"""
            traj_idx, trajectory = traj_data
            
            # Collect all observations for this trajectory
            all_obs = []
            task_descriptions = []
            
            for obs_idx in range(len(trajectory.ep['obs'])):
                all_obs.append(trajectory.ep['obs'][obs_idx])
                task_descriptions.append(trajectory.task_description)
            
            # Extract features in batches for this trajectory
            lang_feat_list = []
            vis_feat_list = []
            
            for batch_start in range(0, len(all_obs), batch_size):
                batch_end = min(batch_start + batch_size, len(all_obs))
                batch_obs = all_obs[batch_start:batch_end]
                batch_task_descs = task_descriptions[batch_start:batch_end]
                
                # Extract features (thread-safe with lock)
                with extractor_lock:
                    lang_feat_batch, vis_feat_batch = openvla_extractor.extract_features(
                        batch_obs, task_descriptions=batch_task_descs
                    )
                
                # Convert to CPU and split into individual features
                lang_feat_batch = lang_feat_batch.cpu()
                vis_feat_batch = vis_feat_batch.cpu()
                
                # Split batch into individual features
                for i in range(batch_end - batch_start):
                    lang_feat_list.append(lang_feat_batch[i].clone())
                    vis_feat_list.append(vis_feat_batch[i].clone())
            
            # Store features in trajectory
            trajectory.lang_feat = lang_feat_list
            trajectory.vis_feat = vis_feat_list
            trajectory.features_extracted = True
            
            return traj_idx, len(all_obs)
        
        # Process trajectories in parallel
        completed = 0
        total_obs_processed = 0
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all trajectory processing tasks
            future_to_traj = {
                executor.submit(process_trajectory, traj_data): traj_data 
                for traj_data in trajectories_to_process
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(trajectories_to_process), desc="Extracting features") as pbar:
                for future in as_completed(future_to_traj):
                    try:
                        traj_idx, num_obs = future.result()
                        completed += 1
                        total_obs_processed += num_obs
                        pbar.update(1)
                        pbar.set_postfix({'episodes': f"{completed}/{len(trajectories_to_process)}", 'obs': total_obs_processed})
                    except Exception as e:
                        traj_data = future_to_traj[future]
                        print(f"\nError processing trajectory {traj_data[0]}: {e}")
                        import traceback
                        traceback.print_exc()
        
        # Now pad all features globally to max length (for faster training)
        print(f"\n{'='*80}")
        print("Applying global padding to all features...")
        print(f"{'='*80}\n")
        
        # Find global max sequence lengths across ALL features
        global_max_lang_len = 0
        global_max_vis_len = 0
        
        print("Finding global max sequence lengths...")
        for trajectory in tqdm(self.ep, desc="Scanning features"):
            if trajectory.features_extracted:
                for lang_feat in trajectory.lang_feat:
                    if lang_feat is not None:
                        global_max_lang_len = max(global_max_lang_len, lang_feat.shape[0])
                for vis_feat in trajectory.vis_feat:
                    if vis_feat is not None:
                        global_max_vis_len = max(global_max_vis_len, vis_feat.shape[0])
        
        print(f"Global max lengths: lang={global_max_lang_len}, vis={global_max_vis_len}")
        print("Padding all features to global max lengths...")
        
        # Pad all features to global max lengths
        for trajectory in tqdm(self.ep, desc="Padding features"):
            if trajectory.features_extracted:
                # Pad language features
                padded_lang_feat = []
                for lang_feat in trajectory.lang_feat:
                    if lang_feat is not None:
                        current_len = lang_feat.shape[0]
                        if current_len < global_max_lang_len:
                            pad_size = global_max_lang_len - current_len
                            padding = torch.zeros(pad_size, *lang_feat.shape[1:], dtype=lang_feat.dtype)
                            padded_feat = torch.cat([lang_feat, padding], dim=0)
                        else:
                            padded_feat = lang_feat
                        padded_lang_feat.append(padded_feat)
                    else:
                        padded_lang_feat.append(None)
                
                # Pad vision features
                padded_vis_feat = []
                for vis_feat in trajectory.vis_feat:
                    if vis_feat is not None:
                        current_len = vis_feat.shape[0]
                        if current_len < global_max_vis_len:
                            pad_size = global_max_vis_len - current_len
                            padding = torch.zeros(pad_size, *vis_feat.shape[1:], dtype=vis_feat.dtype)
                            padded_feat = torch.cat([vis_feat, padding], dim=0)
                        else:
                            padded_feat = vis_feat
                        padded_vis_feat.append(padded_feat)
                    else:
                        padded_vis_feat.append(None)
                
                # Replace with padded features
                trajectory.lang_feat = padded_lang_feat
                trajectory.vis_feat = padded_vis_feat
        
        # Store global max lengths for reference
        self.global_max_lang_len = global_max_lang_len
        self.global_max_vis_len = global_max_vis_len
        
        self.features_pre_extracted = True
        
        # Save features to cache
        self._save_features_to_cache()
        
        print(f"\n{'='*80}")
        print("✓ Feature pre-extraction and global padding complete!")
        print(f"  Extracted features for {len(trajectories_to_process)} trajectories, {total_obs_processed} total observations")
        print(f"  All features padded to: lang_seq_len={global_max_lang_len}, vis_seq_len={global_max_vis_len}")
        print(f"{'='*80}\n")

