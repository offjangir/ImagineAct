"""
One-time script to enhance the features cache with actions, rewards, and done flags.
This allows the dataset to load everything from cache without needing RLDS.
"""
import os
import pickle
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

# Configure TensorFlow to not use GPU
tf.config.set_visible_devices([], 'GPU')

def enhance_features_cache(
    features_cache_path: str,
    rlds_dataset_path: str,
    output_cache_path: str = None
):
    """
    Enhance features cache with actions, rewards, and done flags from RLDS.
    
    Args:
        features_cache_path: Path to existing features cache
        rlds_dataset_path: Path to RLDS dataset
        output_cache_path: Path to save enhanced cache (if None, overwrites original)
    """
    if output_cache_path is None:
        output_cache_path = features_cache_path
    
    print(f"Loading existing features cache from: {features_cache_path}")
    with open(features_cache_path, 'rb') as f:
        cached_data = pickle.load(f)
    
    features_dict = cached_data.get('features', {})
    print(f"Found {len(features_dict)} trajectories in cache")
    
    # Check if already enhanced
    sample_traj = features_dict.get(0, {})
    if 'actions' in sample_traj and 'rewards' in sample_traj:
        print("Cache already contains actions and rewards. Skipping enhancement.")
        return
    
    # Load RLDS episodes
    print(f"\nLoading RLDS episodes from: {rlds_dataset_path}")
    if not os.path.exists(rlds_dataset_path):
        raise FileNotFoundError(f"RLDS dataset path not found: {rlds_dataset_path}")
    
    builder = tfds.builder_from_directory(rlds_dataset_path)
    dataset = builder.as_dataset(split='train')
    
    # Convert to list of episodes
    episodes = []
    try:
        dataset_iter = dataset.as_numpy_iterator()
    except:
        dataset_iter = iter(dataset)
    
    print("Processing RLDS episodes...")
    episode_count = 0
    for episode in dataset_iter:
        episode_count += 1
        if episode_count % 50 == 0:
            print(f"  Processed {episode_count} episodes...")
        
        episode_dict = {}
        episode_dict['steps'] = []
        
        # Extract task description
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
        
        episode_dict['task_description'] = language_instruction if language_instruction else ""
        
        # Convert steps
        steps_data = episode['steps']
        if hasattr(steps_data, '__iter__') and not isinstance(steps_data, (list, np.ndarray)):
            try:
                steps_list = list(steps_data.as_numpy_iterator())
            except:
                steps_list = list(steps_data)
        else:
            steps_list = steps_data
        
        for step in steps_list:
            # Extract action
            action = step['action']
            if hasattr(action, 'numpy'):
                action_np = action.numpy()
            else:
                action_np = np.array(action, dtype=np.float32)
            
            # Extract reward
            reward = step['reward']
            if hasattr(reward, 'numpy'):
                reward_val = float(reward.numpy())
            else:
                reward_val = float(reward)
            
            # Extract flags
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
            
            episode_dict['steps'].append({
                'action': action_np,
                'reward': reward_val,
                'is_terminal': is_terminal,
                'is_last': is_last,
            })
        
        episodes.append(episode_dict)
    
    print(f"Loaded {len(episodes)} episodes from RLDS dataset")
    
    # Enhance features cache with RLDS data
    print("\nEnhancing features cache...")
    enhanced_count = 0
    skipped_count = 0
    
    for traj_idx, episode in enumerate(episodes):
        if traj_idx not in features_dict:
            skipped_count += 1
            continue
        
        steps = episode['steps']
        num_steps = len(steps)
        
        # Extract actions, rewards, and done flags
        actions = []
        rewards = []
        done_flags = []
        
        for step in steps:
            actions.append(step['action'])
            rewards.append(step['reward'])
            done_flags.append(float(step['is_terminal'] or step['is_last']))
        
        # Add to features dict
        features_dict[traj_idx]['actions'] = np.array(actions, dtype=np.float32)
        features_dict[traj_idx]['rewards'] = np.array(rewards, dtype=np.float32)
        features_dict[traj_idx]['done'] = np.array(done_flags, dtype=np.float32)
        features_dict[traj_idx]['task_description'] = episode.get('task_description', '')
        
        enhanced_count += 1
        if enhanced_count % 50 == 0:
            print(f"  Enhanced {enhanced_count} trajectories...")
    
    print(f"Enhanced {enhanced_count} trajectories, skipped {skipped_count}")
    
    # Update cached data
    cached_data['features'] = features_dict
    cached_data['enhanced'] = True  # Flag to indicate cache is enhanced
    
    # Save enhanced cache
    print(f"\nSaving enhanced cache to: {output_cache_path}")
    os.makedirs(os.path.dirname(output_cache_path), exist_ok=True)
    with open(output_cache_path, 'wb') as f:
        pickle.dump(cached_data, f)
    
    file_size_mb = os.path.getsize(output_cache_path) / (1024 * 1024)
    print(f"✓ Enhanced cache saved! File size: {file_size_mb:.2f} MB")
    print(f"✓ Cache now contains: features, actions, rewards, done flags, task descriptions")
    print(f"✓ Dataset can now load everything from cache without RLDS!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhance features cache with RLDS data")
    parser.add_argument(
        "--features_cache_path",
        type=str,
        default="/data/kmirakho/JustImagine/Randomized-Return-Decomposition/log/feature_cache/openvla_features_8fabf830.pkl",
        help="Path to existing features cache"
    )
    parser.add_argument(
        "--rlds_dataset_path",
        type=str,
        default="/data/kmirakho/JustImagine/modified_libero_rlds/libero_goal_no_noops/1.0.0",
        help="Path to RLDS dataset"
    )
    parser.add_argument(
        "--output_cache_path",
        type=str,
        default=None,
        help="Path to save enhanced cache (if None, overwrites original)"
    )
    
    args = parser.parse_args()
    
    enhance_features_cache(
        features_cache_path=args.features_cache_path,
        rlds_dataset_path=args.rlds_dataset_path,
        output_cache_path=args.output_cache_path
    )

