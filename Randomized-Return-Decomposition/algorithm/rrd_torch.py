import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.torch_utils import get_vars
from algorithm import basis_algorithm_collection
from algorithm.openvla_reward_net import OpenVLARewardNet, OpenVLAFeatureExtractor

class MLPRewardNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        state_dim = obs_dim * 2 + act_dim  # obs + act + (obs - obs_next)
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, obs, acts, obs_next):
        state = torch.cat([obs, acts, obs - obs_next], dim=-1)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ConvRewardNet(nn.Module):
    """Convolutional reward network for image observations with discrete actions (Atari)"""
    def __init__(self, obs_dims, act_num):
        super().__init__()
        self.obs_dims = obs_dims
        self.act_num = act_num
        
        # Convolutional layers for image input
        # Input: obs and obs_diff concatenated on channel dimension
        in_channels = obs_dims[-1] * 2
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding='same')
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding='same')
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same')
        
        # Calculate flattened size
        # This is a rough estimate, adjust based on actual input size
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, obs_dims[0], obs_dims[1])
            dummy_out = self.conv3(self.conv2(self.conv1(dummy_input)))
            self.flat_size = int(np.prod(dummy_out.shape[1:]))
        
        self.fc_act = nn.Linear(self.flat_size, 512)
        self.fc_out = nn.Linear(512, act_num)
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc_act.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)

    def forward(self, obs, acts, obs_next):
        # Check if we need to flatten batch with sample dimension
        flatten = len(obs.shape) == len(self.obs_dims) + 2
        
        if flatten:
            batch_size, sample_size = obs.shape[:2]
            # Reshape to (batch_size * sample_size, H, W, C)
            obs = obs.reshape(-1, *self.obs_dims)
            obs_next = obs_next.reshape(-1, *self.obs_dims)
            acts = acts.reshape(-1, self.act_num)
        
        # Concatenate obs and obs_diff on channel dimension
        # Assuming obs is (B, H, W, C), convert to (B, C, H, W) for PyTorch
        obs = obs.permute(0, 3, 1, 2)
        obs_next = obs_next.permute(0, 3, 1, 2)
        state = torch.cat([obs, obs - obs_next], dim=1)
        
        # Conv layers
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        
        # FC layers
        x = F.relu(self.fc_act(x))
        r_act = self.fc_out(x)
        
        # Multiply with action (for discrete actions)
        r = torch.sum(r_act * acts, dim=-1, keepdim=True)
        
        if flatten:
            r = r.reshape(batch_size, sample_size, 1)
        
        return r

class ConvRewardNetContinuous(nn.Module):
    """Convolutional reward network for image observations with continuous actions (LIBERO)"""
    def __init__(self, obs_dims, act_dim):
        super().__init__()
        self.obs_dims = obs_dims
        self.act_dim = act_dim
        
        # Convolutional layers for image input
        # Input: obs and obs_diff concatenated on channel dimension
        in_channels = obs_dims[-1] * 2
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, obs_dims[0], obs_dims[1])
            dummy_out = self.conv3(self.conv2(self.conv1(dummy_input)))
            self.flat_size = int(np.prod(dummy_out.shape[1:]))
        
        # Concatenate image features with action
        self.fc1 = nn.Linear(self.flat_size + act_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_out = nn.Linear(256, 1)
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)

    def forward(self, obs, acts, obs_next):
        # Check if we need to flatten batch with sample dimension
        flatten = len(obs.shape) == len(self.obs_dims) + 2
        
        if flatten:
            batch_size, sample_size = obs.shape[:2]
            # Reshape to (batch_size * sample_size, H, W, C)
            obs = obs.reshape(-1, *self.obs_dims)
            obs_next = obs_next.reshape(-1, *self.obs_dims)
            acts = acts.reshape(-1, self.act_dim)
        
        # Concatenate obs and obs_diff on channel dimension
        # Assuming obs is (B, H, W, C), convert to (B, C, H, W) for PyTorch
        obs = obs.permute(0, 3, 1, 2)
        obs_next = obs_next.permute(0, 3, 1, 2)
        state = torch.cat([obs, obs - obs_next], dim=1)
        
        # Conv layers
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)  # Flatten
        
        # Concatenate image features with action
        x = torch.cat([x, acts], dim=-1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        r = self.fc_out(x)  # Scalar reward
        
        if flatten:
            r = r.reshape(batch_size, sample_size, 1)
        
        return r

def RRD(args):
    # Get basis algorithm class name with _torch suffix
    basis_alg_name = args.basis_alg
    if basis_alg_name in basis_algorithm_collection:
        # Import the PyTorch version
        if basis_alg_name == 'sac':
            from .basis_alg.sac_torch import SAC as basis_alg_class
        elif basis_alg_name == 'ddpg':
            from .basis_alg.ddpg_torch import DDPG as basis_alg_class
        elif basis_alg_name == 'td3':
            from .basis_alg.td3_torch import TD3 as basis_alg_class
        elif basis_alg_name == 'dqn':
            from .basis_alg.dqn_torch import DQN as basis_alg_class
        else:
            raise ValueError(f"Unsupported basis algorithm: {basis_alg_name}")
    else:
        raise ValueError(f"Unknown basis algorithm: {basis_alg_name}")
    
    class RandomizedReturnDecomposition(basis_alg_class):
        def __init__(self, args):
            self.args = args
            # Initialize parent class
            basis_alg_class.__init__(self, args)
            
            # Initialize train_info for RRD-specific logging
            # These will be populated by train() method
            self.train_info_r = {}
            # Initialize R_loss in train_info so it gets added to logger items
            # This ensures R_loss appears in the log table
            if 'R_loss' not in self.train_info:
                self.train_info['R_loss'] = 0.0
            if self.args.rrd_bias_correction and 'R_var' not in self.train_info:
                self.train_info['R_var'] = 0.0
            # train_info will be merged from parent and RRD-specific info

        def create_network(self):
            # Create basis algorithm networks
            super().create_network()
            
            # Create reward network
            # Check if using OpenVLA features
            if hasattr(self.args, 'use_openvla_features') and self.args.use_openvla_features:
                # Initialize OpenVLA feature extractor if not already initialized
                if not hasattr(self, 'openvla_extractor'):
                    if not hasattr(self.args, 'openvla_model') or self.args.openvla_model is None:
                        raise ValueError("use_openvla_features=True requires openvla_model to be set in args")
                    if not hasattr(self.args, 'openvla_processor') or self.args.openvla_processor is None:
                        raise ValueError("use_openvla_features=True requires openvla_processor to be set in args")
                    
                    task_desc = getattr(self.args, 'task_description', '')
                    self.openvla_extractor = OpenVLAFeatureExtractor(
                        self.args.openvla_model,
                        self.args.openvla_processor,
                        task_description=task_desc,
                        device=self.device,
                    )
                
                # Get feature dimensions from a dummy forward pass
                # This is a bit hacky but necessary to get the dimensions
                dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
                lang_feat, vis_feat = self.openvla_extractor.extract_features([dummy_img])
                language_feature_dim = lang_feat.shape[-1]
                vision_feature_dim = vis_feat.shape[-1]
                act_dim = self.args.acts_dims[0]
                
                # Expected dimensions:
                # - lang_feat: [batch, seq_len, 4096] where seq_len varies (e.g., 22, 25)
                # - vis_feat: [batch, num_patches, 4096] where num_patches is typically 256
                print(f"[*] OpenVLA feature dimensions:")
                print(f"    Language features: {lang_feat.shape} (embed_dim={language_feature_dim})")
                print(f"    Vision features: {vis_feat.shape} (embed_dim={vision_feature_dim})")
                print(f"    Action dimension: {act_dim}")
                
                hidden_dim = getattr(self.args, 'openvla_reward_hidden_dim', 512)
                num_layers = getattr(self.args, 'openvla_reward_num_layers', 3)
                
                self.reward_net = OpenVLARewardNet(
                    language_feature_dim=language_feature_dim,
                    vision_feature_dim=vision_feature_dim,
                    act_dim=act_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                ).to(self.device)
                print(f"[*] Reward network initialized with hidden_dim={hidden_dim}, num_layers={num_layers}")
            elif len(self.args.obs_dims) == 1:
                # MLP for flat observations
                obs_dim = self.args.obs_dims[0]
                act_dim = self.args.acts_dims[0]
                self.reward_net = MLPRewardNet(obs_dim, act_dim).to(self.device)
            else:
                # Conv for image observations
                if self.args.env_category == 'atari':
                    # Discrete actions (Atari)
                    act_num = self.args.acts_num if hasattr(self.args, 'acts_num') else self.args.acts_dims[0]
                    self.reward_net = ConvRewardNet(self.args.obs_dims, act_num).to(self.device)
                elif self.args.env_category == 'libero':
                    # Continuous actions (LIBERO)
                    act_dim = self.args.acts_dims[0]
                    self.reward_net = ConvRewardNetContinuous(self.args.obs_dims, act_dim).to(self.device)
                else:
                    # Fallback: flatten image and use MLP
                    obs_dim = np.prod(self.args.obs_dims)
                    act_dim = self.args.acts_dims[0]
                    self.reward_net = MLPRewardNet(obs_dim, act_dim).to(self.device)

        def create_optimizer(self):
            # Create basis algorithm optimizers
            super().create_optimizer()
            
            # Create reward optimizer
            self.r_optimizer = torch.optim.Adam(self.reward_net.parameters(), lr=self.args.r_lr)

        def preprocess_batch(self, batch):
            """Convert batch to tensors"""
            # Convert lists to numpy arrays if needed
            def to_numpy(x):
                if isinstance(x, list):
                    return np.array(x)
                return x
            
            # Always process observations (needed for policy training)
            obs = torch.from_numpy(to_numpy(batch['obs'])).float().to(self.device)
            obs_next = torch.from_numpy(to_numpy(batch['obs_next'])).float().to(self.device)
            # Normalize observations
            obs = self.normalize_obs(obs)
            obs_next = self.normalize_obs(obs_next)
            
            # Extract OpenVLA features for reward model if enabled
            if hasattr(self.args, 'use_openvla_features') and self.args.use_openvla_features:
                # Check if features are already pre-extracted (cached in batch)
                if 'lang_feat' in batch and 'vis_feat' in batch and len(batch['lang_feat']) > 0:
                    # Use pre-extracted features from buffer
                    # Stack individual features into batch tensors
                    lang_feat_list = batch['lang_feat']
                    vis_feat_list = batch['vis_feat']
                    
                    # Get next features if available, otherwise use current features shifted
                    if 'lang_feat_next' in batch and len(batch['lang_feat_next']) > 0:
                        lang_feat_next_list = batch['lang_feat_next']
                        vis_feat_next_list = batch['vis_feat_next']
                    else:
                        # Fallback: use shifted current features (for compatibility)
                        lang_feat_next_list = lang_feat_list[1:] + [lang_feat_list[-1]]
                        vis_feat_next_list = vis_feat_list[1:] + [vis_feat_list[-1]]
                    
                    # Stack into batch tensors (handle both numpy arrays and torch tensors)
                    def to_tensor(f):
                        if isinstance(f, np.ndarray):
                            return torch.from_numpy(f)
                        elif isinstance(f, torch.Tensor):
                            return f
                        else:
                            return torch.tensor(f)
                    
                    # Convert to tensors first
                    lang_feat_tensors = [to_tensor(f) for f in lang_feat_list]
                    vis_feat_tensors = [to_tensor(f) for f in vis_feat_list]
                    lang_feat_next_tensors = [to_tensor(f) for f in lang_feat_next_list]
                    vis_feat_next_tensors = [to_tensor(f) for f in vis_feat_next_list]
                    
                    # Features are already globally padded during pre-extraction
                    # Just convert to tensors, move to GPU, and stack (no padding needed!)
                    def stack_features(feat_list):
                        """Stack features that are already globally padded"""
                        if not feat_list:
                            return None
                        
                        # Convert to tensors and move to GPU in one step
                        feat_tensors = []
                        for f in feat_list:
                            if isinstance(f, np.ndarray):
                                feat_tensors.append(torch.from_numpy(f).to(self.device).float())
                            elif isinstance(f, torch.Tensor):
                                feat_tensors.append(f.to(self.device).float())
                            else:
                                feat_tensors.append(torch.tensor(f, device=self.device).float())
                        
                        # Stack all features (they're already the same length from global padding)
                        return torch.stack(feat_tensors)
                    
                    # Stack language and vision features (already globally padded)
                    lang_feat = stack_features(lang_feat_tensors)
                    vis_feat = stack_features(vis_feat_tensors)
                    lang_feat_next = stack_features(lang_feat_next_tensors)
                    vis_feat_next = stack_features(vis_feat_next_tensors)
                else:
                    # Extract features on-the-fly (fallback if not pre-extracted)
                    obs_np = to_numpy(batch['obs'])
                    obs_next_np = to_numpy(batch['obs_next'])
                    
                    # Get task descriptions if available
                    task_descriptions = None
                    if 'task_descriptions' in batch and batch.get('task_descriptions') is not None:
                        task_descriptions = batch['task_descriptions']
                    
                    # Extract features
                    lang_feat, vis_feat = self.openvla_extractor.extract_features(
                        obs_np, task_descriptions=task_descriptions
                    )
                    lang_feat_next, vis_feat_next = self.openvla_extractor.extract_features(
                        obs_next_np, task_descriptions=task_descriptions
                    )
                    
                    # Move to device and ensure float32 dtype (features are converted from bfloat16 in extractor)
                    lang_feat = lang_feat.to(self.device).float()
                    vis_feat = vis_feat.to(self.device).float()
                    lang_feat_next = lang_feat_next.to(self.device).float()
                    vis_feat_next = vis_feat_next.to(self.device).float()
            else:
                lang_feat = vis_feat = lang_feat_next = vis_feat_next = None
            
            acts = torch.from_numpy(to_numpy(batch['acts'])).float().to(self.device)
            rews = torch.from_numpy(to_numpy(batch['rews'])).float().to(self.device)
            done = torch.from_numpy(to_numpy(batch['done'])).float().to(self.device)
            
            # Check if this is a full RRD batch (with rrd_* keys) or just a standard batch
            if 'rrd_obs' in batch:
                # RRD specific inputs
                rrd_obs = to_numpy(batch['rrd_obs'])
                rrd_obs_next = to_numpy(batch['rrd_obs_next'])
                rrd_rews = torch.from_numpy(to_numpy(batch['rrd_rews'])).float().to(self.device)
                
                # Extract OpenVLA features if enabled
                if hasattr(self.args, 'use_openvla_features') and self.args.use_openvla_features:
                    batch_size, sample_size = rrd_obs.shape[:2]
                    
                    # Check if features are already pre-extracted (cached in batch)
                    if 'rrd_lang_feat' in batch and 'rrd_vis_feat' in batch:
                        # Use pre-extracted features from buffer
                        # batch['rrd_lang_feat'] is a list of lists: [batch_item][sample_item] -> feature tensor
                        rrd_lang_feat_list = batch['rrd_lang_feat']
                        rrd_vis_feat_list = batch['rrd_vis_feat']
                        rrd_lang_feat_next_list = batch.get('rrd_lang_feat_next', batch['rrd_lang_feat'])
                        rrd_vis_feat_next_list = batch.get('rrd_vis_feat_next', batch['rrd_vis_feat'])
                        
                        # Stack into [batch_size, sample_size, ...] tensors
                        # Each item in rrd_lang_feat_list is a list of sample_size features
                        # Need to pad features to same length within each sample batch
                        def stack_rrd_features(feat_list_of_lists):
                            """Stack RRD features that are already globally padded: [batch_item][sample_item] -> [batch_size, sample_size, seq_len, embed_dim]"""
                            if not feat_list_of_lists:
                                return None
                            
                            # Stack each batch item's samples (features are already globally padded)
                            stacked_batches = []
                            for item_list in feat_list_of_lists:
                                stacked_samples = []
                                for feat in item_list:
                                    # Convert to tensor and move to GPU in one step
                                    if isinstance(feat, np.ndarray):
                                        feat_tensor = torch.from_numpy(feat).to(self.device).float()
                                    elif isinstance(feat, torch.Tensor):
                                        feat_tensor = feat.to(self.device).float()
                                    else:
                                        feat_tensor = torch.tensor(feat, device=self.device).float()
                                    stacked_samples.append(feat_tensor)
                                stacked_batches.append(torch.stack(stacked_samples))
                            
                            return torch.stack(stacked_batches)
                        
                        rrd_lang_feat = stack_rrd_features(rrd_lang_feat_list)
                        rrd_vis_feat = stack_rrd_features(rrd_vis_feat_list)
                        rrd_lang_feat_next = stack_rrd_features(rrd_lang_feat_next_list)
                        rrd_vis_feat_next = stack_rrd_features(rrd_vis_feat_next_list)
                    else:
                        # Extract features on-the-fly (fallback if not pre-extracted)
                        # Flatten batch and sample dimensions for feature extraction
                        rrd_obs_flat = rrd_obs.reshape(-1, *rrd_obs.shape[2:])
                        rrd_obs_next_flat = rrd_obs_next.reshape(-1, *rrd_obs_next.shape[2:])
                        
                        # Get task descriptions if available
                        # Get task descriptions for RRD batches (prefer rrd_task_descriptions)
                        task_descriptions = None
                        if 'rrd_task_descriptions' in batch and batch.get('rrd_task_descriptions') is not None:
                            task_descriptions = batch['rrd_task_descriptions']
                            if isinstance(task_descriptions, list):
                                # Expand to match flattened batch
                                task_descriptions = [td for td in task_descriptions for _ in range(sample_size)]
                        elif 'task_descriptions' in batch and batch.get('task_descriptions') is not None:
                            # Fallback to regular task_descriptions
                            task_descriptions = batch['task_descriptions']
                            if isinstance(task_descriptions, list):
                                # Expand to match flattened batch
                                task_descriptions = [td for td in task_descriptions for _ in range(sample_size)]
                        
                        # Extract features
                        lang_feat, vis_feat = self.openvla_extractor.extract_features(
                            rrd_obs_flat, task_descriptions=task_descriptions
                        )
                        lang_feat_next, vis_feat_next = self.openvla_extractor.extract_features(
                            rrd_obs_next_flat, task_descriptions=task_descriptions
                        )
                        
                        # Reshape back to [batch_size, sample_size, ...]
                        rrd_lang_feat = lang_feat.reshape(batch_size, sample_size, *lang_feat.shape[1:])
                        rrd_vis_feat = vis_feat.reshape(batch_size, sample_size, *vis_feat.shape[1:])
                        rrd_lang_feat_next = lang_feat_next.reshape(batch_size, sample_size, *lang_feat_next.shape[1:])
                        rrd_vis_feat_next = vis_feat_next.reshape(batch_size, sample_size, *vis_feat_next.shape[1:])
                        
                        # Move to device and ensure float32 dtype (features are converted from bfloat16 in extractor)
                        rrd_lang_feat = rrd_lang_feat.to(self.device).float()
                        rrd_vis_feat = rrd_vis_feat.to(self.device).float()
                        rrd_lang_feat_next = rrd_lang_feat_next.to(self.device).float()
                        rrd_vis_feat_next = rrd_vis_feat_next.to(self.device).float()
                else:
                    # Standard processing: convert to tensors
                    rrd_obs = torch.from_numpy(rrd_obs).float().to(self.device)
                    rrd_obs_next = torch.from_numpy(rrd_obs_next).float().to(self.device)
                    rrd_lang_feat = rrd_vis_feat = rrd_lang_feat_next = rrd_vis_feat_next = None
                
                # Handle actions - convert to one-hot for Atari, keep continuous for others
                if self.args.env_category == 'atari':
                    # Discrete actions: convert to one-hot
                    rrd_acts = torch.from_numpy(to_numpy(batch['rrd_acts'])).long().to(self.device)
                    batch_size, sample_size = rrd_acts.shape
                    act_num = self.args.acts_num if hasattr(self.args, 'acts_num') else self.args.acts_dims[0]
                    rrd_acts_onehot = torch.zeros(batch_size, sample_size, act_num, device=self.device)
                    rrd_acts_onehot.scatter_(2, rrd_acts.unsqueeze(-1), 1.0)
                    rrd_acts = rrd_acts_onehot
                else:
                    # Continuous actions (MuJoCo, LIBERO): keep as float
                    rrd_acts = torch.from_numpy(to_numpy(batch['rrd_acts'])).float().to(self.device)
                
                # Normalize observations only if not using OpenVLA features
                if not (hasattr(self.args, 'use_openvla_features') and self.args.use_openvla_features):
                    rrd_obs = self.normalize_obs(rrd_obs)
                    rrd_obs_next = self.normalize_obs(rrd_obs_next)
                
                batch_tensors = {
                    'obs': obs,
                    'obs_next': obs_next,
                    'acts': acts,
                    'rews': rews,
                    'done': done,
                    'rrd_obs': rrd_obs,
                    'rrd_obs_next': rrd_obs_next,
                    'rrd_acts': rrd_acts,
                    'rrd_rews': rrd_rews
                }
                
                # Add OpenVLA features if available
                if hasattr(self.args, 'use_openvla_features') and self.args.use_openvla_features:
                    # Regular batch features
                    batch_tensors['lang_feat'] = lang_feat
                    batch_tensors['vis_feat'] = vis_feat
                    batch_tensors['lang_feat_next'] = lang_feat_next
                    batch_tensors['vis_feat_next'] = vis_feat_next
                    # RRD batch features
                    batch_tensors['rrd_lang_feat'] = rrd_lang_feat
                    batch_tensors['rrd_vis_feat'] = rrd_vis_feat
                    batch_tensors['rrd_lang_feat_next'] = rrd_lang_feat_next
                    batch_tensors['rrd_vis_feat_next'] = rrd_vis_feat_next
                
                # Get episode length if available (for scaling predictions to full episodic return)
                if 'rrd_ep_length' in batch:
                    batch_tensors['rrd_ep_length'] = torch.from_numpy(to_numpy(batch['rrd_ep_length'])).float().to(self.device)
                
                if self.args.rrd_bias_correction and 'rrd_var_coef' in batch:
                    batch_tensors['rrd_var_coef'] = torch.from_numpy(to_numpy(batch['rrd_var_coef'])).float().to(self.device)
                
                return batch_tensors
            else:
                # Standard batch without RRD keys
                # Store OpenVLA features as instance variables for access in train methods
                if hasattr(self.args, 'use_openvla_features') and self.args.use_openvla_features:
                    self._current_lang_feat = lang_feat
                    self._current_vis_feat = vis_feat
                    self._current_lang_feat_next = lang_feat_next
                    self._current_vis_feat_next = vis_feat_next
                
                # Return tuple to match parent class interface
                # Note: obs/obs_next are still returned for policy training
                return obs, obs_next, acts, rews, done

        def train_r(self, batch):
            """Train reward network"""
            batch_tensors = self.preprocess_batch(batch)
            
            self.reward_net.train()
            
            # Predict rewards for each sample in the RRD batch
            if hasattr(self.args, 'use_openvla_features') and self.args.use_openvla_features:
                # Use OpenVLA features
                rrd_rews_pred = self.reward_net(
                    batch_tensors['rrd_lang_feat'],
                    batch_tensors['rrd_vis_feat'],
                    batch_tensors['rrd_acts'],
                    batch_tensors['rrd_lang_feat_next'],
                    batch_tensors['rrd_vis_feat_next'],
                )
            else:
                # Use standard observations
                rrd_rews_pred = self.reward_net(
                    batch_tensors['rrd_obs'],
                    batch_tensors['rrd_acts'],
                    batch_tensors['rrd_obs_next']
                )
            
            # According to RRD paper (https://arxiv.org/pdf/2111.13485):
            # "the reward model is trained to predict the episodic return from a random subsequence"
            # 
            # Mathematical formulation:
            # - Sample subsequence of k steps from episode of length T
            # - Predict per-step rewards: r_1, ..., r_k
            # - Scale to predict full episodic return: sum(r_1, ..., r_k) * (T/k) ≈ R_ep
            # - Compare predicted full return to actual episodic return R_ep
            
            # rrd_rews_pred shape: (batch_size, sample_size, 1)
            sample_size = rrd_rews_pred.shape[1]
            rrd_sum_subsequence = torch.sum(rrd_rews_pred, dim=1)  # Sum over subsequence
            
            # Scale to predict full episodic return
            if 'rrd_ep_length' in batch_tensors:
                # Use actual episode length: scale by (T / k)
                ep_lengths = batch_tensors['rrd_ep_length']
                rrd = rrd_sum_subsequence * (ep_lengths / sample_size)
            else:
                # Fallback: if episode length not available, use average (surrogate objective)
                # This matches the original implementation's approach
                # avg_pred * T ≈ R_ep, so we compare averages
                rrd = torch.mean(rrd_rews_pred, dim=1)
                # Note: In this case, target should be average per step, not episodic return
                # But for backward compatibility, we'll use the provided target as-is
            
            # Reward loss: compare predicted episodic return to actual episodic return
            r_loss = F.mse_loss(rrd, batch_tensors['rrd_rews'])
            
            # Bias correction (variance penalty)
            if self.args.rrd_bias_correction:
                assert self.args.rrd_sample_size > 1
                n = self.args.rrd_sample_size
                r_var_single = torch.sum(
                    torch.square(rrd_rews_pred - torch.mean(rrd_rews_pred, dim=1, keepdim=True)),
                    dim=1
                ) / (n - 1)
                r_var = torch.mean(r_var_single * batch_tensors['rrd_var_coef'] / n)
                r_total_loss = r_loss - r_var
                
                info = {
                    'R_loss': r_loss.item(),
                    'R_var': r_var.item()
                }
            else:
                r_total_loss = r_loss
                info = {'R_loss': r_loss.item()}
            
            # Update reward network
            self.r_optimizer.zero_grad()
            r_total_loss.backward()
            self.r_optimizer.step()
            
            return info

        def train_q(self, batch):
            """Train Q-value with predicted rewards"""
            # Preprocess batch (this stores OpenVLA features in instance vars if enabled)
            obs, obs_next, acts, rews, done = self.preprocess_batch(batch)
            
            # Get predicted rewards for regular transitions
            with torch.no_grad():
                self.reward_net.eval()
                if hasattr(self.args, 'use_openvla_features') and self.args.use_openvla_features:
                    # Use stored OpenVLA features
                    rews_pred = self.reward_net(
                        self._current_lang_feat,
                        self._current_vis_feat,
                        acts,
                        self._current_lang_feat_next,
                        self._current_vis_feat_next,
                    )
                else:
                    rews_pred = self.reward_net(obs, acts, obs_next)
            
            # Replace rewards in batch with predicted rewards
            # Convert back to numpy for parent class
            batch_with_pred_rews = {
                'obs': obs.cpu().numpy(),
                'obs_next': obs_next.cpu().numpy(),
                'acts': acts.cpu().numpy(),
                'rews': rews_pred.cpu().numpy(),
                'done': done.cpu().numpy()
            }
            
            # Train Q-value using parent class method
            q_info = super().train_q(batch_with_pred_rews)
            
            # Also train reward network
            r_info = self.train_r(batch)
            
            return {**q_info, **r_info}

        def train(self, batch):
            """Train both policy and Q-value (with reward network)"""
            # Preprocess batch (this stores OpenVLA features in instance vars if enabled)
            obs, obs_next, acts, rews, done = self.preprocess_batch(batch)
            
            # Get predicted rewards
            with torch.no_grad():
                self.reward_net.eval()
                if hasattr(self.args, 'use_openvla_features') and self.args.use_openvla_features:
                    # Use stored OpenVLA features
                    rews_pred = self.reward_net(
                        self._current_lang_feat,
                        self._current_vis_feat,
                        acts,
                        self._current_lang_feat_next,
                        self._current_vis_feat_next,
                    )
                else:
                    rews_pred = self.reward_net(obs, acts, obs_next)
            
            # Create batch with predicted rewards for parent class
            batch_with_pred_rews = {
                'obs': obs.cpu().numpy(),
                'obs_next': obs_next.cpu().numpy(),
                'acts': acts.cpu().numpy(),
                'rews': rews_pred.cpu().numpy(),
                'done': done.cpu().numpy()
            }
            
            # Train policy and Q-value
            pi_info = super().train_pi(batch_with_pred_rews)
            q_info = super().train_q(batch_with_pred_rews)
            
            # Train reward network
            r_info = self.train_r(batch)
            
            # Update train_info for logging
            self.train_info_r = r_info
            self.train_info_q = {**self.train_info_q, **r_info}
            self.train_info = {**self.train_info, **r_info}
            
            return {**pi_info, **q_info, **r_info}

    return RandomizedReturnDecomposition(args)






