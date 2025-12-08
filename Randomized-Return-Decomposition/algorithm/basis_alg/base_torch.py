import numpy as np
import torch
import torch.nn as nn
from utils.torch_utils import Normalizer

class Base:
    def __init__(self, args):
        self.args = args
        # Handle cuda argument - default to True if CUDA is available
        use_cuda = getattr(args, 'cuda', torch.cuda.is_available())
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
        # PyTorch doesn't use graph/sess like TensorFlow, but we add them for compatibility
        self.graph = None
        self.sess = None
        self.create_model()

    def create_network(self):
        raise NotImplementedError

    def create_optimizer(self):
        raise NotImplementedError

    def create_model(self):
        self.create_network()
        self.create_optimizer()
        self.create_normalizer()
        self.init_network()

    def create_normalizer(self):
        if self.args.obs_normalization:
            self.obs_normalizer = Normalizer(self.args.obs_dims, device=self.device)
        else:
            self.obs_normalizer = None

    def normalize_obs(self, obs):
        """Normalize observation"""
        if self.obs_normalizer is not None:
            return self.obs_normalizer.normalize(obs)
        if isinstance(obs, np.ndarray):
            return torch.from_numpy(obs).float().to(self.device)
        return obs

    def init_network(self):
        """Initialize target networks"""
        if hasattr(self, 'target_init'):
            self.target_init()

    def normalizer_update(self, batch):
        """Update observation normalizer"""
        if self.args.obs_normalization:
            obs_concat = np.concatenate([batch['obs'], batch['obs_next']], axis=0)
            self.obs_normalizer.update(obs_concat)

    def target_update(self):
        """Update target networks"""
        if hasattr(self, '_target_update'):
            self._target_update()

    def save_model(self, save_path):
        """Save model checkpoint"""
        state_dict = {
            'networks': {}
        }
        
        # Save all network modules
        for name, module in self.__dict__.items():
            if isinstance(module, nn.Module):
                state_dict['networks'][name] = module.state_dict()
        
        # Save normalizer if it exists
        if self.obs_normalizer is not None:
            state_dict['normalizer'] = self.obs_normalizer.state_dict()
        
        torch.save(state_dict, save_path)

    def load_model(self, load_path):
        """Load model checkpoint"""
        state_dict = torch.load(load_path, map_location=self.device)
        
        # Load network modules
        for name, params in state_dict['networks'].items():
            if hasattr(self, name) and isinstance(getattr(self, name), nn.Module):
                getattr(self, name).load_state_dict(params)
        
        # Load normalizer if it exists
        if 'normalizer' in state_dict and self.obs_normalizer is not None:
            self.obs_normalizer.load_state_dict(state_dict['normalizer'])






