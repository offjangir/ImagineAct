import numpy as np
import torch
import torch.nn as nn

def get_vars(module, name_filter=''):
    """Get parameters from a module that match the name filter"""
    if name_filter == '':
        return list(module.parameters())
    params = []
    for name, param in module.named_parameters():
        if name_filter in name:
            params.append(param)
    assert len(params) > 0, f"No parameters found with filter: {name_filter}"
    return params

def get_reg_loss(module):
    """Get regularization loss from module"""
    reg_loss = 0
    for param in module.parameters():
        if hasattr(param, 'regularization_loss'):
            reg_loss += param.regularization_loss
    return reg_loss

class RandomNormal:
    """Normal distribution with mean and log std"""
    def __init__(self, mean, logstd):
        self.raw_logstd = logstd
        # Broadcast logstd if needed
        if len(mean.shape) > len(logstd.shape):
            logstd = mean * 0.0 + logstd
        self.mean = mean
        self.logstd = logstd
        self.std = torch.maximum(torch.exp(logstd), torch.tensor(1e-2, device=logstd.device))

    def log_p(self, x):
        """Compute log probability of x"""
        return torch.sum(
            -0.5*np.log(2.0*np.pi) - self.logstd - 0.5*torch.square((x-self.mean)/self.std),
            dim=-1, keepdim=True
        )

    def entropy(self):
        """Compute entropy"""
        return torch.sum(
            self.logstd + 0.5*np.log(2.0*np.pi*np.e), 
            dim=-1, keepdim=True
        )

    def kl(self, other):
        """Compute KL divergence with another RandomNormal distribution"""
        return torch.sum(
            -0.5 + other.logstd - self.logstd
            + 0.5*torch.square(self.std/other.std)
            + 0.5*torch.square((self.mean-other.mean)/other.std),
            dim=-1, keepdim=True
        )

class Normalizer:
    """Online normalizer for observations"""
    def __init__(self, shape, device='cpu', eps_std=1e-2, norm_clip=5.0):
        self.shape = shape
        self.device = device
        self.eps_std = eps_std
        self.norm_clip = norm_clip

        self.sum = torch.zeros(shape, dtype=torch.float32, device=device)
        self.sum_sqr = torch.zeros(shape, dtype=torch.float32, device=device)
        self.cnt = torch.zeros(1, dtype=torch.float32, device=device)
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.std = torch.ones(shape, dtype=torch.float32, device=device)

    def get_mean(self):
        return self.mean.cpu().numpy()
    
    def get_std(self):
        return self.std.cpu().numpy()

    def normalize(self, inputs):
        """Normalize inputs"""
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float().to(self.device)
        return torch.clamp((inputs - self.mean) / self.std, -self.norm_clip, self.norm_clip)

    def normalize_prefix(self, inputs):
        """Normalize using only the prefix of mean/std"""
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float().to(self.device)
        dim = inputs.shape[-1]
        return torch.clamp(
            (inputs - self.mean[:dim]) / self.std[:dim], 
            -self.norm_clip, self.norm_clip
        )

    def normalize_suffix(self, inputs):
        """Normalize using only the suffix of mean/std"""
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float().to(self.device)
        dim = inputs.shape[-1]
        return torch.clamp(
            (inputs - self.mean[-dim:]) / self.std[-dim:], 
            -self.norm_clip, self.norm_clip
        )

    def update(self, inputs):
        """Update running statistics"""
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.cpu().numpy()
        
        add_sum = np.sum(inputs, axis=0)
        add_sum_sqr = np.sum(np.square(inputs), axis=0)
        add_cnt = inputs.shape[0]

        self.sum += torch.from_numpy(add_sum).float().to(self.device)
        self.sum_sqr += torch.from_numpy(add_sum_sqr).float().to(self.device)
        self.cnt += add_cnt

        self.mean = self.sum / self.cnt
        self.std = torch.maximum(
            torch.tensor(self.eps_std, device=self.device),
            torch.sqrt(self.sum_sqr / self.cnt - torch.square(self.sum / self.cnt))
        )

    def state_dict(self):
        """Get state dict for saving"""
        return {
            'sum': self.sum,
            'sum_sqr': self.sum_sqr,
            'cnt': self.cnt,
            'mean': self.mean,
            'std': self.std
        }

    def load_state_dict(self, state_dict):
        """Load state dict"""
        self.sum = state_dict['sum'].to(self.device)
        self.sum_sqr = state_dict['sum_sqr'].to(self.device)
        self.cnt = state_dict['cnt'].to(self.device)
        self.mean = state_dict['mean'].to(self.device)
        self.std = state_dict['std'].to(self.device)






