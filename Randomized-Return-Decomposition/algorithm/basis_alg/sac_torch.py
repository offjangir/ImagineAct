import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.torch_utils import get_vars, RandomNormal
from .ddpg_torch import DDPG

class MLPStochasticPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, act_dim * 2)  # mean and logstd
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        act_dim = x.shape[-1] // 2
        mean = x[..., :act_dim]
        logstd = torch.clamp(x[..., act_dim:], -20.0, 2.0)
        
        return RandomNormal(mean, logstd)

class MLPQValueSAC(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, obs, acts):
        x = torch.cat([obs, acts], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SAC(DDPG):
    def __init__(self, args):
        self.args = args
        # Call Base.__init__ directly to avoid DDPG's network creation
        from .base_torch import Base
        Base.__init__(self, args)
        
        # Define train_info and step_info for logging compatibility
        self.train_info_pi = {}
        self.train_info_q = {}
        self.train_info = {}
        self.step_info = {}

    def create_network(self):
        obs_dim = np.prod(self.args.obs_dims)
        act_dim = self.args.acts_dims[0]
        
        # Policy network
        self.policy = MLPStochasticPolicy(obs_dim, act_dim).to(self.device)
        
        # Two Q-value networks (for double Q-learning)
        self.q_value_1 = MLPQValueSAC(obs_dim, act_dim).to(self.device)
        self.q_value_2 = MLPQValueSAC(obs_dim, act_dim).to(self.device)
        
        # Target Q-value networks
        self.q_value_1_target = MLPQValueSAC(obs_dim, act_dim).to(self.device)
        self.q_value_2_target = MLPQValueSAC(obs_dim, act_dim).to(self.device)
        
        # Initialize target networks
        self.q_value_1_target.load_state_dict(self.q_value_1.state_dict())
        self.q_value_2_target.load_state_dict(self.q_value_2.state_dict())
        
        # Temperature parameter (alpha)
        self.log_alpha = nn.Parameter(
            torch.tensor(np.log(self.args.alpha_init), dtype=torch.float32, device=self.device)
        )

    def create_optimizer(self):
        self.pi_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.args.pi_lr)
        self.q_optimizer = torch.optim.Adam(
            list(self.q_value_1.parameters()) + list(self.q_value_2.parameters()), 
            lr=self.args.q_lr
        )
        
        if self.args.alpha_lr > 0:
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.args.alpha_lr)
        else:
            self.alpha_optimizer = None

    def target_init(self):
        """Initialize target networks"""
        self.q_value_1_target.load_state_dict(self.q_value_1.state_dict())
        self.q_value_2_target.load_state_dict(self.q_value_2.state_dict())

    def _target_update(self):
        """Soft update of target networks"""
        with torch.no_grad():
            for param, target_param in zip(self.q_value_1.parameters(), self.q_value_1_target.parameters()):
                target_param.data.copy_(
                    self.args.polyak * target_param.data + (1.0 - self.args.polyak) * param.data
                )
            for param, target_param in zip(self.q_value_2.parameters(), self.q_value_2_target.parameters()):
                target_param.data.copy_(
                    self.args.polyak * target_param.data + (1.0 - self.args.polyak) * param.data
                )

    def get_action_and_log_prob(self, pi_dist, noise):
        """Sample action from policy and compute log probability"""
        pi_sample = pi_dist.mean + noise * pi_dist.std
        pi_act = torch.tanh(pi_sample)
        # Compute log probability with change of variables for tanh
        pi_log_p = pi_dist.log_p(pi_sample) - torch.sum(
            torch.log(1 - torch.square(pi_act) + 1e-6), dim=-1, keepdim=True
        )
        return pi_act, pi_log_p

    def step(self, obs, explore=False, test_info=False):
        """Take a step in the environment"""
        self.policy.eval()
        with torch.no_grad():
            obs_tensor = self.normalize_obs(np.array([obs]))
            pi_dist = self.policy(obs_tensor)
            
            if explore:
                noise = torch.randn_like(pi_dist.mean)
            else:
                noise = torch.zeros_like(pi_dist.mean)
            
            pi_act, _ = self.get_action_and_log_prob(pi_dist, noise)
            action = pi_act.cpu().numpy()[0]
        
        if test_info:
            with torch.no_grad():
                q1 = self.q_value_1(obs_tensor, pi_act)
                q2 = self.q_value_2(obs_tensor, pi_act)
                q_pi = torch.minimum(q1, q2)
                alpha = torch.exp(self.log_alpha)
                info = {
                    'Q_average': q_pi.item(),
                    'Pi_step_std': pi_dist.std.mean().item()
                }
                self.step_info = info
            return action, info
        return action

    def train(self, batch):
        """Train both policy and Q-value"""
        pi_info = self.train_pi(batch)
        q_info = self.train_q(batch)
        # Update train_info for logging
        self.train_info_pi = pi_info
        self.train_info_q = q_info
        self.train_info = {**pi_info, **q_info}
        return self.train_info

    def train_pi(self, batch):
        """Train policy"""
        obs, obs_next, acts, rews, done = self.preprocess_batch(batch)
        
        self.policy.train()
        self.q_value_1.eval()
        self.q_value_2.eval()
        
        # Sample actions from current policy
        pi_dist = self.policy(obs)
        noise = torch.randn_like(pi_dist.mean)
        pi_act, pi_log_p = self.get_action_and_log_prob(pi_dist, noise)
        
        # Compute Q-values
        q_pi_1 = self.q_value_1(obs, pi_act)
        q_pi_2 = self.q_value_2(obs, pi_act)
        q_pi = torch.minimum(q_pi_1, q_pi_2)
        
        # Policy loss
        alpha = torch.exp(self.log_alpha).detach()
        pi_loss = torch.mean(-q_pi + alpha * pi_log_p)
        
        # Update policy
        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self.pi_optimizer.step()
        
        # Update temperature (alpha)
        if self.alpha_optimizer is not None:
            alpha_loss = -self.log_alpha * (pi_log_p.detach() + np.prod(self.args.acts_dims)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        return {'Pi_loss': pi_loss.item()}

    def train_q(self, batch):
        """Train Q-value"""
        obs, obs_next, acts, rews, done = self.preprocess_batch(batch)
        
        self.policy.eval()
        self.q_value_1.train()
        self.q_value_2.train()
        
        # Compute target Q-value
        with torch.no_grad():
            pi_next_dist = self.policy(obs_next)
            noise_next = torch.randn_like(pi_next_dist.mean)
            pi_next_act, pi_next_log_p = self.get_action_and_log_prob(pi_next_dist, noise_next)
            
            q_next_1 = self.q_value_1_target(obs_next, pi_next_act)
            q_next_2 = self.q_value_2_target(obs_next, pi_next_act)
            q_next = torch.minimum(q_next_1, q_next_2)
            
            alpha = torch.exp(self.log_alpha)
            q_next = q_next - alpha * pi_next_log_p
            q_target = rews + (1.0 - done) * self.args.gamma * q_next
        
        # Q-value losses
        q_pred_1 = self.q_value_1(obs, acts)
        q_pred_2 = self.q_value_2(obs, acts)
        q_loss = F.mse_loss(q_pred_1, q_target) + F.mse_loss(q_pred_2, q_target)
        
        # Update Q-values
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        return {'Q_loss': q_loss.item()}






