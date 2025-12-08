import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.torch_utils import get_vars, get_reg_loss
from .base_torch import Base

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, act_dim)
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class MLPQValue(nn.Module):
    def __init__(self, obs_dim, act_dim, l2_reg=0.0):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 400)
        self.fc2 = nn.Linear(400 + act_dim, 300)
        self.fc3 = nn.Linear(300, 1)
        self.l2_reg = l2_reg
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, obs, acts):
        x = F.relu(self.fc1(obs))
        x = torch.cat([x, acts], dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DDPG(Base):
    def __init__(self, args):
        self.args = args
        super().__init__(args)
        
        # Define train_info and step_info for logging compatibility
        # These will be populated by the train() and step() methods
        self.train_info_pi = {}
        self.train_info_q = {}
        self.train_info = {}
        self.step_info = {}

    def create_network(self):
        obs_dim = np.prod(self.args.obs_dims)
        act_dim = self.args.acts_dims[0]
        
        # Main networks
        self.policy = MLPPolicy(obs_dim, act_dim).to(self.device)
        self.q_value = MLPQValue(obs_dim, act_dim, self.args.q_reg).to(self.device)
        
        # Target networks
        self.policy_target = MLPPolicy(obs_dim, act_dim).to(self.device)
        self.q_value_target = MLPQValue(obs_dim, act_dim).to(self.device)
        
        # Initialize target networks
        self.policy_target.load_state_dict(self.policy.state_dict())
        self.q_value_target.load_state_dict(self.q_value.state_dict())

    def create_optimizer(self):
        self.pi_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.args.pi_lr)
        self.q_optimizer = torch.optim.Adam(self.q_value.parameters(), lr=self.args.q_lr)

    def target_init(self):
        """Initialize target networks"""
        self.policy_target.load_state_dict(self.policy.state_dict())
        self.q_value_target.load_state_dict(self.q_value.state_dict())

    def _target_update(self):
        """Soft update of target networks"""
        with torch.no_grad():
            for param, target_param in zip(self.policy.parameters(), self.policy_target.parameters()):
                target_param.data.copy_(
                    self.args.polyak * target_param.data + (1.0 - self.args.polyak) * param.data
                )
            for param, target_param in zip(self.q_value.parameters(), self.q_value_target.parameters()):
                target_param.data.copy_(
                    self.args.polyak * target_param.data + (1.0 - self.args.polyak) * param.data
                )

    def step(self, obs, explore=False, test_info=False):
        """Take a step in the environment"""
        self.policy.eval()
        with torch.no_grad():
            obs_tensor = self.normalize_obs(np.array([obs]))
            action = self.policy(obs_tensor).cpu().numpy()[0]
        
        # Add exploration noise
        if explore:
            action += np.random.normal(0, self.args.std_act, size=self.args.acts_dims)
        action = np.clip(action, -1, 1)
        
        if test_info:
            with torch.no_grad():
                q_value = self.q_value(obs_tensor, torch.from_numpy(action[None]).float().to(self.device))
                info = {'Q_average': q_value.item()}
                self.step_info = info
            return action, info
        return action

    def preprocess_batch(self, batch):
        """Convert batch to tensors"""
        # Convert lists to numpy arrays if needed
        def to_numpy(x):
            if isinstance(x, list):
                return np.array(x)
            return x
        
        obs = torch.from_numpy(to_numpy(batch['obs'])).float().to(self.device)
        obs_next = torch.from_numpy(to_numpy(batch['obs_next'])).float().to(self.device)
        acts = torch.from_numpy(to_numpy(batch['acts'])).float().to(self.device)
        rews = torch.from_numpy(to_numpy(batch['rews'])).float().to(self.device)
        done = torch.from_numpy(to_numpy(batch['done'])).float().to(self.device)
        
        # Normalize observations
        obs = self.normalize_obs(obs)
        obs_next = self.normalize_obs(obs_next)
        
        return obs, obs_next, acts, rews, done

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
        self.q_value.eval()
        
        # Policy loss
        pi_acts = self.policy(obs)
        q_pi = self.q_value(obs, pi_acts)
        pi_loss = -q_pi.mean()
        
        # Update policy
        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self.pi_optimizer.step()
        
        return {'Pi_loss': pi_loss.item()}

    def train_q(self, batch):
        """Train Q-value"""
        obs, obs_next, acts, rews, done = self.preprocess_batch(batch)
        
        self.policy.eval()
        self.q_value.train()
        
        # Compute target Q-value
        with torch.no_grad():
            pi_next = self.policy_target(obs_next)
            q_next = self.q_value_target(obs_next, pi_next)
            q_target = rews + (1.0 - done) * self.args.gamma * q_next
        
        # Q-value loss
        q_pred = self.q_value(obs, acts)
        q_loss = F.mse_loss(q_pred, q_target)
        
        # Add L2 regularization if specified
        if self.args.q_reg > 0:
            l2_reg = sum(torch.sum(param ** 2) for param in self.q_value.parameters())
            q_reg_loss = self.args.q_reg * l2_reg
            total_loss = q_loss + q_reg_loss
        else:
            q_reg_loss = torch.tensor(0.0)
            total_loss = q_loss
        
        # Update Q-value
        self.q_optimizer.zero_grad()
        total_loss.backward()
        self.q_optimizer.step()
        
        return {
            'Q_loss': q_loss.item(),
            'Q_reg_loss': q_reg_loss.item() if isinstance(q_reg_loss, torch.Tensor) else q_reg_loss
        }






