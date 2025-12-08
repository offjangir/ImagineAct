# RRD Training Data and Outputs

This document explains what the RRD model is trained on and what it outputs.

## Overview

RRD (Randomized Reward Decomposition) consists of two main components:
1. **Reward Network**: Learns to predict per-step rewards
2. **Policy/Q-Networks**: Standard RL agent (SAC, DDPG, etc.) trained with predicted rewards

## What RRD is Trained On

### Training Data for Reward Network

The reward network is trained on **random subsequences** from complete episodes:

#### Input Data Structure

For each training batch, the reward network receives:

1. **RRD Batch** (for training the reward network):
   - `rrd_obs`: Observations from random subsequences `(batch_size, sample_size, obs_dim)`
   - `rrd_acts`: Actions from random subsequences `(batch_size, sample_size, act_dim)`
   - `rrd_obs_next`: Next observations `(batch_size, sample_size, obs_dim)`
   - `rrd_rews`: **Episodic return** (sum of all rewards in the full episode) `(batch_size, 1)`

2. **Regular Batch** (for training Q-network):
   - `obs`, `acts`, `obs_next`: Standard transitions
   - `rews`: Environment rewards (replaced with predicted rewards during training)

#### Key Insight: Randomized Subsequences

The core idea of RRD is to train the reward network on **random subsequences** from episodes:

```python
# From rrd_buffer.py
def rrd_sample(self, sample_size, store_coef=False):
    # Sample random indices from episode
    idx = np.random.choice(self.length, sample_size, replace=(sample_size>self.length))
    
    # Get observations, actions at those indices
    info = {
        'rrd_obs': self.ep['obs'][idx],
        'rrd_obs_next': self.ep['obs'][idx+1],
        'rrd_acts': self.ep['acts'][idx],
        'rrd_rews': [self.sum_rews/self.length]  # Episodic return
    }
    return info
```

**Example:**
- Full episode: 1000 timesteps
- Sample size: 64 (randomly selected timesteps)
- Target: Total episodic return (sum of all 1000 rewards)
- Task: Predict the full episode return from just 64 random state-action pairs

### Training Objective

The reward network is trained to minimize:

```
L_RRD = MSE(predicted_return, actual_episodic_return)
```

Where:
- `predicted_return = mean(reward_net(obs_i, act_i, obs_next_i))` for all sampled `i`
- `actual_episodic_return = sum(all_rewards_in_episode)`

#### Standard RRD Loss

```python
# Predict rewards for each sample
rrd_rews_pred = reward_net(rrd_obs, rrd_acts, rrd_obs_next)  # (batch, sample_size, 1)

# Average predictions
rrd = torch.mean(rrd_rews_pred, dim=1)  # (batch, 1)

# Loss: MSE between predicted and actual episodic return
r_loss = F.mse_loss(rrd, rrd_rews)  # rrd_rews is episodic return
```

#### Bias-Corrected RRD Loss (L-RD)

When `--rrd_bias_correction=True`:

```python
# Additional variance penalty term
r_var = variance_of_predictions * variance_coefficient
r_total_loss = r_loss - r_var  # Subtract variance (encourage consistency)
```

This encourages the reward network to make consistent predictions across different random subsequences.

## What RRD Outputs

### Reward Network Outputs

The reward network outputs **per-step reward predictions**:

1. **For Training (RRD batches)**:
   - Input: Random subsequences `(obs, action, obs_next)`
   - Output: Predicted rewards for each sample `(batch_size, sample_size, 1)`
   - These are averaged to predict the episodic return

2. **For Q-Network Training**:
   - Input: Regular transitions `(obs, action, obs_next)`
   - Output: **Per-step reward prediction** `(batch_size, 1)`
   - This replaces the environment reward in Q-learning

### Code Flow

```python
# 1. Train reward network on random subsequences
def train_r(self, batch):
    # Predict rewards for random subsequences
    rrd_rews_pred = self.reward_net(
        batch['rrd_obs'],      # Random observations
        batch['rrd_acts'],     # Random actions
        batch['rrd_obs_next']  # Random next observations
    )
    # Shape: (batch_size, sample_size, 1)
    
    # Average to get predicted episodic return
    rrd = torch.mean(rrd_rews_pred, dim=1)  # (batch_size, 1)
    
    # Compare with actual episodic return
    r_loss = F.mse_loss(rrd, batch['rrd_rews'])  # batch['rrd_rews'] is episodic return
    
    return {'R_loss': r_loss.item()}

# 2. Use reward network to predict rewards for Q-learning
def train_q(self, batch):
    # Predict per-step rewards
    with torch.no_grad():
        self.reward_net.eval()
        rews_pred = self.reward_net(
            batch['obs'],       # Regular observations
            batch['acts'],      # Regular actions
            batch['obs_next']   # Regular next observations
        )
    # Shape: (batch_size, 1) - per-step reward prediction
    
    # Replace environment rewards with predicted rewards
    batch_with_pred_rews = {
        'obs': batch['obs'],
        'obs_next': batch['obs_next'],
        'acts': batch['acts'],
        'rews': rews_pred,  # Use predicted rewards instead of env rewards
        'done': batch['done']
    }
    
    # Train Q-network using predicted rewards
    q_info = super().train_q(batch_with_pred_rews)
    
    return q_info
```

## Training Process Summary

### Step 1: Data Collection
```
Agent interacts with environment:
  obs → action → obs_next, reward, done
```

### Step 2: Episode Storage
```
Complete episodes stored in replay buffer:
  Episode = [(obs_0, act_0, obs_1, rew_0),
             (obs_1, act_1, obs_2, rew_1),
             ...
             (obs_T, act_T, obs_T+1, rew_T)]
  Episodic return = sum(rew_0, rew_1, ..., rew_T)
```

### Step 3: RRD Batch Creation
```
For each RRD batch:
  - Sample random episode
  - Sample random subsequence (e.g., 64 timesteps from 1000)
  - Target: Full episodic return
```

### Step 4: Reward Network Training
```
Input: Random subsequence (obs, action, obs_next)
Output: Predicted rewards for each sample
Average: Mean of predicted rewards
Loss: MSE(predicted_return, actual_episodic_return)
```

### Step 5: Q-Network Training
```
Input: Regular transitions (obs, action, obs_next)
Reward Network: Predicts per-step rewards
Q-Network: Trained with predicted rewards (not environment rewards)
```

## Key Properties

### 1. Scalability

- **Traditional approach**: Requires processing entire trajectory (expensive for long episodes)
- **RRD approach**: Only processes random subsequences (scalable to long episodes)

### 2. Reward Redistribution

- Environment provides **sparse rewards** (only at episode end or key events)
- RRD learns to **redistribute** these rewards to individual steps
- Enables better credit assignment in long-horizon tasks

### 3. Self-Supervised Learning

- No external reward labels needed
- Uses only episodic returns from environment
- Learns to decompose returns into per-step rewards

## Example: Ant-v2 Environment

### Episode Structure
```
Episode length: ~1000 timesteps
Environment rewards: Sparse (only at certain events)
Episodic return: Sum of all rewards (e.g., 1000.0)
```

### RRD Training
```
Sample: 64 random timesteps from episode
Input: (obs_i, action_i, obs_next_i) for i in [random 64 indices]
Target: 1000.0 (full episodic return)
Task: Predict 1000.0 from just 64 state-action pairs
```

### Reward Network Output
```
For each transition (obs, action, obs_next):
  Output: Predicted reward (e.g., 0.5, 1.2, 0.8, ...)
  
Sum of predictions ≈ 1000.0 (episodic return)
```

### Q-Network Training
```
Instead of using sparse environment rewards:
  Use dense predicted rewards from reward network
  
This enables better learning in long-horizon tasks
```

## Summary

**What RRD is trained on:**
- Random subsequences of state-action pairs from episodes
- Target: Episodic return (sum of all rewards in episode)
- Objective: Predict full episode return from random subsequences

**What RRD outputs:**
- Per-step reward predictions for any (obs, action, obs_next) tuple
- These predicted rewards replace environment rewards in Q-learning
- Enables dense reward signals for better credit assignment

**Key Innovation:**
- Learns to decompose sparse episodic returns into dense per-step rewards
- Scalable to long-horizon tasks by using random subsequences
- Self-supervised: only needs episodic returns, not per-step reward labels

