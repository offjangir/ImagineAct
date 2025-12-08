# RRD for Image Observations

This document explains how Randomized Reward Decomposition (RRD) works with image-based observations, such as Atari games.

## Overview

RRD adapts to image observations by using a **Convolutional Reward Network** (`ConvRewardNet`) instead of the MLP reward network used for vector observations. The core RRD algorithm remains the same, but the reward network architecture changes to process spatial image data.

## Architecture: ConvRewardNet

### Network Structure

For image observations, RRD uses a convolutional neural network with the following architecture:

```
Input: [obs, obs_diff] concatenated on channel dimension
  ↓
Conv1: 32 filters, 8×8 kernel, stride 4, padding 'same'
  ↓ ReLU
Conv2: 64 filters, 4×4 kernel, stride 2, padding 'same'
  ↓ ReLU
Conv3: 64 filters, 3×3 kernel, stride 1, padding 'same'
  ↓ ReLU
Flatten
  ↓
FC1: 512 units
  ↓ ReLU
FC2: act_num units (one per action)
  ↓
Multiply with one-hot action vector
  ↓
Output: Scalar reward prediction
```

### Key Components

1. **Input Processing**:
   - Takes current observation `obs` and next observation `obs_next`
   - Computes `obs_diff = obs - obs_next` (captures temporal changes)
   - Concatenates `[obs, obs_diff]` along the channel dimension
   - For Atari: `obs` shape is `(H, W, C)` where C is number of stacked frames (typically 4)
   - Input to network: `(B, 2*C, H, W)` after channel-first conversion

2. **Convolutional Layers**:
   - **Conv1**: 8×8 kernel, stride 4 → Reduces spatial dimensions by ~4×
   - **Conv2**: 4×4 kernel, stride 2 → Reduces spatial dimensions by ~2×
   - **Conv3**: 3×3 kernel, stride 1 → Feature refinement (no downsampling)
   - All use ReLU activation and 'same' padding to preserve spatial structure

3. **Fully Connected Layers**:
   - Flattened conv features → 512-dimensional hidden layer
   - Output layer: `act_num` units (one per discrete action)

4. **Action Integration**:
   - For discrete actions (Atari), outputs a reward prediction for each action
   - Multiplies with one-hot action vector: `r = sum(r_act * action_onehot)`
   - This allows the network to learn action-specific reward patterns

## Image Observation Format

### Atari Environment

For Atari games, observations are:
- **Shape**: `(84, 84, 4)` - 84×84 grayscale images with 4 stacked frames
- **Preprocessing**:
  1. RGB frame → Grayscale conversion
  2. Resize to 84×84
  3. Stack 4 consecutive frames (for temporal information)
  4. Normalize to [0, 1] range

### Frame Stacking

The environment maintains a frame stack:
```python
frames_stack = [frame_t-3, frame_t-2, frame_t-1, frame_t]
obs = np.stack(frames_stack, axis=-1)  # Shape: (84, 84, 4)
```

This provides temporal context needed for reward prediction.

## RRD Training Process for Images

### 1. Data Collection

```python
# Agent interacts with environment
obs = env.get_obs()  # Shape: (84, 84, 4)
action = agent.step(obs, explore=True)
obs_next, reward, done, info = env.step(action)
```

### 2. RRD Batch Creation

For each episode, the replay buffer creates RRD batches:
- Samples random subsequences of state-action pairs
- For images: samples `(obs, action, obs_next)` tuples
- Target: episodic return `R = sum(rewards)` for the episode

### 3. Reward Network Training

```python
# Forward pass through ConvRewardNet
rrd_rews_pred = reward_net(rrd_obs, rrd_acts, rrd_obs_next)
# rrd_rews_pred shape: (batch_size, sample_size, 1)

# Average predictions
rrd = torch.mean(rrd_rews_pred, dim=1)  # (batch_size, 1)

# Loss: MSE between predicted and actual episodic return
r_loss = F.mse_loss(rrd, rrd_rews)  # rrd_rews is episodic return
```

### 4. Q-Network Training

```python
# Predict rewards for regular transitions
rews_pred = reward_net(obs, acts, obs_next)  # Shape: (batch_size, 1)

# Replace environment rewards with predicted rewards
# Train Q-network using predicted rewards
q_info = train_q(batch_with_predicted_rewards)
```

## Differences from Vector Observations

| Aspect | Vector (MuJoCo) | Image (Atari) |
|--------|------------------|---------------|
| **Reward Network** | `MLPRewardNet` | `ConvRewardNet` |
| **Input** | `[obs, action, obs_diff]` concatenated | `[obs, obs_diff]` on channels |
| **Architecture** | 3-layer MLP (256→256→1) | 3 Conv + 2 FC layers |
| **Action Handling** | Continuous: direct concatenation | Discrete: one-hot encoding |
| **Observation Shape** | `(obs_dim,)` e.g., (111,) | `(H, W, C)` e.g., (84, 84, 4) |
| **Spatial Features** | None | Extracted via convolutions |

## Code Flow

### Network Selection

```python
# In rrd_torch.py
def create_network(self):
    if len(self.args.obs_dims) == 1:
        # Vector observations → MLP
        self.reward_net = MLPRewardNet(obs_dim, act_dim)
    else:
        # Image observations → Conv
        if self.args.env_category == 'atari':
            self.reward_net = ConvRewardNet(self.args.obs_dims, act_num)
```

### Action Processing

```python
# For Atari (discrete actions)
if self.args.env_category == 'atari':
    # Convert action indices to one-hot
    rrd_acts_onehot = torch.zeros(batch_size, sample_size, act_num)
    rrd_acts_onehot.scatter_(2, rrd_acts.unsqueeze(-1), 1.0)
    rrd_acts = rrd_acts_onehot
```

### Forward Pass

```python
# ConvRewardNet.forward()
# 1. Concatenate obs and obs_diff on channels
state = torch.cat([obs, obs - obs_next], dim=1)  # (B, 2*C, H, W)

# 2. Convolutional feature extraction
x = F.relu(self.conv1(state))
x = F.relu(self.conv2(x))
x = F.relu(self.conv3(x))
x = x.reshape(x.shape[0], -1)  # Flatten

# 3. Fully connected layers
x = F.relu(self.fc_act(x))
r_act = self.fc_out(x)  # (B, act_num)

# 4. Action-weighted reward
r = torch.sum(r_act * acts, dim=-1, keepdim=True)  # (B, 1)
```

## Key Design Choices

### 1. Why Concatenate obs and obs_diff?

- **obs**: Current state information
- **obs_diff**: Temporal change information (motion, dynamics)
- Together: Captures both static and dynamic aspects needed for reward prediction

### 2. Why Action-Specific Outputs?

For discrete actions, the network outputs a reward prediction for each action:
- Allows learning action-specific reward patterns
- More expressive than a single scalar output
- Matches the discrete action space structure

### 3. Why Same Architecture as DQN?

The ConvRewardNet uses the same convolutional architecture as DQN's value network:
- Proven effective for Atari games
- Enables transfer of insights from value-based RL
- Consistent feature extraction across components

## Example Usage

```bash
# Train RRD-DQN on Atari game
python scripts/train.py \
    --tag='RRD-DQN Assault' \
    --alg=rrd \
    --basis_alg=dqn \
    --env=Assault \
    --frames=4 \
    --rrd_batch_size=32 \
    --rrd_sample_size=32
```

## Limitations

1. **Not Well-Tuned**: The README notes that RRD-DQN on Atari "has not been well tuned"
2. **Frame Stacking**: Requires maintaining frame history, increasing memory
3. **Computational Cost**: Convolutions are more expensive than MLPs
4. **Spatial Assumptions**: Assumes spatial structure is important (true for Atari, may not be for all image tasks)

## Summary

RRD for image observations:
- Uses **ConvRewardNet** with 3 convolutional + 2 fully connected layers
- Processes **stacked frames** (typically 4) for temporal context
- Handles **discrete actions** via one-hot encoding and action-specific outputs
- Learns to predict **episodic returns** from random subsequences of image observations
- Maintains the same **core RRD algorithm** (randomized return decomposition) as vector observations

The key insight is that RRD adapts its reward network architecture to the observation type while keeping the same learning objective: learning to decompose episodic returns into per-step rewards from random subsequences.

