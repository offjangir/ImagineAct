# Repository Structure Guide

This document explains the organization of the Randomized-Return-Decomposition repository, including where neural networks, environment simulators, and training logic are located.

## 📁 Directory Structure

```
Randomized-Return-Decomposition/
├── algorithm/              # Neural network models and RL algorithms
│   ├── basis_alg/         # Base RL algorithms (SAC, DDPG, TD3, DQN)
│   │   ├── base_torch.py  # Base class for PyTorch algorithms
│   │   ├── sac_torch.py   # SAC implementation (PyTorch)
│   │   ├── ddpg_torch.py  # DDPG implementation (PyTorch)
│   │   └── ...
│   ├── rrd_torch.py       # Randomized Reward Decomposition (PyTorch)
│   ├── rrd.py             # RRD (TensorFlow - original)
│   └── replay_buffer/      # Experience replay buffers
├── envs/                   # Environment simulators
│   ├── normal_mujoco.py   # MuJoCo physics simulator wrapper
│   ├── normal_atari.py    # Atari game environment wrapper
│   └── ep_rews.py         # Episodic rewards wrapper
├── learner/                # Training loop logic
│   ├── mujoco.py          # MuJoCo-specific training loop
│   └── atari.py           # Atari-specific training loop
├── utils/                  # Utility functions
│   ├── torch_utils.py     # PyTorch utilities (normalizers, etc.)
│   └── os_utils.py        # Logging and OS utilities
├── scripts/                # Executable scripts
│   ├── train.py           # Main training script
│   ├── test.py            # Testing/evaluation script
│   └── *.sh               # Installation scripts
├── requirements/           # Dependency files
└── docs/                   # Documentation
```

## 🧠 Neural Network Models

### Location: `algorithm/` directory

The neural network architectures are defined in the algorithm files:

#### 1. **Base Algorithms** (`algorithm/basis_alg/`)

**PyTorch Implementations:**
- **`base_torch.py`**: Base class for all PyTorch algorithms
  - Handles device management (CPU/GPU)
  - Observation normalization
  - Model saving/loading

- **`sac_torch.py`**: Soft Actor-Critic (SAC) algorithm
  - **Policy Network** (`MLPStochasticPolicy`): 
    - 3-layer MLP: `obs_dim → 256 → 256 → act_dim*2`
    - Outputs mean and logstd for stochastic policy
  - **Q-Value Networks** (`MLPQValueSAC`):
    - 3-layer MLP: `(obs_dim + act_dim) → 256 → 256 → 1`
    - Two Q-networks for double Q-learning
    - Target networks for stability

- **`ddpg_torch.py`**: Deep Deterministic Policy Gradient
  - Similar architecture to SAC but with deterministic policy

#### 2. **RRD Algorithm** (`algorithm/rrd_torch.py`)

**Reward Decomposition Networks:**
- **`MLPRewardNet`**: For continuous control (MuJoCo)
  - Input: `[obs, action, obs - obs_next]`
  - Architecture: `state_dim → 256 → 256 → 1`
  - Predicts decomposed rewards

- **`ConvRewardNet`**: For image-based tasks (Atari)
  - Convolutional layers: `32 → 64 → 64 filters`
  - Fully connected: `512 → act_num`
  - Processes image observations

**Key Components:**
- `RRD` class wraps a basis algorithm (SAC/DDPG) and adds reward decomposition
- Reward network learns to predict rewards from state-action pairs
- Used for randomized reward decomposition learning

### Network Architecture Summary

```
SAC Policy Network:
  obs (obs_dim) 
    → Linear(256) + ReLU
    → Linear(256) + ReLU  
    → Linear(act_dim*2)  [mean, logstd]

SAC Q-Network:
  [obs, action] (obs_dim + act_dim)
    → Linear(256) + ReLU
    → Linear(256) + ReLU
    → Linear(1)  [Q-value]

RRD Reward Network (MLP):
  [obs, action, obs_diff] (obs_dim*2 + act_dim)
    → Linear(256) + ReLU
    → Linear(256) + ReLU
    → Linear(1)  [reward]
```

## 🌍 Environment Simulators

### Location: `envs/` directory

#### 1. **MuJoCo Environments** (`envs/normal_mujoco.py`)

**`MuJoCoNormalEnv`** class:
- Wraps OpenAI Gym MuJoCo environments
- Supported environments:
  - `Ant-v2`, `HalfCheetah-v2`, `Walker2d-v2`
  - `Humanoid-v2`, `Reacher-v2`, `Swimmer-v2`
  - `Hopper-v2`, `HumanoidStandup-v2`

**Key Methods:**
- `reset()`: Reset environment to initial state
- `step(action)`: Execute action, return (obs, reward, done, info)
- `get_obs()`: Get current observation
- Handles both old and new Gym API formats

**Physics Engine:**
- Uses `gym.make(env_name)` which loads MuJoCo physics
- MuJoCo runs on **CPU** (not GPU)
- This is why GPU utilization is low during training

#### 2. **Atari Environments** (`envs/normal_atari.py`)

**`AtariNormalEnv`** class:
- Wraps OpenAI Gym Atari environments
- Handles frame stacking and preprocessing
- Supports all standard Atari games

#### 3. **Environment Factory** (`envs/__init__.py`)

**`make_env(args)`** function:
- Creates appropriate environment based on `args.env`
- Returns wrapped environment with episodic rewards if needed
- Maps environment names to categories (atari/mujoco)

## 🔄 Training Loop

### Location: `learner/` directory

#### **MuJoCo Learner** (`learner/mujoco.py`)

**`MuJoCoLearner`** class:
- Manages the training loop for MuJoCo environments
- **Key Process:**
  1. **Data Collection**: 
     - Agent interacts with environment
     - Stores transitions in replay buffer
  2. **Training**:
     - Samples batches from replay buffer
     - Updates policy and Q-networks
     - Updates target networks (for stability)
  3. **Logging**:
     - Tracks episodes, timesteps, rewards

**Training Flow:**
```
For each iteration:
  For each timestep:
    action = agent.step(obs, explore=True)
    obs, reward, done, info = env.step(action)
    buffer.store_transition(obs, action, reward, ...)
    
    if buffer.size >= warmup:
      for train_batches:
        batch = buffer.sample_batch()
        agent.train(batch)  # Update networks
```

## 🚀 Entry Points

### Main Scripts: `scripts/` directory

1. **`train.py`**: Main training script
   - Parses arguments via `common.py`
   - Sets up environment, agent, buffer, learner
   - Runs training loop with logging

2. **`test.py`**: Evaluation script
   - Loads trained model
   - Runs evaluation rollouts
   - Reports performance metrics

3. **`common.py`**: Configuration and setup
   - Argument parsing
   - Creates environment, agent, buffer, learner
   - Initializes logging

## 📊 Data Flow

```
┌─────────────┐
│  Environment│ (MuJoCo/Atari simulator)
│  (envs/)    │
└──────┬──────┘
       │ obs, reward, done
       ▼
┌─────────────┐
│   Agent     │ (Neural networks)
│ (algorithm/)│
└──────┬──────┘
       │ action
       ▼
┌─────────────┐
│   Buffer    │ (Experience replay)
│(replay_buf/)│
└──────┬──────┘
       │ batch samples
       ▼
┌─────────────┐
│   Learner   │ (Training loop)
│  (learner/) │
└─────────────┘
```

## 🔑 Key Files Summary

| Component | Location | Purpose |
|-----------|----------|---------|
| **Neural Networks** | `algorithm/basis_alg/*_torch.py` | Policy and Q-value networks |
| **Reward Networks** | `algorithm/rrd_torch.py` | Reward decomposition networks |
| **Environment** | `envs/normal_mujoco.py` | MuJoCo physics simulator wrapper |
| **Training Loop** | `learner/mujoco.py` | Data collection and training logic |
| **Replay Buffer** | `algorithm/replay_buffer/` | Experience storage and sampling |
| **Main Script** | `scripts/train.py` | Entry point for training |
| **Config** | `common.py` | Argument parsing and setup |

## 💡 Important Notes

1. **GPU vs CPU**: 
   - Neural networks run on GPU (if available)
   - MuJoCo physics runs on CPU (this is why GPU utilization is low)

2. **Backend Support**:
   - PyTorch implementations in `*_torch.py` files
   - Original TensorFlow implementations in `*.py` files (no `_torch` suffix)
   - Controlled by `USE_PYTORCH` environment variable

3. **Architecture**:
   - All networks are **MLPs** (Multi-Layer Perceptrons)
   - No convolutional layers for MuJoCo (vector observations)
   - Convolutional layers only for Atari (image observations)

4. **Training Speed**:
   - Bottleneck is CPU-based MuJoCo simulation
   - Not GPU computation (networks are small MLPs)
   - Parallelization helps but is limited by CPU cores

