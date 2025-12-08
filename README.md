# ImagineAct: Model-Based RL for Vision-Language-Action Models

**ImagineAct** is a comprehensive framework for improving out-of-distribution (OOD) generalization of Vision-Language-Action (VLA) models through model-based reinforcement learning. This project combines diffusion-based world modeling, learned reward functions, and actor-critic RL to fine-tune VLAs for better generalization on robotic manipulation tasks.

## 🎯 Overview

This project addresses the challenge of training VLA models that generalize well beyond their training distribution. Our approach uses three key components:

1. **Diffusion-based World Model**: A video prediction model trained on LIBERO dataset that can generate future robot states conditioned on actions
2. **Learned Proxy Rewards**: Video-based reward models trained using Randomized Return Decomposition (RRD) that extract dense reward signals from visual observations
3. **RL Fine-tuning Pipeline**: An end-to-end system that uses the world model as a simulator and learned rewards to fine-tune VLA policies

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ImagineAct Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. World Model Training                                        │
│     [LIBERO Dataset] → [Diffusion World Model]                 │
│                                                                 │
│  2. Reward Model Learning                                       │
│     [LIBERO Trajectories] → [OpenVLA Features] → [RRD] →       │
│     [Learned Reward Function]                                   │
│                                                                 │
│  3. RL Fine-tuning                                              │
│     [OpenVLA Actor] → [World Model Env] → [Reward Model] →     │
│     [Critic] → [PPO Updates] → [Fine-tuned Policy]             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Project Structure

```
ImagineAct/
├── world-model-eval-babel/          # Diffusion-based world model
│   ├── world_model.py               # World model implementation
│   ├── model.py                     # Diffusion Transformer (DiT)
│   ├── vae.py                       # Variational Autoencoder
│   └── diffusion.py                 # Diffusion sampling
│
├── Randomized-Return-Decomposition/  # RRD for reward learning
│   ├── algorithm/
│   │   ├── openvla_reward_net.py   # OpenVLA-based reward network
│   │   ├── rrd_torch.py            # Randomized Return Decomposition
│   │   └── replay_buffer/          # Offline data buffers
│   ├── offline_openvla_ac.py       # Offline actor-critic
│   └── scripts/
│       ├── train_offline_openvla_ac.py      # Offline RL training
│       ├── train_openvla_rl_worldmodel.py   # Online RL with world model
│       └── train_with_openvla.py            # Reward model training
│
└── openvla/                         # OpenVLA VLA model
    ├── prismatic/                   # Core model implementation
    └── experiments/                 # Evaluation scripts
```

## 🚀 Key Components

### 1. Diffusion-Based World Model

A video prediction model that generates future robot states conditioned on actions. The world model:
- Uses a Diffusion Transformer (DiT) architecture for temporal modeling
- Encodes/decodes frames using a VAE
- Trained on LIBERO manipulation dataset
- Supports autoregressive generation for long-horizon rollouts

**Location**: `world-model-eval-babel/`

### 2. Learned Proxy Rewards via RRD

Reward functions learned from offline data using Randomized Return Decomposition:
- Uses OpenVLA's vision and language features as input
- Trains dense reward models that correlate with task success
- Leverages RRD's return decomposition for stable learning
- Supports both offline and online reward computation

**Key Files**:
- `algorithm/openvla_reward_net.py`: Reward network architecture
- `scripts/train_with_openvla.py`: Training script for reward models
- `scripts/evaluate_rewards.py`: Evaluation utilities

### 3. RL Fine-tuning Pipeline

End-to-end RL system for fine-tuning VLA policies:
- **Offline Actor-Critic**: Train on pre-collected LIBERO trajectories
- **Online RL with World Model**: Use world model as environment for policy optimization
- **Actor**: OpenVLA model generating continuous actions
- **Critic**: Value function network using OpenVLA features
- **Reward**: Learned reward model for dense supervision
- **Algorithm**: PPO with GAE for advantage estimation

**Key Scripts**:
- `scripts/train_offline_openvla_ac.py`: Offline RL training
- `scripts/train_openvla_rl_worldmodel.py`: Online RL with world model

## 📋 Requirements

### Dependencies

- Python 3.8+
- PyTorch 2.0+
- Transformers (HuggingFace)
- OpenVLA model checkpoint
- LIBERO dataset (RLDS format)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ImagineAct

# Install dependencies for each component
cd Randomized-Return-Decomposition
pip install -r requirements/requirements_pytorch.txt

# Set up environment variables
export OPENVLA_PATH=$(pwd)/../openvla
export LIBERO_DATASET_PATH=/path/to/libero/dataset
```

## 🎓 Usage

### Step 1: Train/Load World Model

The world model should be pre-trained on LIBERO data. To use an existing checkpoint:

```python
from world_model import WorldModel

world_model = WorldModel(
    checkpoint_path="/path/to/world_model.pt",
    use_pixel_rope=False,
    default_cfg=1.0
)
```

### Step 2: Train Reward Model

Train a reward model using RRD on LIBERO trajectories:

```bash
cd Randomized-Return-Decomposition

python scripts/train_with_openvla.py \
    --env libero-10 \
    --use_openvla_features True \
    --openvla_checkpoint /path/to/openvla/model \
    --libero_dataset_path /path/to/libero/dataset \
    --rrd_reward_only True \
    --epochs 100 \
    --tag "openvla_reward_libero_10"
```

### Step 3: Offline Actor-Critic Training

Train OpenVLA using offline RL on LIBERO data:

```bash
python scripts/train_offline_openvla_ac.py \
    --features_cache_path log/feature_cache/openvla_features.pkl \
    --rlds_dataset_path /path/to/libero/dataset \
    --openvla_checkpoint /path/to/openvla/model \
    --reward_model_checkpoint log/checkpoints/openvla_reward_libero_10/checkpoint_best.pt \
    --critic_hidden_dims 2048 512 128 \
    --epochs 121 \
    --batch_size 16 \
    --vla_lr 5e-6 \
    --critic_lr 3e-4
```

### Step 4: RL Fine-tuning with World Model

Fine-tune the policy using the world model as environment:

```bash
python scripts/train_openvla_rl_worldmodel.py \
    --openvla_checkpoint /path/to/openvla/model \
    --world_model_checkpoint /path/to/world_model.pt \
    --reward_model_checkpoint log/checkpoints/openvla_reward_libero_10/checkpoint_best.pt \
    --initial_states_path /path/to/initial/states \
    --num_envs 8 \
    --rollout_steps 2048 \
    --num_updates 1000 \
    --ppo_epochs 4 \
    --vla_lr 1e-5 \
    --critic_lr 3e-4 \
    --use_wandb
```

## 📊 Key Features

### World Model
- ✅ Autoregressive video generation
- ✅ Action-conditioned state prediction
- ✅ Support for long-horizon rollouts
- ✅ Efficient sampling with DDIM

### Reward Learning
- ✅ OpenVLA feature-based rewards
- ✅ Dense reward signals for better learning
- ✅ Compatible with offline datasets
- ✅ Evaluation and visualization tools

### RL Training
- ✅ Offline actor-critic on real data
- ✅ Online RL with world model simulator
- ✅ PPO with GAE advantage estimation
- ✅ Behavior cloning for stability
- ✅ Distributed training support

## 🧪 Evaluation

Evaluate the trained policy:

```bash
# Evaluate offline trained model
python scripts/evaluate_offline_openvla_ac.py \
    --checkpoint_path log/checkpoints/offline_openvla_ac_final.pt \
    --eval_dataset_path /path/to/eval/dataset

# Evaluate reward predictions
python scripts/evaluate_rewards.py \
    --checkpoint_path log/checkpoints/openvla_reward_libero_10/checkpoint_best.pt
```

## 📈 Results

The framework enables:
- **Improved OOD Generalization**: Fine-tuned policies generalize better to unseen task configurations
- **Efficient Training**: World model allows fast policy iteration without real robot interaction
- **Dense Rewards**: Learned rewards provide better learning signal than sparse task rewards
- **Scalable**: Can leverage large offline datasets for initial training

## 🙏 Acknowledgments

This project builds upon and integrates several outstanding open-source projects:

### World-Model-Eval
We extend the [world-model-eval](https://github.com/world-model-eval/world-model-eval) codebase for diffusion-based world modeling. The world model implementation uses their diffusion transformer architecture and VAE for frame encoding/decoding.

### OpenVLA
We use [OpenVLA](https://github.com/prisms-center/openvla) as our base VLA model. OpenVLA provides excellent pretrained vision-language-action policies that serve as the foundation for our fine-tuning approach.

### Randomized Return Decomposition (RRD)
Our reward learning builds on the [Randomized Return Decomposition](https://github.com/google-research/google-research/tree/master/rrd) framework. We adapt RRD to work with OpenVLA features for learning dense reward functions from offline robot data.

We are grateful to all the contributors and researchers behind these projects for making this work possible.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{imagineact2025,
  title={ImagineAct: Model-Based RL for Vision-Language-Action Models},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/ImagineAct}
}
```

## 📄 License

This project integrates code from multiple sources. Please refer to:
- `openvla/LICENSE` for OpenVLA licensing
- `Randomized-Return-Decomposition/LICENSE` for RRD licensing
- Individual source files for world-model-eval licensing

## 🔗 References

- **OpenVLA**: [Paper](https://arxiv.org/abs/2310.15273) | [Code](https://github.com/prisms-center/openvla)
- **Randomized Return Decomposition**: [Paper](https://arxiv.org/abs/2106.05380)
- **World-Model-Eval**: [Code](https://github.com/world-model-eval/world-model-eval)
- **LIBERO**: [Paper](https://arxiv.org/abs/2306.14438) | [Dataset](https://libero-project.github.io/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This project is a research framework. Results may vary based on hardware, dataset versions, and hyperparameters. Please refer to individual component documentation for detailed setup instructions.

