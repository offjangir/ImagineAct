# Batch Size Parameters Explained

## Overview

RRD training uses **4 different batch size parameters** that control different aspects of training:

## Parameters

### 1. `TRAIN_BATCHES` (64)
**What it is**: Number of training batches per iteration

**How it's used**:
```python
# In learner/libero.py
for batch_idx in range(args.train_batches):  # Train 64 batches
    batch = buffer.sample_batch()
    agent.train_r(batch)  # or agent.train(batch)
```

**Meaning**: 
- Each iteration, we train on **64 batches** of data
- Higher = more training per iteration (but slower)
- Lower = less training per iteration (but faster)

**Example**: `TRAIN_BATCHES=64` means train on 64 batches per iteration

---

### 2. `BATCH_SIZE` (32)
**What it is**: Number of **regular transitions** sampled for Q-network/policy training

**How it's used**:
```python
# In buffer.sample_batch()
for i in range(batch_size):  # Sample 32 transitions
    transition = sample_random_transition()
    batch['obs'].append(transition['obs'])
    batch['acts'].append(transition['acts'])
    # ...
```

**Meaning**:
- Number of individual (state, action, reward, next_state) transitions
- Used for training Q-networks and policies
- Each transition is a single timestep from an episode

**Example**: `BATCH_SIZE=32` means sample 32 individual transitions

---

### 3. `RRD_BATCH_SIZE` (32)
**What it is**: Total number of **random subsequences** for reward network training

**How it's used**:
```python
# In buffer.sample_batch()
num_subsequences = rrd_batch_size // rrd_sample_size  # 32 // 32 = 1
for i in range(num_subsequences):  # Sample 1 subsequence per episode
    subsequence = episode.rrd_sample(rrd_sample_size)  # Get 32 timesteps
    batch['rrd_obs'].append(subsequence['rrd_obs'])
    # ...
```

**Meaning**:
- Total number of timesteps across all subsequences for reward network
- Each subsequence comes from a different episode
- Formula: `num_subsequences = RRD_BATCH_SIZE // RRD_SAMPLE_SIZE`

**Example**: 
- `RRD_BATCH_SIZE=32`, `RRD_SAMPLE_SIZE=32` → 1 subsequence per episode
- `RRD_BATCH_SIZE=64`, `RRD_SAMPLE_SIZE=32` → 2 subsequences per episode

---

### 4. `RRD_SAMPLE_SIZE` (32)
**What it is**: Size of each **random subsequence** (number of timesteps)

**How it's used**:
```python
# In Trajectory.rrd_sample()
idx = np.random.choice(episode_length, sample_size, replace=True)
# Sample 32 random timesteps from episode
subsequence = {
    'rrd_obs': episode['obs'][idx],      # 32 observations
    'rrd_acts': episode['acts'][idx],     # 32 actions
    'rrd_rews': [episodic_return]        # 1 target (full episode return)
}
```

**Meaning**:
- Number of timesteps in each random subsequence
- These timesteps are randomly sampled from a full episode
- The reward network predicts the **full episodic return** from just these timesteps

**Example**: 
- Episode has 200 timesteps
- `RRD_SAMPLE_SIZE=32` → randomly sample 32 timesteps
- Task: Predict full episode return (sum of all 200 rewards) from just 32 timesteps

---

## Visual Example

### Training Flow

```
Iteration 1:
  ├─ Batch 1: Train on 32 transitions + 1 subsequence (32 timesteps)
  ├─ Batch 2: Train on 32 transitions + 1 subsequence (32 timesteps)
  ├─ ...
  └─ Batch 64: Train on 32 transitions + 1 subsequence (32 timesteps)
```

### Batch Structure

```
Regular Batch (for Q-network):
  ├─ obs: [32 transitions] × [obs_dim]
  ├─ acts: [32 transitions] × [act_dim]
  └─ rews: [32 transitions] × [1]

RRD Batch (for reward network):
  ├─ rrd_obs: [1 subsequence] × [32 timesteps] × [obs_dim]
  ├─ rrd_acts: [1 subsequence] × [32 timesteps] × [act_dim]
  └─ rrd_rews: [1] × [episodic_return]  # Target: full episode return
```

---

## Recommended Settings

### For LIBERO (Image Observations)
```bash
TRAIN_BATCHES=50-100      # More batches for image data
BATCH_SIZE=32             # Standard size
RRD_BATCH_SIZE=32         # Match batch_size
RRD_SAMPLE_SIZE=32        # Good balance
```

### For MuJoCO (Vector Observations)
```bash
TRAIN_BATCHES=25          # Fewer batches (faster)
BATCH_SIZE=256            # Larger (vector is smaller)
RRD_BATCH_SIZE=256        # Match batch_size
RRD_SAMPLE_SIZE=64        # Larger subsequences
```

---

## Key Relationships

1. **More `TRAIN_BATCHES`** = More training per iteration = Slower but better convergence
2. **Larger `BATCH_SIZE`** = More stable gradients = Better for Q-network training
3. **Larger `RRD_BATCH_SIZE`** = More subsequences = More diverse reward training
4. **Larger `RRD_SAMPLE_SIZE`** = Longer subsequences = Easier to predict returns (but less randomization)

---

## Memory Considerations

**Total memory per batch** ≈
- Regular batch: `BATCH_SIZE × (obs_dim + act_dim + ...)`
- RRD batch: `(RRD_BATCH_SIZE // RRD_SAMPLE_SIZE) × RRD_SAMPLE_SIZE × (obs_dim + act_dim + ...)`

For images:
- `BATCH_SIZE=32` with `256×256×3` images = ~6MB per batch
- `RRD_BATCH_SIZE=32`, `RRD_SAMPLE_SIZE=32` = ~6MB per batch
- Total: ~12MB per batch × `TRAIN_BATCHES=64` = ~768MB per iteration

