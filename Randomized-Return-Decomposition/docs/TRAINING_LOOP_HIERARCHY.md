# Training Loop Hierarchy

This document explains the nested loop structure used in the training process.

## Hierarchy Overview

The training follows a 4-level nested loop structure:

```
Epoch (outermost)
  └── Cycle
      └── Iteration
          └── Train Batch (innermost)
```

## Detailed Structure

### 1. **Epoch** (Outermost Loop)
- **Location**: `scripts/train.py` line 58
- **Loop**: `for epoch in range(args.epochs)`
- **Purpose**: Highest level of training organization
- **Actions per epoch**:
  - Runs `args.cycles` cycles
  - Calls `tester.epoch_summary()` at the end
  - Saves checkpoint if `(epoch + 1) % args.save_freq == 0`
- **Default**: `epochs=3` (but often set to 10+ for longer training)

### 2. **Cycle** (Per Epoch)
- **Location**: `scripts/train.py` line 59
- **Loop**: `for cycle in range(args.cycles)`
- **Purpose**: Medium-level training unit where metrics are logged
- **Actions per cycle**:
  - Calls `learner.learn()` (which contains iterations)
  - Optionally runs `tester.cycle_summary()` (if not reward-only mode)
  - Logs metrics to console and TensorBoard via `logger.tabular_show()` and `logger.summary_show()`
  - Records: Epoch, Cycle, Episodes, Timesteps, TimeCost(sec)/train, TimeCost(sec)/test
- **Default**: `cycles=100` (but often set to 10-50 for shorter cycles)

### 3. **Iteration** (Per Cycle)
- **Location**: `learner/libero.py` line 25
- **Loop**: `for iteration in range(args.iterations)`
- **Purpose**: Groups batches together, updates normalizer
- **Actions per iteration**:
  - Updates observation normalizer (if enabled)
  - Runs `args.train_batches` training batches
  - Logs average R_loss at end of iteration (if reward-only training)
- **Default**: `iterations=100`

### 4. **Train Batch** (Innermost Loop)
- **Location**: `learner/libero.py` line 34
- **Loop**: `for batch_idx in range(args.train_batches)`
- **Purpose**: Individual gradient update step
- **Actions per batch**:
  - Samples a batch from the replay buffer
  - Calls `agent.train_r(batch)` (for reward-only training)
  - Records `R_loss` and logs via `logger.add_dict(info)`
  - Updates target networks if `target_count % train_target == 0`
- **Default**: `train_batches=25-100` (varies by environment)

## Visual Representation

```
Training Start
│
├── Epoch 0/10
│   ├── Cycle 0/100
│   │   ├── Iteration 0/100
│   │   │   ├── Batch 0/50 → train_r() → R_loss recorded
│   │   │   ├── Batch 1/50 → train_r() → R_loss recorded
│   │   │   ├── ...
│   │   │   └── Batch 49/50 → train_r() → R_loss recorded
│   │   │   └── [Log: Avg R_loss for iteration 0]
│   │   ├── Iteration 1/100
│   │   │   ├── Batch 0/50 → train_r() → R_loss recorded
│   │   │   ├── ...
│   │   │   └── Batch 49/50 → train_r() → R_loss recorded
│   │   │   └── [Log: Avg R_loss for iteration 1]
│   │   ├── ... (98 more iterations)
│   │   └── Iteration 99/100
│   │       └── ... (50 batches)
│   │   └── [Log: Avg R_loss for iteration 99]
│   │   └── [Log metrics to TensorBoard, show table]
│   ├── Cycle 1/100
│   │   └── ... (same structure)
│   ├── ... (98 more cycles)
│   └── Cycle 99/100
│       └── ... (same structure)
│   └── [Epoch summary, save checkpoint if needed]
│
├── Epoch 1/10
│   └── ... (same structure as Epoch 0)
│
├── ... (8 more epochs)
│
└── Epoch 9/10
    └── ... (same structure)
    └── [Save final checkpoint]
```

## Your Current Configuration

Based on your training command:
- `--epochs 10`
- `--cycles 100`
- `--iterations 100`
- `--train_batches 50`

### Total Training Steps

- **Batches per iteration**: 50
- **Batches per cycle**: 100 iterations × 50 batches = **5,000 batches**
- **Batches per epoch**: 100 cycles × 5,000 batches = **500,000 batches**
- **Total batches**: 10 epochs × 500,000 batches = **5,000,000 batches**

### Logging Frequency

1. **R_loss per batch**: Recorded via `logger.add_dict(info)` (50,000 times per cycle)
2. **R_loss summary**: Printed every 100 batches (50 times per iteration)
3. **R_loss iteration summary**: Printed at end of each iteration (100 times per cycle)
4. **Cycle summary**: Logged to TensorBoard and console (100 times per epoch)
5. **Epoch summary**: Printed at end of each epoch (10 times total)

## What Happens at Each Level

### At the Batch Level (50 times per iteration):
- Sample batch from buffer
- Forward pass through reward network
- Compute loss
- Backward pass (gradient update)
- Record `R_loss` to logger

### At the Iteration Level (100 times per cycle):
- Update observation normalizer (if enabled)
- Run 50 training batches
- Print average R_loss over the iteration

### At the Cycle Level (100 times per epoch):
- Run 100 iterations (5,000 batches total)
- Log all metrics to TensorBoard
- Display progress table in console
- Record: Epoch, Cycle, Episodes, Timesteps, TimeCost, R_loss (averaged)

### At the Epoch Level (10 times total):
- Run 100 cycles (500,000 batches total)
- Run epoch summary (test rollouts if not reward-only)
- Save checkpoint if `(epoch + 1) % save_freq == 0`

## Metric Aggregation

### R_loss Aggregation:
- **Per batch**: Individual loss value
- **Per iteration**: Average over 50 batches (printed to console)
- **Per cycle**: Average over 5,000 batches (logged to TensorBoard)
- **Per epoch**: Average over 500,000 batches (can be computed from TensorBoard)

### TensorBoard X-axis:
- Uses `learner.step_counter` (total timesteps in buffer: 101,469)
- This remains constant for offline training
- Each cycle writes one data point to TensorBoard

## Typical Training Flow

```
Epoch 0/10
  Cycle 0/100
    Iteration 0/100: [50 batches] → Avg R_loss: 0.012345
    Iteration 1/100: [50 batches] → Avg R_loss: 0.011234
    ...
    Iteration 99/100: [50 batches] → Avg R_loss: 0.005678
    → [TensorBoard: R_loss = 0.008901 (avg over cycle)]
    → [Console: Display progress table]
  
  Cycle 1/100
    ...
  
  ... (98 more cycles)
  
  → [Epoch summary, checkpoint saved]

Epoch 1/10
  ...
```

## Recommendations

### For Faster Development:
- Reduce `--cycles` to 10-20 (faster feedback)
- Reduce `--iterations` to 50 (still good coverage)
- Keep `--train_batches` at 50 (good batch size)

### For Full Training:
- Keep `--epochs` at 10+ (sufficient training)
- Keep `--cycles` at 50-100 (good granularity)
- Keep `--iterations` at 100 (standard)
- Keep `--train_batches` at 50 (good batch size)

### For Debugging:
- Set `--epochs 1`, `--cycles 1`, `--iterations 1`, `--train_batches 1`
- This runs just 1 batch total for quick testing

## Summary

The hierarchy allows for:
1. **Fine-grained control**: Adjust training at different levels
2. **Efficient logging**: Aggregate metrics at appropriate levels
3. **Checkpointing**: Save at epoch boundaries
4. **Progress tracking**: See progress at cycle level without too much verbosity

The key insight: **One cycle = one TensorBoard data point**, which represents the average performance over 5,000 batches (in your config).

