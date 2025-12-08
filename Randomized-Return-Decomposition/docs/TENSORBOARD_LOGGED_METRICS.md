# TensorBoard Logged Metrics

This document details what metrics are automatically logged to TensorBoard during training.

## How Logging Works

1. **Registration**: Metrics are registered as 'scalar' items in `train.py` (lines 31-42)
2. **Recording**: Values are recorded via `logger.add_record()` or `logger.add_dict()` during training
3. **Writing**: Metrics are written to TensorBoard at the end of each cycle via `logger.summary_show(step_counter)`
4. **X-axis**: All metrics use `learner.step_counter` (total timesteps) as the x-axis

## Logged Metrics

### 1. Training Progress Metrics

These are logged every cycle:

- **`Epoch`**: Current epoch number (as string, e.g., "0/10")
- **`Cycle`**: Current cycle number (as string, e.g., "1/50")
- **`Episodes`**: Total number of episodes processed
- **`Timesteps`**: Total number of timesteps processed (used as x-axis)
- **`TimeCost(sec)/train`**: Training time per cycle (seconds)
- **`TimeCost(sec)/test`**: Testing time per cycle (seconds)

**Note**: `Epoch` and `Cycle` are logged as strings, so they may not display well in TensorBoard. The numeric metrics (`Episodes`, `Timesteps`, `TimeCost`) are the most useful.

### 2. Algorithm Training Metrics (from `agent.train_info`)

These come from the algorithm's `train_info` dictionary, which is populated during training:

#### For RRD (Randomized Reward Decomposition):
- **`R_loss`**: Reward network loss (MSE between predicted and actual episodic returns)
  - **Logged**: After each `train_r(batch)` call
  - **Frequency**: Every batch (accumulated and averaged per cycle)
  - **This is the main metric for reward-only training**

- **`R_var`**: Reward variance penalty (only if `--rrd_bias_correction True`)
  - **Logged**: After each `train_r(batch)` call when bias correction is enabled

#### For SAC/DDPG (if training policy):
- **`pi_loss`**: Policy loss
- **`q_loss`**: Q-value loss
- **`alpha`**: Temperature parameter (SAC only)

**Note**: In reward-only training mode (`--rrd_reward_only True`), only `R_loss` (and optionally `R_var`) will be logged, as policy training is skipped.

### 3. Learner Metrics (from `learner.learner_info`)

Currently, `learner.learner_info` is an empty list for LIBERO offline training, so no learner-specific metrics are logged.

### 4. Test Metrics (from `agent.step_info` and `tester.info`)

- **`agent.step_info`**: Metrics from policy rollouts during testing
  - **Note**: Empty for reward-only training (testing is skipped)

- **`tester.info`**: Test performance metrics
  - **Note**: Empty for reward-only training (testing is skipped)

### 5. Environment Metrics (from `env.env_info`)

For LIBERO offline environment:
- **`Steps`**: Number of steps processed (from `env.process_info_steps`)
- **`Rewards`**: Cumulative rewards (from `env.process_info_rewards`)

**Note**: These may not be very meaningful for offline training since we're not interacting with the environment.

## What Gets Logged in Your Current Setup

For your current training command with `--rrd_reward_only True`:

### Metrics Logged Every Cycle:
1. **`R_loss`** ⭐ (Most important - reward network loss)
2. **`Timesteps`** (x-axis for all plots)
3. **`Episodes`** (379 for your dataset)
4. **`TimeCost(sec)/train`** (training time per cycle)
5. **`TimeCost(sec)/test`** (0.0 since testing is skipped)
6. **`Steps`** (from env_info)
7. **`Rewards`** (from env_info)

### Metrics NOT Logged:
- `R_var` (only if `--rrd_bias_correction True`)
- Policy/Q-value losses (skipped in reward-only mode)
- Test metrics (testing is skipped)

## Viewing in TensorBoard

All metrics are written to:
```
log/board/<experiment_name>/debug/events.out.tfevents.*
```

To view:
```bash
tensorboard --logdir=log/board
```

### Key Metrics to Monitor:

1. **`R_loss`**: Should decrease over time as the reward network learns
   - Initial values: May be high (e.g., 0.1-1.0)
   - Target: Should decrease toward 0.01-0.05 range
   - Watch for: Smooth decrease, not oscillating wildly

2. **`Timesteps`**: X-axis showing training progress
   - Your dataset: 101,469 total timesteps
   - This will remain constant for offline training

3. **`TimeCost(sec)/train`**: Training efficiency
   - Monitor to ensure training isn't slowing down

## Example TensorBoard View

When you open TensorBoard, you'll see plots like:

```
Scalars:
  ├── R_loss (decreasing curve)
  ├── Timesteps (constant at 101469)
  ├── Episodes (constant at 379)
  ├── TimeCost(sec)/train (varies by cycle)
  ├── TimeCost(sec)/test (0.0)
  ├── Steps (from env_info)
  └── Rewards (from env_info)
```

## Notes

1. **Averaging**: Values logged to TensorBoard are averaged over the cycle (accumulated sum divided by count)
2. **Step Counter**: The x-axis uses `learner.step_counter`, which for offline training is the buffer size (101,469 for your dataset)
3. **Frequency**: Metrics are written once per cycle, not per batch
4. **R_loss Detail**: While individual batch `R_loss` values are printed to logs, TensorBoard shows the cycle-averaged value

## Troubleshooting

**If `R_loss` doesn't appear in TensorBoard:**
- Check that `R_loss` is in `agent.train_info` (it should be initialized in `rrd_torch.py`)
- Verify it's being logged via `logger.add_dict(info)` in `learner/libero.py`
- Ensure TensorBoard is reading from the correct log directory

**If metrics seem constant:**
- This is normal for offline training - `Timesteps` and `Episodes` won't change
- `R_loss` should decrease over cycles/epochs, not timesteps

