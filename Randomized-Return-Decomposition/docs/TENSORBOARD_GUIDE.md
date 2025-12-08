# TensorBoard Visualization Guide

This repository uses TensorBoard for logging and visualizing training metrics. All logs are automatically saved during training.

## Quick Start

### 1. Launch TensorBoard

Navigate to the repository root and run:

```bash
tensorboard --logdir=log/board
```

This will start TensorBoard and display a URL (typically `http://localhost:6006`).

### 2. View in Browser

Open the URL in your web browser. You'll see:
- **Scalars**: Training metrics like `R_loss`, `TimeCost(sec)/train`, etc.
- **Graphs**: Real-time plots of metrics over training steps

## Log Directory Structure

TensorBoard logs are saved in:
```
log/board/
  └── <experiment_name>/
      └── debug/
          └── events.out.tfevents.*
```

For example, your recent experiment:
```
log/board/pretrain_RRD_Libero10-rrd-libero-10-(2025-11-22-20:20:37)/
  └── debug/
      └── events.out.tfevents.1763872002.gretel.3066256.0
```

## Logged Metrics

The following metrics are automatically logged to TensorBoard:

### Training Metrics (from `agent.train_info`):
- **`R_loss`**: Reward network loss (MSE between predicted and actual episodic returns)
- **`R_var`**: Reward variance penalty (if `--rrd_bias_correction True`)
- **`pi_loss`**: Policy loss (if training policy)
- **`q_loss`**: Q-value loss (if training policy)

### Training Progress:
- **`Epoch`**: Current epoch number
- **`Cycle`**: Current cycle number
- **`Episodes`**: Number of episodes processed
- **`Timesteps`**: Total timesteps processed
- **`TimeCost(sec)/train`**: Training time per cycle
- **`TimeCost(sec)/test`**: Testing time per cycle

### Environment Metrics (from `env.env_info`):
- Environment-specific metrics (if available)

### Test Metrics (from `tester.info`):
- Test performance metrics (if testing is enabled)

## Viewing Specific Experiments

### View a Single Experiment

```bash
tensorboard --logdir=log/board/pretrain_RRD_Libero10-rrd-libero-10-(2025-11-22-20:20:37)
```

### Compare Multiple Experiments

```bash
tensorboard --logdir=log/board
```

This will show all experiments in the `log/board/` directory, allowing you to compare different runs.

## Remote Access

If you're running TensorBoard on a remote server, you can:

1. **SSH Port Forwarding**:
   ```bash
   ssh -L 6006:localhost:6006 user@remote-server
   ```
   Then access `http://localhost:6006` on your local machine.

2. **Direct Remote Access** (if firewall allows):
   ```bash
   tensorboard --logdir=log/board --host=0.0.0.0 --port=6006
   ```
   Then access `http://remote-server-ip:6006` from your browser.

## Useful TensorBoard Features

### 1. Filtering Metrics
- Use the search box to filter specific metrics
- Toggle metrics on/off by clicking the eye icon

### 2. Smoothing
- Adjust the smoothing slider to reduce noise in plots
- Useful for visualizing overall trends

### 3. Download Data
- Click the download icon to export metric data as CSV

### 4. Compare Runs
- When viewing multiple experiments, use the "Runs" panel to:
  - Show/hide specific runs
  - Compare metrics side-by-side
  - Align runs by step or time

## Troubleshooting

### TensorBoard Not Starting

If you get an error, ensure TensorBoard is installed:
```bash
pip install tensorboard
```

### No Data Showing

1. **Check log directory exists**: Ensure `log/board/` contains experiment folders
2. **Check experiment name**: Verify the experiment name matches your `--tag` argument
3. **Refresh browser**: Sometimes TensorBoard needs a refresh to load new data

### Missing Metrics

If `R_loss` or other metrics don't appear:
- Ensure the metric is in `agent.train_info` (it should be initialized automatically)
- Check that the metric is being logged via `args.logger.add_dict(info)`
- Verify the metric is added as a scalar: `args.logger.add_item(key, 'scalar')`

## Example: Viewing Your Current Training

For your current LIBERO-10 reward-only training:

```bash
# Start TensorBoard
cd /data/kmirakho/JustImagine/Randomized-Return-Decomposition
tensorboard --logdir=log/board

# Or view specific experiment
tensorboard --logdir=log/board/pretrain_RRD_Libero10-rrd-libero-10-\(2025-11-22-20:20:37\)
```

Then open `http://localhost:6006` in your browser to see:
- **R_loss**: Should show the reward network loss decreasing over time
- **TimeCost(sec)/train**: Training time per cycle
- **Episodes** and **Timesteps**: Dataset statistics

## Advanced: Custom Logging

To add custom metrics to TensorBoard, ensure they're in the appropriate info dictionary:

```python
# In your algorithm's train() method:
info = {
    'custom_metric': value,
    'R_loss': r_loss.item()
}
return info  # This gets logged via args.logger.add_dict(info)
```

The metric will automatically appear in TensorBoard if it's registered as a scalar in `train.py`.

