# Outstanding Issues & Implementation Details

1. Checkpointing and Training Design

- Does this repository support checkpointing for experiments?
- What are the best practices or expectations for saving cumulative rewards?

2. Offline Training with Libero

- Offline training with Libero may require a custom wrapper inside the `envs` directory.
- The Libero dataset should contain (obs, act, rew) tuples for compatible training.
- For this experiment, the intention is to train only a reward model in an offline manner, without updating an agent policy.

3. Libero Data Format

- What is the native storage format for Libero data, and does it require conversion (e.g., to RLDS or another format) for use in training pipelines?

4. OpenVLA

- On what data is OpenVLA trained?
- What camera images or input view(s) are required for OpenVLA models?



# RRD Paper Summary & Contribution

## Reward Model Design

The paper highlights that constructing a robust reward function is critical for learning effective policies. Instead of static, engineered reward design, the paradigm here is online, trial-and-error based reward design. Here, an agent:
- First learns a proxy reward function from collected experience.
- Refines this reward model iteratively based on environmental feedback, allowing more accurate alignment with task objectives.

## The Proposed Algorithm: Randomized Return Decomposition (RRD)

- RRD is presented as a scalable approach for reward modeling in long-horizon tasks.
- The algorithm trains the reward model to predict episodic return based on random subsequences of a trajectory, enforcing that learned proxy rewards can approximately reconstruct the overall trajectory reward using only a subset of state-action pairs.
- This surrogate optimization enables return decomposition on shorter subsequences, making it practical for high-dimensional or long task horizons.

### Loss Functions

- Two loss functions are supported:
    - L-RD (standard least-squares return decomposition)
    - L-Rand-RD (randomized surrogate used in RRD)
- RRD uses `L-Rand-RD` by default (`--alg=rrd`), while RRD-LRD uses `L-RD` with bias correction (`--alg=rrd --rrd_bias_correction=True`).

Buffer (rrd_buffer.py): Now provides:
rrd_rews: Full episodic return R_ep(τ) (sum of all rewards in trajectory)
rrd_ep_length: Full episode length T (needed for scaling)
Training (rrd_torch.py): Now implements the paper's formulation:
Predicts per-step rewards for the subsequence: r_1, ..., r_k
Sums them: sum_pred = Σ r_i
Scales to predict full episodic return: rrd = sum_pred * (T / k)
Compares to actual episodic return: loss = MSE(rrd, R_ep)

### Scalability

- RRD is more scalable and stable than previous return decomposition methods, and it integrates well with off-policy learning for better sample efficiency.
- Empirical results show that RRD achieves superior sample efficiency and policy quality compared to RRD-LRD, with regularized reward redistribution leading to more accurate reward modeling.

# Libero Benchmark: Description and Insights

## Task Structure

- LIBERO provides 130 language-conditioned robot tasks inspired by human activities, grouped into four benchmark suites to probe transfer along spatial, object, goal, or mixed dimensions.
- Suites: LIBERO-SPATIAL, LIBERO-OBJECT, LIBERO-GOAL, LIBERO-100 (last includes 100 heterogeneous tasks).
- Example: LIBERO-SPATIAL focuses on spatial relationships, requiring the agent to memorize varying object arrangements; LIBERO-OBJECT emphasizes object diversity; LIBERO-GOAL is about diverse motion/behavior goals.

## Data Provided

- High-quality human teleoperation demonstrations for all tasks.
- RGB images from workspace and wrist cameras.
- Proprioceptive data.
- Language-based task specifications.
- PDDL scene descriptions.

## Key Findings

- Architecture (e.g., Transformer vs. RNN, CNN) selection has significant bearing on performance.
- Sequential finetuning outperforms all evaluated lifelong learning approaches for forward transfer.
- Pretraining with raw language embeddings was not substantially better than using task IDs.
- Simple supervised pretraining can hinder subsequent reinforcement learning in the lifelong setting.


# OpenVLA: Training, Data, and Evaluation

## Model Training

- OpenVLA is a 7B-parameter Visual Language Action model, trained on 970k real-world robot demonstrations, using LLAMA-2, DINOv2, and SigLIP features.
- Trained over 21,500 A100 GPU-hours at batch size 2048; inference (bfloat16) requires ~15GB on an RTX 4090 (6Hz throughput).
- Fine-tuning is supported via LoRA (low-rank adaptation), allowing efficient updates on consumer GPUs by adjusting batch size and accumulation steps.

## Data Input Requirements

- OpenVLA models are fine-tuned and evaluated using demonstration datasets (e.g., RLDS-converted Libero, Open X-Embodiment mixtures).
- Primary camera input: Standardized RGB workspace views—center-cropped; random crop data augmentation is used during fine-tuning, so center cropping is recommended at inference for best performance.
- Control signals are expected at 5–10Hz (high-frequency data should be downsampled), and consistent task strategies are encouraged for robust learning.

## Data Conversion

- Datasets not originally in RLDS format should be converted for compatibility.
- Example scripts and templates are provided to convert Libero or custom data to RLDS for OpenVLA training.

---

# Dataset Storage & Compatibility

- LIBERO: Stores workspace/wrist RGB, proprioception, language specifications, and scene descriptions; format may be HDF5 or RLDS depending on downstream pipeline use.
- OpenVLA: Prefers RLDS-formatted data for demonstration ingestion, but custom PyTorch wrappers are an alternative.

---

# Practical Notes for Fine-Tuning & Evaluation

- Best finetuning results require demonstration data to match training expectations (control frequency, camera setup, action format).
- OpenVLA evaluation scripts and model checkpoints are available; logging and visualization are integrated (e.g., Weights & Biases, result files).
- Package versions for evaluation and training are specified to ensure reproducibility.


## Command to run training
`
python scripts/train.py     --env libero-10     --env_type normal     --alg rrd     --basis_alg sac     --rrd_reward_only True     --tag pretrain_RRD_Libero10     --libero_dataset_path /data/kmirakho/JustImagine/modified_libero_rlds/libero_10_no_noops/1.0.0     --libero_image_size 256     --cuda     --epochs 10     --cycles 50     --iterations 100     --train_batches 50     --batch_size 32     --rrd_batch_size 32     --rrd_sample_size 32     --save_freq 2     --checkpoint_dir log/checkpoints     --save_final True
`


