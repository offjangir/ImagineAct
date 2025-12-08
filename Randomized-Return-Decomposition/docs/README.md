# Randomized Return Decomposition (RRD)

This is a TensorFlow implementation for our paper [Learning Long-Term Reward Redistribution via Randomized Return Decomposition](https://arxiv.org/abs/2111.13485) accepted by ICLR 2022.

## Requirements
1. Python 3.6.13
2. gym == 0.18.3
3. TensorFlow == 1.12.0
4. BeautifulTable == 0.8.0
5. opencv-python == 4.5.3.56

## Running Commands

Run the following commands to reproduce our main results shown in section 4.1.

```bash
python train.py --tag='RRD Ant-v2' --alg=rrd --basis_alg=sac --env=Ant-v2
python train.py --tag='RRD-L(RD) Ant-v2' --alg=rrd --basis_alg=sac --rrd_bias_correction=True --env=Ant-v2
```

The following commands to switch the back-end algorithm of RRD.

```bash
python train.py --tag='RRD-TD3 Ant-v2' --alg=rrd --basis_alg=td3 --env=Ant-v2
python train.py --tag='RRD-DDPG Ant-v2' --alg=rrd --basis_alg=ddpg --env=Ant-v2
```

We include an *unofficial* implementation of IRCR for the ease of baseline comparison.  
Please refer to [tgangwani/GuidanceRewards](https://github.com/tgangwani/GuidanceRewards) for the official implementation of IRCR.

```bash
python train.py --tag='IRCR-SAC Ant-v2' --alg=ircr --basis_alg=sac --env=Ant-v2
python train.py --tag='IRCR-TD3 Ant-v2' --alg=ircr --basis_alg=td3 --env=Ant-v2
python train.py --tag='IRCR-DDPG Ant-v2' --alg=ircr --basis_alg=ddpg --env=Ant-v2
```

The following commands support the experiments on Atari games with episodic rewards.  

```bash
python train.py --tag='RRD-DQN Assault' --alg=rrd --basis_alg=dqn --env=Assault
python train.py --tag='IRCR-DQN Assault' --alg=ircr --basis_alg=dqn --env=Assault
```

**Note:**
The implementation of RRD upon DQN on the Atari benchmark has not been well tuned. We release this interface only for the ease of future studies.




## New things
1. does this repo do checkpointing?
2. offline training with libero
    - probably need a wrapper over libero that goes inside envs directory
    - libero dataset needs to have (obs, act, rew) tuples
    - I do not want to train an agent policy, just the reward model in an offline manner
3. whether to save cumulative rewards 
4. what is openvla trained on?
5. what format is libero data stored in? does it need conversion to someother format?


## What is this paper doing - 

The design of such a reward function is crucial to the performance of
the learned policies. To avoid the unintended behaviors induced by misspecified reward engineering, a common paradigm is considering the reward design as an online problem within the trial-and-error loop of reinforcement learning. The agent first learns a proxy reward function from the experience data and then optimizes its policy based on the learned per-step rewards. By iterating this procedure and interacting with the environment, the agent is able to continuously refine its reward model so that the learned proxy reward function can better approximate the actual objective given by the environmental feedback. 


Our proposed algorithm, randomized return decomposition (RRD), establishes a surrogate optimization of return decomposition to improve the scalability in long-horizon tasks. In this surrogate problem, the reward model is trained to predict the episodic return from a random subsequence of the agent trajectory, i.e., we conduct a structural constraint that the learned proxy rewards can approximately reconstruct environmental trajectory return from a small subset of state-action pairs.

One limitation of the least-squares-based return decomposition method specified by Eq. (4) is its scalability in terms of the computation costs. Note that the trajectory-wise episodic reward is the only environmental supervision for reward modeling. Computing the loss function LRD(θ) with a single episodic reward label requires to enumerate all state-action pairs along the whole trajectory. This computation procedure can be expensive in numerous situations, e.g., when the task horizon T is quite long, or the state space S is high-dimensional. To address this practical barrier, recent works focus on designing reward redistribution mechanisms that can be easily integrated in complex tasks. We will discuss the implementation subtlety of existing methods in section 4.

In this section, we introduce our approach, randomized return decomposition (RRD), which sets
up a surrogate optimization problem of the least-squares-based return decomposition. The proposed
surrogate objective allows us to conduct return decomposition on short subsequences of agent trajectories, which is scalable in long-horizon tasks.

In this section, we show that our approach is an interpolation
between between the return decomposition paradigm and uniform reward redistribution, which can
be controlled by the hyper-parameter K used in the sampling distribution (see Eq. (7))


2 loss functions supported - 
Use L-RD over L-Rand-RD?

RRD uses L-Rand-RD: --alg=rrd 
RRD-Lrd used L-RD: --alg=rrd --rrd_bias_correction=True

In comparison, RRD is a more
scalable and stable implementation which can better integrate with off-policy learning for improving sample efficiency. 

As presented in Figure 1, RRD achieves higher sample efficiency than RRD-LRD in most testing environments. The quality of the
learned policy of RRD is also better than that of RRD-LRD. It suggests that the regularized reward redistribution can better approximate the actual environmental objective.


# Libero - 
For benchmarking purpose, LIBERO generates 130 language-conditioned robot manipulation tasks
inspired by human activities [22] and, grouped into four suites. The four task suites are designed
to examine distribution shifts in the object types, the spatial arrangement of objects, the task goals,
or the mixture of the previous three 
To support sample-efficient learning, we provide high-quality
human-teleoperated demonstration data for all tasks. Our extensive experiments
present several insightful or even unexpected discoveries: sequential finetuning
outperforms existing lifelong learning methods in forward transfer, no single visual
encoder architecture excels at all types of knowledge transfer, and naive supervised
pretraining can hinder agents’ performance in the subsequent LLDM.

 Based on
our experiments, we make several insightful or even unexpected observations:
1. Policy architecture design is as crucial as lifelong learning algorithms. The transformer
architecture is better at abstracting temporal information than a recurrent neural network.
Vision transformers work well on tasks with rich visual information (e.g., a variety of
objects). Convolution networks work well when tasks primarily need procedural knowledge.
2. While the lifelong learning algorithms we evaluated are effective at preventing forgetting,
they generally perform worse than sequential finetuning in terms of forward transfer.
3. Our experiment shows that using pretrained language embeddings of semantically-rich task
descriptions yields performance no better than using those of the task IDs.
2
4. Basic supervised pretraining on a large-scale offline dataset can have a negative impact on
the learner’s downstream performance in LLDM

LIBERO has four task suites: LIBERO-SPATIAL,
LIBERO-OBJECT, LIBERO-GOAL, and LIBERO-100. The first three task suites are curated to
disentangle the transfer of declarative and procedural knowledge (as mentioned in (T1)), while
LIBERO-100 is a suite of 100 tasks with entangled knowledge transfer.

LIBERO-X = LIBERO-SPATIAL + LIBERO-OBJECT + LIBERO-GOAL all have 10 tasks
and are designed to investigate the controlled transfer of knowledge about spatial information
(declarative), objects (declarative), and task goals (procedural). Specifically, all tasks in LIBEROSPATIAL request the robot to place a bowl, among the same set of objects, on a plate. But there are two identical bowls that differ only in their location or spatial relationship to other objects. Hence, to successfully complete LIBERO-SPATIAL, the robot needs to continually learn and memorize new spatial relationships. All tasks in LIBERO-OBJECT request the robot to pick-place a unique object. Hence, to accomplish LIBERO-OBJECT, the robot needs to continually learn and memorize new object types. All tasks in LIBERO-GOAL share the same objects with fixed spatial relationships but differ only in the task goal. Hence, to accomplish LIBERO-GOAL, the robot needs to continually earn new knowledge about motions and behaviors. More details are in Appendix C.

LIBERO-100 LIBERO-100 contains 100 tasks that entail diverse object interactions and versatile
motor skills. In this paper, we split LIBERO-100 into 90 short-horizon tasks (LIBERO-90) and 10
long-horizon tasks (LIBERO-LONG). LIBERO-90 serves as the data source for pretraining (T5)
and LIBERO-LONG for downstream evaluation of lifelong learning algorithms.

### Libero dataset
What are included
✓ RGB images from workspace and wrist cameras
✓ Proprioception
✓ Language task specifications
✓ PDDL scene descriptions

## OpenVLA
OpenVLA, a 7B-parameter open-source VLA trained on a diverse collection
of 970k real-world robot demonstrations. OpenVLA builds on a Llama 2 language
model combined with a visual encoder that fuses pretrained features from DINOv2
and SigLIP. 
We also explore compute efficiency; as a separate contribution, we show that OpenVLA can be
fine-tuned on consumer GPUs via modern low-rank adaptation methods and served
efficiently via quantization without a hit to downstream success rate. Finally, we
release model checkpoints, fine-tuning notebooks, and our PyTorch codebase with
built-in support for training VLAs at scale on Open X-Embodiment datasets.

what kind of camera images does it need?


The OpenX (Open X-Embodiment) trajectory dataset does not natively contain reward signals for each transition or episode. The dataset is primarily composed of state-action pairs, observations (including images), and task instructions, but it does not include explicit reward annotations for reinforcement learning.
The dataset is designed for imitation learning and behavior cloning, where the agent learns from expert demonstrations rather than from reward-based feedback.​

The absence of reward signals means that the dataset is not directly suitable for standard reinforcement learning algorithms that require reward labels for each step.

The final OpenVLA model is trained on a cluster of 64 A100 GPUs for 14 days, or a total of
21,500 A100-hours, using a batch size of 2048. During inference, OpenVLA requires 15GB of GPU
memory when loaded in bfloat16 precision (i.e., without quantization) and runs at approximately
6Hz on one NVIDIA RTX 4090 GPU (without compilation, speculative decoding, or other inference
speed-up tricks). We can further reduce the memory footprint of OpenVLA during inference via
quantization, without compromising performance in real-world robotics tasks, as shown in Section 5.4.

This repository was built using Python 3.10, but should be backwards compatible with any Python >= 3.8. We require PyTorch 2.2.* -- installation instructions can be found here. The latest version of this repository was developed and thoroughly tested with:

PyTorch 2.2.0, torchvision 0.17.0, transformers 4.40.1, tokenizers 0.19.1, timm 0.9.10, and flash-attn 2.5.5

Now, launch the LoRA fine-tuning script, as shown below. Note that --batch_size==16 with --grad_accumulation_steps==1 requires ~72 GB GPU memory. If you have a smaller GPU, you should reduce --batch_size and increase --grad_accumulation_steps to maintain an effective batch size that is large enough for stable training. If you have multiple GPUs and wish to train via PyTorch Distributed Data Parallel (DDP), simply set --nproc-per-node in the torchrun command below to the number of available GPUs.


To LoRA fine-tune on a different dataset, you can download the dataset from the Open X-Embodiment (OXE) mixture (see this custom script for an example of how to download datasets from OXE). Alternatively, if you have a custom dataset that is not part of OXE, you can either (a) convert the dataset to the RLDS format which is compatible with our fine-tuning script (see this repo for instructions on this), or (b) use your own custom PyTorch Dataset wrapper (see comments in vla-scripts/finetune.py for instructions). We recommend option (a) for most users; the RLDS dataset and dataloader are tested more extensively since we used these for all of our pretraining and fine-tuning experiments.

For option (a), after you converted your dataset to RLDS, you need to register it with our data loader, by registering a dataset config here and a dataset transform function here.

### Finetuning
OpenVLA typically requires fine-tuning on a small demonstration dataset (~100 demos) from your target domain robot. Out-of-the-box, it only works well on domains from the training dataset.

Best practices for fine-tuning data collection: If your setup passed the above two sanity checks, the issue may not be in model training, but in the data you fine-tuned the model with. Some best practices for data collection:

Collect at a control frequency around 5-10Hz. OpenVLA is not trained with action chunking, empirically the model struggles with high-frequency data. If your robot setup uses a high-frequency controller (eg 50 Hz), consider downsampling your actions to 5Hz. Verify first that your robot can still solve the task when using 5Hz actions (ie repeat sanity check (1) above with 5Hz actions)
Avoid pauses / small actions during data collection. Because OpenVLA is trained without action chunking, the model can be sensitive to idle actions in the fine-tuning data. If your data contains steps in which the robot barely moves, the model may "get stuck" in these steps at inference time. Try to collect fine-tuning demonstrations with continuous, slow movement.
Ensure sufficient data coverage. If you plan to test the model with some variation, e.g. different initial positions of objects, make sure that your fine-tuning data contains sufficient diversity of such conditions as well, e.g. shows demonstrations with diverse initial conditions.
Use consistent task strategies during data collection. This is not a hard constraint, but may make your life easier. Try to demonstrate tasks in consistent ways, e.g. approach objects from the same side, perform sub-steps in the same order even if they could be performed in arbitrary sequences. Being consistent gives you a less multi-modal fine-tuning dataset, which makes the modeling problem easier.

### Evaluating openvla on Libero -

(Optional) To download the modified versions of the LIBERO datasets that we used in our fine-tuning experiments, run the command below. This will download the LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-10 datasets in RLDS data format (~10 GB total). You can use these to fine-tune OpenVLA or train other methods. This step is optional since we provide pretrained OpenVLA checkpoints below. (Also, you can find the script we used to generate the modified datasets in raw HDF5 format here and the code we used to convert these datasets to the RLDS format here.)

Notes:

The evaluation script will run 500 trials by default (10 tasks x 50 episodes each). You can modify the number of trials per task by setting --num_trials_per_task. You can also change the random seed via --seed.
NOTE: Setting --center_crop True is important because we fine-tuned OpenVLA with random crop augmentations (we took a random crop with 90% area in every training sample, so at test time we simply take the center 90% crop).
The evaluation script logs results locally. You can also log results in Weights & Biases by setting --use_wandb True and specifying --wandb_project <PROJECT> and --wandb_entity <ENTITY>.
The results reported in our paper were obtained using Python 3.10.13, PyTorch 2.2.0, transformers 4.40.1, and flash-attn 2.5.5 on an NVIDIA A100 GPU, averaged over three random seeds. Please stick to these package versions. Note that results may vary slightly if you use a different GPU for evaluation due to GPU nondeterminism in large models (though we have tested that results were consistent across different machines with A100 GPUs).