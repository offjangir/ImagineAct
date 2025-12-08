import argparse
import os
import sys
import datetime
import torch
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
import imageio

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
# from .world_model import WorldModel

try:
    # When run as a module from the project root, e.g.
    #   python -m scripts.evaluate_offline_openvla_ac
    # we can import from the `scripts` package.
    from scripts.train_offline_openvla_ac import load_openvla_model
except ModuleNotFoundError:
    # When run as a plain script from the `scripts/` directory, e.g.
    #   python scripts/evaluate_offline_openvla_ac.py
    # Python's `sys.path[0]` becomes the `scripts/` directory itself,
    # so we need a relative import instead.
    from train_offline_openvla_ac import load_openvla_model


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """Match OpenVLA eval normalization for the gripper dimension."""
    action = np.array(action, copy=True)
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """Flip gripper sign convention (dataset vs. simulator)."""
    action = np.array(action, copy=True)
    action[..., -1] = action[..., -1] * -1.0
    return action


def resize_image(img, resize_size):
    """
    Resize image using the same scheme as OpenVLA's LIBERO utils.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_libero_image(obs, resize_size):
    """
    Extract and preprocess LIBERO observation image for OpenVLA.
    Mirrors openvla_RL.experiments.robot.libero.libero_utils.get_libero_image.
    """
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # Rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    # Convert numpy array to PIL.Image for OpenVLA processor
    return Image.fromarray(img)


def get_world_model_image(prev_img, action, resize_size):
    pass
    # generated_latents = diffusion.generate(
    #                 model,
    #                 latents_input,
    #                 actions_for_gen,
    #                 n_context_frames=n_context_frames,
    #                 n_frames=n_context_frames + 1,  # Generate 1 new frame
    #                 window_len=window_len,
    #                 horizon=1,  # Generate 1 frame at a time
    #             )
    # new_latent = generated_latents[:, -1:]  # [1, 1, H_lat, W_lat, C_lat]
    # new_frame = vae.decode(new_latent)  # [1, 1, H, W, C]



def prepare_processor_inputs(processor, image, task_description, device):
    """Tokenize image/text pair and move tensors to the correct device / dtype."""
    inputs = processor(images=image, text=task_description, return_tensors="pt")
    processed = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if k == "pixel_values":
                processed[k] = v.to(device=device, dtype=torch.bfloat16)
            else:
                processed[k] = v.to(device)
        else:
            processed[k] = v
    return processed


def extract_openvla_features(vla_model, processor, image, task_description, device):
    """Match the training-time feature pipeline: tokenizer -> embeddings + vision patches."""
    inputs = prepare_processor_inputs(processor, image, task_description, device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    
    with torch.no_grad():
        language_features = vla_model.get_input_embeddings()(input_ids)
        pixel_values = inputs["pixel_values"]
        if len(pixel_values.shape) == 5:
            pixel_values = pixel_values.reshape(-1, *pixel_values.shape[2:])
        vision_features = vla_model.vision_backbone(pixel_values)
    
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    return language_features, vision_features, attention_mask


def compute_action_logits_from_features(
    vla_model,
    language_features,
    vision_features,
    attention_mask,
    unnorm_key,
):
    """Reuse the decoder-only path (language embeddings + projected vision patches)."""
    projector = getattr(vla_model, "projector", None)
    if projector is None:
        raise ValueError("Projector not found in vla_model")
    
    projector_dtype = next(projector.parameters()).dtype
    vision_features = vision_features.to(dtype=projector_dtype)
    if vision_features.dim() == 2:
        vision_features = vision_features.unsqueeze(0)
    
    projected_patch_embeddings = projector(vision_features)
    
    if hasattr(vla_model, "language_model"):
        decoder = vla_model.language_model
    elif hasattr(vla_model, "llm_backbone") and hasattr(vla_model.llm_backbone, "llm"):
        decoder = vla_model.llm_backbone.llm
    else:
        raise ValueError("Decoder (language_model or llm_backbone.llm) not found in vla_model")
    
    decoder_dtype = next(decoder.parameters()).dtype
    language_features = language_features.to(dtype=decoder_dtype)
    
    multimodal_embeddings = torch.cat(
        [
            language_features[:, :1, :],
            projected_patch_embeddings,
            language_features[:, 1:, :],
        ],
        dim=1,
    )
    
    multimodal_attention_mask = None
    if attention_mask is not None:
        num_patches = projected_patch_embeddings.shape[1]
        patch_attention_mask = torch.ones(
            (attention_mask.shape[0], num_patches),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        multimodal_attention_mask = torch.cat(
            [attention_mask[:, :1], patch_attention_mask, attention_mask[:, 1:]], dim=1
        )
    
    decoder_outputs = decoder(
        input_ids=None,
        inputs_embeds=multimodal_embeddings,
        attention_mask=multimodal_attention_mask,
        return_dict=True,
        use_cache=False,
    )
    
    if not hasattr(decoder_outputs, "logits"):
        raise ValueError("Decoder output is missing logits")
    
    logits = decoder_outputs.logits
    action_dim = vla_model.get_action_dim(unnorm_key)
    action_logits = logits[:, -action_dim:, :]
    return action_logits


def decode_action_from_logits(vla_model, action_logits, unnorm_key):
    """Convert decoder argmax tokens into continuous actions via OpenVLA bins."""
    vocab_size = getattr(vla_model, "vocab_size", vla_model.config.text_config.vocab_size)
    action_tokens = torch.argmax(action_logits, dim=-1).cpu().numpy()  # [B, action_dim]
    discretized_actions = vocab_size - action_tokens
    discretized_actions = np.clip(discretized_actions - 1, 0, vla_model.bin_centers.shape[0] - 1)
    normalized_actions = vla_model.bin_centers[discretized_actions]
    
    action_stats = vla_model.get_action_stats(unnorm_key)
    mask = action_stats.get("mask", np.ones_like(action_stats["q01"], dtype=bool))
    action_high = np.array(action_stats["q99"])
    action_low = np.array(action_stats["q01"])
    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )
    return actions


def save_rollout_video(frames, task_description, episode_idx, success, output_root="rollouts"):
    """Save evaluation rollout frames to an MP4, mirroring OpenVLA logging."""
    if not frames:
        return
    date_str = datetime.datetime.now().strftime("%Y_%m_%d")
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    task_slug = task_description.lower().replace(" ", "_").replace("\n", "_")
    rollout_dir = os.path.join(output_root, date_str)
    os.makedirs(rollout_dir, exist_ok=True)
    filename = f"{timestamp}--episode={episode_idx}--success={success}--task={task_slug}.mp4"
    video_path = os.path.join(rollout_dir, filename)
    with imageio.get_writer(video_path, fps=30) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"Saved rollout video: {video_path}", flush=True)


def load_actor_critic_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load OpenVLA model (frozen backbone + decoder with LoRA)
    # We only need the VLA for policy actions; critic and reward are optional here.
    # The training script always used the same OpenVLA checkpoint, so we reuse that.
    # NOTE: This assumes you pass --openvla_checkpoint consistent with training.
    return checkpoint


def evaluate_policy_in_libero(
    checkpoint_path: str,
    openvla_checkpoint: str,
    task_name: str,
    num_episodes: int,
    device: str = "cuda:0",
    max_steps: int = 200,
):
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load base OpenVLA model from HuggingFace (vision + language backbone)
    vla_model, processor = load_openvla_model(openvla_checkpoint, device=device)
    vla_model.eval()
    
    # Restore only the parts we actually trained (decoder / LoRA / action head).
    # We intentionally keep the vision backbone and base language model weights
    # from HuggingFace, since they were frozen during training.
    if "vla_model_state_dict" in checkpoint:
        full_state = checkpoint["vla_model_state_dict"]
        model_state = vla_model.state_dict()
        
        # Filter checkpoint keys to only those that exist in the current model
        # to avoid strict mismatches between PEFT-wrapped training model and
        # plain HF model at evaluation.
        filtered_state = {}
        for k, v in full_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered_state[k] = v
        
        missing_before = set(model_state.keys()) - set(filtered_state.keys())
        unexpected_before = set(full_state.keys()) - set(filtered_state.keys())
        print(
            f"[Eval] Loading {len(filtered_state)}/{len(full_state)} parameters from checkpoint "
            f"into OpenVLA model (skipping backbone / mismatched keys).",
            flush=True,
        )
        if len(missing_before) > 0:
            print(
                f"[Eval] Skipped {len(missing_before)} model params that had no matching checkpoint entry.",
                flush=True,
            )
        if len(unexpected_before) > 0:
            print(
                f"[Eval] Ignored {len(unexpected_before)} checkpoint params not present in the eval model (e.g., PEFT wrappers).",
                flush=True,
            )
        
        model_state.update(filtered_state)
        vla_model.load_state_dict(model_state)
    
    # Determine un-normalization key for action decoding (matches OpenVLA LIBERO eval)
    unnorm_key = "libero_goal"
    if hasattr(vla_model, "norm_stats") and isinstance(vla_model.norm_stats, dict):
        if unnorm_key not in vla_model.norm_stats and f"{unnorm_key}_no_noops" in vla_model.norm_stats:
            unnorm_key = f"{unnorm_key}_no_noops"
    
    # Create LIBERO env for a specific LIBERO Goal task
    # task_name should correspond to a BDDL file stem in libero_goal, e.g. "put_the_bowl_on_the_plate"
    bddl_dir = os.path.join(get_libero_path("bddl_files"), "libero_goal")
    bddl_file = os.path.join(bddl_dir, f"{task_name}.bddl")
    if not os.path.exists(bddl_file):
        raise FileNotFoundError(f"LIBERO BDDL file not found for task '{task_name}': {bddl_file}")
    
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 256,
        "camera_widths": 256,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    
    success_count = 0
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        step = 0
        rollout_frames = []
        
        # LIBERO exposes the language instruction on the environment
        task_description = getattr(env, "language_instruction", task_name)
        
        while not done and step < max_steps:
            # Convert LIBERO observation dict to image expected by OpenVLA
            # Follows OpenVLA's get_libero_image helper (uses obs["agentview_image"])
            image = get_libero_image(obs, resize_size=256)
            
            with torch.no_grad():
                language_features, vision_features, attention_mask = extract_openvla_features(
                    vla_model, processor, image, task_description, device
                )
                action_logits = compute_action_logits_from_features(
                    vla_model,
                    language_features,
                    vision_features,
                    attention_mask,
                    unnorm_key,
                )
                actions = decode_action_from_logits(vla_model, action_logits, unnorm_key)
            
            # Convert to numpy row [action_dim]
            action = actions[0].astype(np.float32)
            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)
            
            obs, reward, done, info = env.step(action.tolist())
            rollout_frames.append(np.array(image))
            step += 1
        
        if isinstance(info, dict) and "success" in info:
            success = bool(info["success"])
        else:
            # Fallback: treat positive final reward as success
            success = float(reward) > 0.0
        
        if success:
            success_count += 1
        print(f"Episode {ep + 1}/{num_episodes}: success={success}")
        save_rollout_video(rollout_frames, task_description, ep + 1, success)
    
    success_rate = success_count / max(num_episodes, 1)
    print(f"\nEvaluation finished: success_rate={success_rate:.3f} over {num_episodes} episodes")
    return success_rate


def evaluate_policy_in_worldmodel(
    checkpoint_path: str,
    openvla_checkpoint: str,
    world_model_checkpoint: str,
    task_name: str,
    num_episodes: int,
    device: str = "cuda:0",
    max_steps: int = 200,
):
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load base OpenVLA model from HuggingFace (vision + language backbone)
    vla_model, processor = load_openvla_model(openvla_checkpoint, device=device)
    vla_model.eval()
    
    # Restore only the parts we actually trained (decoder / LoRA / action head).
    # We intentionally keep the vision backbone and base language model weights
    # from HuggingFace, since they were frozen during training.
    if "vla_model_state_dict" in checkpoint:
        full_state = checkpoint["vla_model_state_dict"]
        model_state = vla_model.state_dict()
        
        # Filter checkpoint keys to only those that exist in the current model
        # to avoid strict mismatches between PEFT-wrapped training model and
        # plain HF model at evaluation.
        filtered_state = {}
        for k, v in full_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered_state[k] = v
        
        missing_before = set(model_state.keys()) - set(filtered_state.keys())
        unexpected_before = set(full_state.keys()) - set(filtered_state.keys())
        print(
            f"[Eval] Loading {len(filtered_state)}/{len(full_state)} parameters from checkpoint "
            f"into OpenVLA model (skipping backbone / mismatched keys).",
            flush=True,
        )
        if len(missing_before) > 0:
            print(
                f"[Eval] Skipped {len(missing_before)} model params that had no matching checkpoint entry.",
                flush=True,
            )
        if len(unexpected_before) > 0:
            print(
                f"[Eval] Ignored {len(unexpected_before)} checkpoint params not present in the eval model (e.g., PEFT wrappers).",
                flush=True,
            )
        
        model_state.update(filtered_state)
        vla_model.load_state_dict(model_state)
    
    # Determine un-normalization key for action decoding (matches OpenVLA LIBERO eval)
    unnorm_key = "libero_goal"
    if hasattr(vla_model, "norm_stats") and isinstance(vla_model.norm_stats, dict):
        if unnorm_key not in vla_model.norm_stats and f"{unnorm_key}_no_noops" in vla_model.norm_stats:
            unnorm_key = f"{unnorm_key}_no_noops"

    ckpt_path = Path(checkpoint_path)
    ckpt_key = ckpt_path.name
    # ckpt_kwargs = CHECKPOINTS_TO_KWARGS.get(ckpt_key, {})
    # wm = WorldModel(world_model_checkpoint, **ckpt_kwargs)
    # wm.reset(torch.from_numpy(start_frame).cuda().float() / 255.0)
    # prompt = f"In: What action should the robot take to {trial['instruction']}?\nOut:"
    


def main():
    parser = argparse.ArgumentParser(description="Evaluate offline OpenVLA actor-critic in LIBERO goal simulator")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to offline OpenVLA AC checkpoint (e.g., log/checkpoints/offline_openvla_ac_libero_goal_epoch_40.pt)",
    )
    parser.add_argument(
        "--openvla_checkpoint",
        type=str,
        default="openvla/openvla-7b-finetuned-libero-goal",
        help="OpenVLA base checkpoint used during training",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="put_the_bowl_on_the_plate",
        help="LIBERO Goal task BDDL stem, e.g. 'put_the_bowl_on_the_plate'",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for evaluation, e.g. cuda:0 or cpu",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Max steps per episode",
    )
    parser.add_argument(
        "--sim_env",
        type=str,
        choices=["libero", "worldmodel"],
        default="libero",
        help="Simulation environment to use for evaluation",
    )
    
    args = parser.parse_args()
    if args.sim_env == "libero":
        evaluate_policy_in_libero(
            checkpoint_path=args.checkpoint_path,
            openvla_checkpoint=args.openvla_checkpoint,
            task_name=args.task_name,
            num_episodes=args.num_episodes,
            device=args.device,
            max_steps=args.max_steps,
        )
    elif args.sim_env == "worldmodel":
        evaluate_policy_in_worldmodel(
            checkpoint_path=args.checkpoint_path,
            openvla_checkpoint=args.openvla_checkpoint,
            world_model_checkpoint="path/to/world_model_checkpoint.pt",
            task_name=args.task_name,
            num_episodes=args.num_episodes,
            device=args.device,
            max_steps=args.max_steps,
        )
    else:
        raise NotImplementedError("Only 'libero' and 'worldmodel' sim_env are implemented in this script.")


if __name__ == "__main__":
    main()


