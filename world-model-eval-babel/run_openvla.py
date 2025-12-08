import argparse
from typing import Sequence

from utils import (
    rescale_bridge_action,
    discover_trials,
    discover_trials_from_videos,
    # predict,
    aggregate_model_results,
    print_results_table,
)
from transformers import AutoModelForVision2Seq, AutoProcessor
from world_model import WorldModel
import numpy as np
from PIL import Image
import mediapy as media
from tqdm import tqdm
import torch
from pathlib import Path

import einops


def rescale_libero_action_for_worldmodel(action: np.ndarray) -> np.ndarray:
    """
    Rescale OpenVLA's LIBERO actions to match the world model's expected format.
    
    OpenVLA outputs 7D actions roughly in [-1, 1] range (unnormalized from training).
    World model expects actions rescaled as:
    - world_vector (dims 0-2): from [-1.0, +1.0] -> [-1.75, +1.75]
    - rotation_delta (dims 3-5): from [-0.4, +0.4] -> [-1.4, +1.4]
    - gripper (dim 6): kept in [0, 1]
    
    Args:
        action: 7D action from OpenVLA [x, y, z, rx, ry, rz, gripper]
    
    Returns:
        7D rescaled action matching world model's training distribution
    """
    action_rescaled = action.copy()
    
    # Rescale world_vector (dims 0-2): from [-1.0, +1.0] to [-1.75, +1.75]
    # Formula: out = (in - low) / (high - low) * (post_max - post_min) + post_min
    # Since OpenVLA output is already roughly [-1, 1], we apply the same rescaling
    # that was used during world model training
    wv_low, wv_high = -1.0, 1.0
    wv_post_min, wv_post_max = -1.75, 1.75
    action_rescaled[0:3] = (action[0:3] - wv_low) / (wv_high - wv_low) * (wv_post_max - wv_post_min) + wv_post_min
    action_rescaled[0:3] = np.clip(action_rescaled[0:3], wv_post_min + 0.01, wv_post_max - 0.01)
    
    # Rescale rotation_delta (dims 3-5): from [-0.4, +0.4] to [-1.4, +1.4]
    rd_low, rd_high = -0.4, 0.4
    rd_post_min, rd_post_max = -1.4, 1.4
    action_rescaled[3:6] = (action[3:6] - rd_low) / (rd_high - rd_low) * (rd_post_max - rd_post_min) + rd_post_min
    action_rescaled[3:6] = np.clip(action_rescaled[3:6], rd_post_min + 0.01, rd_post_max - 0.01)
    
    # Gripper (dim 6) stays in [0, 1]
    action_rescaled[6] = np.clip(action[6], 0.0, 1.0)
    
    return action_rescaled


def evaluate_openvla(wm, vla, processor, trials, unnorm_key, retries=1, rollout_length=40,
                     save_video=False, video_out_dir=None, root_dir=None):
    """
    Rollout an OpenVLA model on a list of tasks, and return the score on each task.
    Arguments:
        wm: WorldModel
        vla: An OpenVLA model from `transformers`
        tasks: A list of N tasks in loaded from a json. See "put_carrot_on_plate.json" for an example of the format.
    Returns:
        scores: A list of N scores from the VLM corresponding to each input task.
    """
    results = []
    if save_video and video_out_dir:
        Path(video_out_dir).mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for trial in tqdm(trials, desc="Openvla trials"):
            # Resize to 256x256 for world model
            if "first_frame" in trial:
                start_frame = trial["first_frame"]
                # start_frame = np.array(Image.fromarray(trial["first_frame"]).resize((256, 256)))
            else:
                start_frame = np.array(Image.open(trial["trial_png"]).resize((256, 256)))
            
            # import pdb; pdb.set_trace()
            for r in range(retries):
                wm.reset(torch.from_numpy(start_frame).cuda().float() / 255.0)
                
                frames = [start_frame]
                frames_float = [start_frame.astype(np.float32) / 255.0]  # Keep float version for VLA input
                for step in range(rollout_length):
                    # Use float frame for VLA input (convert to uint8 only for PIL)
                    curr_frame_uint8 = (np.clip(frames_float[-1] * 255, 0, 255)).astype(np.uint8)
                    curr_frame = Image.fromarray(curr_frame_uint8)

                    prompt = f"In: What action should the robot take to {trial['instruction']}?\nOut:"
                    inputs = processor(prompt, curr_frame).to(device="cuda", dtype=torch.bfloat16)
                    actions = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

                    # Rescale actions to match world model's expected format
                    # a = rescale_bridge_action(a)
                    actions_rescaled = rescale_libero_action_for_worldmodel(actions)
                    a = torch.tensor(actions_rescaled).cuda()
                    # OpenVLA outputs 7-dim actions, while the world model was trained with up to 10-dim actions.
                    a = torch.cat([a, a.new_zeros(3)], dim=-1)  # pad with zeros
                    print(f"Step {step}: OpenVLA action {actions}")
                    print(f"Step {step}: Rescaled action {a.cpu().numpy()}")
                    
                    # a = torch.tensor([0.18749988079071045, 0.3234374523162842, 0, -0.04499995708465576, 0, -0.05624997615814209, -1, 0, 0, 0]).cuda()
                    for i, x in wm.generate_chunk(a):
                        new_frame_float = x[0, 0].cpu().numpy()  # Keep in [0, 1] range
                        new_frame_float = np.clip(new_frame_float, 0, 1)
                        frames_float.append(new_frame_float)
                        # Convert to uint8 only for final video
                        new_frame = (new_frame_float * 255).astype(np.uint8)
                        frames.append(new_frame)

                rollout_video = np.stack(frames)
                if save_video and video_out_dir:
                    trial_video = Path(trial["trial_video"])
                    target_dir = Path(video_out_dir)
                    if root_dir is not None:
                        try:
                            rel_parent = trial_video.parent.relative_to(Path(root_dir))
                            target_dir = target_dir / rel_parent
                        except ValueError:
                            target_dir = target_dir / trial_video.parent.name
                    else:
                        target_dir = target_dir / trial_video.parent.name
                    target_dir.mkdir(parents=True, exist_ok=True)
                    vid_name = trial_video.stem
                    out_name = f"{vid_name}_rollout.mp4"
                    media.write_video(str(target_dir / out_name), rollout_video, fps=20)
                # score = predict(rollout_video, trial)
                score = 0.0
                results.append({
                    "task_key": trial["task_key"],
                    "task_display": trial["task_display"],
                    "score": float(score),
                })
    return results

CHECKPOINTS_TO_KWARGS = {
    "bridge_v2_ckpt.pt": {
        "use_pixel_rope": True,
    },
    "mixed_openx_9robots_20frames_0p1actiondropout_580ksteps.pt": {
        "use_pixel_rope": False,
        "default_cfg": 3.0,
    },
}


def run(
    checkpoint_path: str = "mixed_openx_9robots_20frames_0p1actiondropout_580ksteps.pt",
    model_name: str = "openvla-7b",
    root_dir: str | None = None,
    *,
    rollout_length: int = 40,
    retries: int = 1,
    save_video: bool = False,
    video_out_dir: str | None = None,
) -> dict[str, dict[str, float]]:
    """Run the OpenVLA evaluation loop."""

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}; download it manually and retry.")
    ckpt_key = ckpt_path.name
    ckpt_kwargs = CHECKPOINTS_TO_KWARGS.get(ckpt_key, {})
    wm = WorldModel(ckpt_path, **ckpt_kwargs)
    unnorm_key_dict = {"openvla-7b": "bridge_orig", 
                       "openvla-7b-finetuned-libero-goal": "libero_goal", 
                       "openvla-7b-finetuned-libero-action": "libero_action"} 
    unnorm_key = unnorm_key_dict.get(model_name, "bridge_orig")
    processor = AutoProcessor.from_pretrained(f"openvla/{model_name}", trust_remote_code=True, local_files_only=True,)
    vla = AutoModelForVision2Seq.from_pretrained(
        f"openvla/{model_name}",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
        local_files_only=True,
    ).cuda().eval()

    if root_dir is None:
        raise ValueError("root_dir must be provided; pass --root-dir to point at the evaluation dataset.")
    # trials = discover_trials(root_dir)
    trials = discover_trials_from_videos(root_dir)
    print(f"Discovered {len(trials)} trials.")

    results = evaluate_openvla(
        wm,
        vla,
        processor,
        trials,
        unnorm_key=unnorm_key,
        rollout_length=rollout_length,
        retries=retries,
        save_video=save_video,
        video_out_dir=video_out_dir,
        root_dir=root_dir,
    )

    agg = aggregate_model_results(results)
    print_results_table(agg)
    return agg


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an OpenVLA policy in the Bridge world model")
    parser.add_argument("--checkpoint-path", default="mixed_openx_9robots_20frames_0p1actiondropout_580ksteps.pt")
    parser.add_argument("--model-name", default="openvla-7b")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--rollout-length", type=int, default=40)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-out-dir")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, dict[str, float]]:  # pragma: no cover - CLI entry point
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    return run(
        checkpoint_path=args.checkpoint_path,
        model_name=args.model_name,
        root_dir=args.root_dir,
        rollout_length=args.rollout_length,
        retries=args.retries,
        save_video=args.save_video,
        video_out_dir=args.video_out_dir,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
