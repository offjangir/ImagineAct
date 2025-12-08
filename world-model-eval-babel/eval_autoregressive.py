import torch
import imageio
from pathlib import Path
from model import DiT
from vae import VAE
from diffusion import Diffusion
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import einops
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision import transforms


def load_single_video(
    video_path: str,
    action_path: str,
    input_h: int,
    input_w: int,
    frame_skip: int = 1,
    action_dim: int = 10,
    n_clips: int = 1,
):
    """Load a single video file and divide it into n_clips, returning clips and actions.
    
    Actions are aligned with video frames: actions[i] corresponds to the i-th frame
    of the original video (before frame_skip). After applying frame_skip, both
    video and actions are downsampled together to maintain alignment.
    
    Args:
        n_clips: Number of clips to divide the video into. Each clip will have
                 approximately equal length.
    
    Returns:
        List of tuples: [(clip_0, actions_0), (clip_1, actions_1), ...]
    """
    video_path = Path(video_path)
    action_path = Path(action_path)
    
    # Load actions - these are indexed by original video frame number
    actions_full = np.load(action_path)["arr_0"]
    assert actions_full.shape[1] == action_dim, f"Unexpected action dim: {actions_full.shape[1]} != {action_dim}"
    action_length = len(actions_full)
    
    # Load full video - align with action length using frame-based time calculation
    video = EncodedVideo.from_path(video_path, decode_audio=False)
    fps = video._container.streams.video[0].guessed_rate
    # Calculate end time based on action length (actions are frame-indexed)
    end_sec = action_length / fps
    clip = video.get_clip(start_sec=0.0, end_sec=end_sec)["video"]
    clip = einops.rearrange(clip, "c t h w -> t h w c")
    
    # Ensure video and actions have the same length (take minimum)
    video_length = len(clip)
    min_length = min(video_length, action_length)
    clip = clip[:min_length]
    actions_full = actions_full[:min_length]
    
    # Apply frame skip to both video and actions together
    clip = clip[::frame_skip]
    actions_full = actions_full[::frame_skip]
    
    # Normalize and resize
    clip = clip.float() / 255.0
    clip = einops.rearrange(clip, "t h w c -> t c h w")
    transform = transforms.Resize((int(input_h), int(input_w)))
    clip = transform(clip)
    clip = einops.rearrange(clip, "t c h w -> t h w c")
    actions_full = torch.from_numpy(actions_full).float()
    
    # Divide video into n_clips
    total_frames = clip.shape[0]
    clips = []
    actions_list = []
    
    if n_clips <= 1:
        # Return single clip (entire video)
        return [(clip, actions_full)]
    
    # Calculate clip boundaries
    frames_per_clip = total_frames // n_clips
    
    for i in range(n_clips):
        start_idx = i * frames_per_clip
        # Last clip gets any remaining frames
        if i == n_clips - 1:
            end_idx = total_frames
        else:
            end_idx = (i + 1) * frames_per_clip
        
        clip_segment = clip[start_idx:end_idx]
        actions_segment = actions_full[start_idx:end_idx]
        
        clips.append(clip_segment)
        actions_list.append(actions_segment)
    
    return list(zip(clips, actions_list))


@torch.no_grad()
def run_autoregressive_eval(
    checkpoint_path: str,
    video_path: str,
    action_path: str | None = None,
    input_h: int = 256,
    input_w: int = 256,
    frame_skip: int = 1,
    action_dim: int = 10,
    timesteps: int = 1000,
    sampling_timesteps: int = 10,
    patch_size: int = 2,
    model_dim: int = 1024,
    layers: int = 16,
    heads: int = 16,
    window_len: int | None = None,
    horizon: int = 1,
    n_context_frames: int = 1,
    output_gif: str = "eval_autoregressive.gif",
    n_clips: int = 1,
):
    """
    Autoregressive evaluation: Divide video into clips, then for each clip,
    start with frame 0 and iteratively generate each subsequent frame using
    the previously generated frame as context.
    
    Process for each clip:
    1. Start with ground truth frame 0 (or first n_context_frames)
    2. Use action 0 to generate frame 1
    3. Use generated frame 1 as context, action 1 to generate frame 2
    4. Continue until all frames in the clip are generated
    
    Args:
        n_clips: Number of clips to divide the video into. Each clip will be
                 evaluated independently with autoregressive generation.
    """
    assert torch.cuda.is_available(), "CUDA required for evaluation"
    device = torch.device("cuda")

    print(f"[Eval] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Load video and actions, divide into clips
    if action_path is None:
        action_path = str(Path(video_path).with_suffix(".npz"))
    print(f"[Eval] Loading video: {video_path}")
    print(f"[Eval] Loading actions from: {action_path}")
    print(f"[Eval] Dividing video into {n_clips} clips")
    
    clip_action_pairs = load_single_video(
        video_path=video_path,
        action_path=action_path,
        input_h=input_h,
        input_w=input_w,
        frame_skip=frame_skip,
        action_dim=action_dim,
        n_clips=n_clips,
    )
    
    print(f"[Eval] Loaded {len(clip_action_pairs)} clip(s)")
    for clip_idx, (clip, actions_clip) in enumerate(clip_action_pairs):
        print(f"[Eval] Clip {clip_idx}: {clip.shape[0]} frames, {actions_clip.shape[0]} actions")
    
    # --- Load models ---
    # Find max frames across all clips to set model max_frames
    max_clip_frames = max(clip.shape[0] for clip, _ in clip_action_pairs)
    
    vae = VAE().to(device)
    model = DiT(
        in_channels=vae.vae.config.latent_channels,
        patch_size=patch_size,
        dim=model_dim,
        num_layers=layers,
        num_heads=heads,
        action_dim=action_dim,
        max_frames=max_clip_frames,
    ).to(device)

    diffusion = Diffusion(
        timesteps=timesteps,
        sampling_timesteps=sampling_timesteps,
        device=device,
    ).to(device)

    # Load weights (prefer EMA)
    if "ema" in ckpt:
        print("[Eval] Using EMA weights for generation.")
        model.load_state_dict(ckpt["ema"])
    else:
        print("[Eval] Using raw model weights.")
        model.load_state_dict(ckpt["model"])

    model.eval()
    vae.eval()

    # Create output directory based on n_clips and n_context_frames
    output_dir = Path(f"n_clips_{n_clips}_context_{n_context_frames}")
    output_dir.mkdir(exist_ok=True)
    print(f"[Eval] Saving outputs to directory: {output_dir}")
    
    # Process each clip separately
    all_mse_scores = []
    
    for clip_idx, (x_gt_clip, actions_clip) in enumerate(clip_action_pairs):
        print(f"\n{'='*60}")
        print(f"[Eval] Processing Clip {clip_idx}/{len(clip_action_pairs)-1}")
        print(f"{'='*60}")
        import pdb; pdb.set_trace()
        # Move to device and add batch dimension
        x_gt = x_gt_clip.to(device).unsqueeze(0)  # [1, T, H, W, C]
        actions = actions_clip.to(device).unsqueeze(0)  # [1, T, action_dim]
        
        T = x_gt.shape[1]  # total frames in this clip
        print(f"[Eval] Clip {clip_idx} has {T} frames, {actions.shape[1]} actions")
        
        print(f"[Eval] Starting autoregressive generation for clip {clip_idx}...")
        print(f"[Eval] Using {n_context_frames} context frame(s), generating one frame at a time")
        
        # Start with ground truth context frames
        x_generated = x_gt[:, :n_context_frames].clone()  # [1, n_context_frames, H, W, C]
        current_frame = n_context_frames
        
        mse_scores = []
        generated_frames_list = [x_generated[0, i].cpu() for i in range(n_context_frames)]
        
        # Autoregressive loop: generate ONE frame at a time
        # Frame 0 -> action 0 -> Frame 1
        # Frame 1 -> action 1 -> Frame 2
        # etc.
        pbar = tqdm(total=T - n_context_frames, desc=f"Clip {clip_idx} - Generating frames")
        
        while current_frame < T:
            # Generate exactly one frame at a time
            # Action index: action[i] generates frame[i+1]
            # So to generate frame[current_frame], we use action[current_frame - 1]
            action_idx = current_frame - 1
            
            # Prepare input: use last n_context_frames of generated sequence as context
            x_context = x_generated[:, -n_context_frames:]  # [1, n_context_frames, H, W, C]
            
            # Encode context frames to latents
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                latents_context = vae.encode(x_context)  # [1, n_context_frames, H_lat, W_lat, C_lat]
                
                # Create input for generation: context latents + noise for 1 new frame
                B, T_ctx, H_lat, W_lat, C_lat = latents_context.shape
                noise_frame = torch.randn(
                    (B, 1, H_lat, W_lat, C_lat),
                    device=device,
                    dtype=latents_context.dtype
                )
                latents_input = torch.cat([latents_context, noise_frame], dim=1)  # [1, n_context_frames+1, ...]
                
                # Actions: need actions for each frame in the sequence
                # For context frames: use actions from (current_frame - n_context_frames) to (current_frame - 1)
                # For new frame: use action[current_frame - 1] (the action that generates current_frame)
                start_action_idx = max(0, current_frame - n_context_frames)
                end_action_idx = min(actions.shape[1], current_frame)
                
                if end_action_idx > start_action_idx:
                    actions_context = actions[:, start_action_idx:end_action_idx]
                    # Pad or trim to match n_context_frames
                    if actions_context.shape[1] < n_context_frames:
                        # Pad with first action if needed
                        first_action = actions_context[:, :1]
                        padding = first_action.expand(-1, n_context_frames - actions_context.shape[1], -1)
                        actions_context = torch.cat([padding, actions_context], dim=1)
                    elif actions_context.shape[1] > n_context_frames:
                        actions_context = actions_context[:, -n_context_frames:]
                else:
                    # Fallback: use last available action
                    actions_context = actions[:, -1:].expand(-1, n_context_frames, -1)
                
                # Action for the new frame (action that generates current_frame)
                if action_idx >= 0 and action_idx < actions.shape[1]:
                    action_new = actions[:, action_idx:action_idx+1]  # [1, 1, action_dim]
                else:
                    action_new = actions[:, -1:]  # Use last action as fallback
                
                # Combine: context actions + action for new frame
                actions_for_gen = torch.cat([actions_context, action_new], dim=1)  # [1, n_context_frames+1, action_dim]
                
                # Generate using diffusion (will generate 1 new frame)
                generated_latents = diffusion.generate(
                    model,
                    latents_input,
                    actions_for_gen,
                    n_context_frames=n_context_frames,
                    n_frames=n_context_frames + 1,  # Generate 1 new frame
                    window_len=window_len,
                    horizon=1,  # Generate 1 frame at a time
                )
                
                # Extract only the newly generated frame (last frame)
                new_latent = generated_latents[:, -1:]  # [1, 1, H_lat, W_lat, C_lat]
                
                # Decode to image
                new_frame = vae.decode(new_latent)  # [1, 1, H, W, C]
                
                # Append generated frame
                x_generated = torch.cat([x_generated, new_frame], dim=1)
                
                # Compute MSE for the generated frame
                if current_frame < T:
                    mse = F.mse_loss(new_frame[0, 0], x_gt[0, current_frame]).item()
                    mse_scores.append(mse)
                    generated_frames_list.append(new_frame[0, 0].cpu())
                    pbar.set_postfix({"frame": current_frame, "mse": f"{mse:.6f}"})
            
            current_frame += 1
            pbar.update(1)
        
        pbar.close()
        
        # Save GIF for this clip in the output directory
        clip_output_gif = output_dir / output_gif.replace(".gif", f"_clip{clip_idx:03d}.gif")
        print(f"\n[Eval] Saving generated clip {clip_idx} to {clip_output_gif}")
        video_np = torch.stack(generated_frames_list, dim=0)
        video_np = (video_np.float().clamp(0, 1) * 255).byte().numpy()
        imageio.mimsave(str(clip_output_gif), video_np, fps=8)
        
        # Also save ground truth for comparison
        gt_gif = output_dir / clip_output_gif.name.replace(".gif", "_ground_truth.gif")
        gt_np = (x_gt[0].float().clamp(0, 1) * 255).byte().cpu().numpy()
        imageio.mimsave(str(gt_gif), gt_np, fps=8)
        print(f"[Eval] Saved ground truth clip {clip_idx} to {gt_gif}")
        
        # Compute statistics for this clip
        clip_avg_mse = sum(mse_scores) / len(mse_scores) if mse_scores else 0.0
        print(f"[Eval] Clip {clip_idx} - Frames generated: {len(mse_scores)}, Avg MSE: {clip_avg_mse:.6f}")
        
        all_mse_scores.extend(mse_scores)
    
    # Compute overall statistics
    global_avg_mse = sum(all_mse_scores) / len(all_mse_scores) if all_mse_scores else 0.0
    print(f"\n{'='*60}")
    print(f"========== AUTOREGRESSIVE EVALUATION SUMMARY ==========")
    print(f"Total clips processed: {len(clip_action_pairs)}")
    print(f"Total frames generated: {len(all_mse_scores)}")
    print(f"Global Average MSE per frame: {global_avg_mse:.6f}")
    print(f"Min MSE: {min(all_mse_scores):.6f}" if all_mse_scores else "N/A")
    print(f"Max MSE: {max(all_mse_scores):.6f}" if all_mse_scores else "N/A")
    print("=======================================================")
    print(f"{'='*60}\n")
    
    return all_mse_scores


if __name__ == "__main__":
    import fire
    fire.Fire(run_autoregressive_eval)

