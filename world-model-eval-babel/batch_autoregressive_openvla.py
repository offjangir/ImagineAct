#!/usr/bin/env python3
"""Batch wrapper to run autoregressive OpenVLA evaluation over an entire dataset.

This script discovers video files under a root directory, pairs them with corresponding
action `.npz` files (same stem), and invokes `run_autoregressive_eval` for each.

Outputs a JSON (and optional CSV) summary of per-video metrics.

Example:
  python batch_autoregressive_openvla.py \
      --dataset-root libero_eval_data/libero_goal \
      --checkpoint mixed_openx_9robots_20frames_0p1actiondropout_580ksteps.pt \
      --model-name openvla-7b \
      --output-json libero_goal_results.json \
      --video-ext mp4 --video-ext avi \
      --n-clips 1 --n-context-frames 1
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import csv
from typing import List, Dict
from eval_autoregressive_openvla import run_autoregressive_eval

VIDEO_EXT_DEFAULT = ["mp4", "avi", "mkv", "webm"]


def discover_videos(root: Path, exts: List[str]) -> List[Path]:
    videos = []
    ext_set = {e.lower().lstrip('.') for e in exts}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower().lstrip('.') in ext_set:
            # require matching .npz and .json sidecar
            npz = p.with_suffix('.npz')
            js = p.with_suffix('.json')
            if npz.exists() and js.exists():
                videos.append(p)
    return sorted(videos)


def write_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    fieldnames = sorted(rows[0].keys())
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description="Batch run autoregressive OpenVLA evaluation across dataset.")
    ap.add_argument('--dataset-root', required=True, type=Path, help='Root directory containing videos + .npz + .json sidecars')
    ap.add_argument('--checkpoint', required=True, type=str, help='Path to world model checkpoint')
    ap.add_argument('--model-name', default='openvla-7b', type=str, help='OpenVLA model name (repo subdir)')
    ap.add_argument('--video-ext', action='append', default=None, help='Video extension to include (can repeat). Default: mp4 avi mkv webm')
    ap.add_argument('--output-json', type=Path, default=Path('batch_autoregressive_results.json'), help='Output JSON summary path')
    ap.add_argument('--output-csv', type=Path, default=None, help='Optional CSV summary path')
    ap.add_argument('--frame-skip', type=int, default=1, help='Frame skip for loading videos')
    ap.add_argument('--action-dim', type=int, default=10, help='Action dimension (world model)')
    ap.add_argument('--timesteps', type=int, default=1000)
    ap.add_argument('--sampling-timesteps', type=int, default=10)
    ap.add_argument('--patch-size', type=int, default=2)
    ap.add_argument('--model-dim', type=int, default=1024)
    ap.add_argument('--layers', type=int, default=16)
    ap.add_argument('--heads', type=int, default=16)
    ap.add_argument('--window-len', type=int, default=None)
    ap.add_argument('--horizon', type=int, default=1)
    ap.add_argument('--n-context-frames', type=int, default=1)
    ap.add_argument('--n-clips', type=int, default=1)
    ap.add_argument('--input-h', type=int, default=256)
    ap.add_argument('--input-w', type=int, default=256)
    ap.add_argument('--output-gif', type=str, default='eval_autoregressive.gif', help='Base GIF name per video (clips appended)')
    ap.add_argument('--limit', type=int, default=None, help='Optional limit on number of videos processed')
    args = ap.parse_args()

    exts = args.video_ext if args.video_ext is not None else VIDEO_EXT_DEFAULT
    videos = discover_videos(args.dataset_root, exts)
    if args.limit is not None:
        videos = videos[:args.limit]

    print(f"[Batch] Discovered {len(videos)} video(s) under {args.dataset_root}")
    if not videos:
        print('[Batch] No videos found matching criteria; exiting.')
        return

    results_summary: List[Dict] = []
    for idx, vid in enumerate(videos):
        print(f"\n[Batch] ({idx+1}/{len(videos)}) Evaluating video: {vid}")
        try:
            res = run_autoregressive_eval(
                checkpoint_path=args.checkpoint,
                model_name=args.model_name,
                video_path=str(vid),
                action_path=None,  # inferred
                input_h=args.input_h,
                input_w=args.input_w,
                frame_skip=args.frame_skip,
                action_dim=args.action_dim,
                timesteps=args.timesteps,
                sampling_timesteps=args.sampling_timesteps,
                patch_size=args.patch_size,
                model_dim=args.model_dim,
                layers=args.layers,
                heads=args.heads,
                window_len=args.window_len,
                horizon=args.horizon,
                n_context_frames=args.n_context_frames,
                output_gif=args.output_gif,
                n_clips=args.n_clips,
            )
            # Flatten per-video summary
            summary_row = {
                'video_path': res['video_path'],
                'n_clips': res['n_clips'],
                'total_frames_generated': res['total_frames_generated'],
                'avg_mse': res['avg_mse'],
                'min_mse': res['min_mse'],
                'max_mse': res['max_mse'],
            }
            results_summary.append(summary_row)
        except Exception as e:
            print(f"[Batch][ERROR] Failed on {vid}: {e}")
            results_summary.append({
                'video_path': str(vid),
                'n_clips': args.n_clips,
                'total_frames_generated': 0,
                'avg_mse': None,
                'min_mse': None,
                'max_mse': None,
                'error': str(e),
            })

    # Write JSON (full rows)
    args.output_json.write_text(json.dumps(results_summary, indent=2))
    print(f"\n[Batch] Wrote JSON summary: {args.output_json}")

    if args.output_csv:
        write_csv(args.output_csv, results_summary)
        print(f"[Batch] Wrote CSV summary: {args.output_csv}")

    # Print quick aggregate
    valid = [r for r in results_summary if r.get('avg_mse') is not None]
    if valid:
        avg_all = sum(r['avg_mse'] for r in valid) / len(valid)
        print(f"[Batch] Aggregate average MSE across {len(valid)} videos: {avg_all:.6f}")
    else:
        print('[Batch] No valid MSE results computed.')


if __name__ == '__main__':
    main()
