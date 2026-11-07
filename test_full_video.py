#!/usr/bin/env python3
"""
Full video test: GPEN-BFR-256 + SR scale=1 (same resolution, fastest config).
Processes entire video with optimized pipeline and saves output.
"""

import os
import sys
import cv2
import time
import numpy as np
import torch
import argparse
from tqdm import tqdm

os.environ['GPEN_NATIVE_OPS'] = '1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from face_enhancement_fast import FastFaceEnhancement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='../test_1min.mp4')
    parser.add_argument('--output', type=str, default='../test_1min_gpen256_sr1.mp4')
    parser.add_argument('--max-frames', type=int, default=0, help='0 = all frames')
    parser.add_argument('--start-time', type=float, default=0, help='Start time in seconds')
    parser.add_argument('--duration', type=float, default=0, help='Duration in seconds (0 = to end)')
    parser.add_argument('--model', type=str, default='GPEN-BFR-256', help='GPEN-BFR-256 or GPEN-BFR-512')
    parser.add_argument('--in-size', type=int, default=256, help='Face model input size (256 or 512)')
    parser.add_argument('--channel-multiplier', type=int, default=1, help='1 for BFR-256, 2 for BFR-512')
    parser.add_argument('--narrow', type=float, default=0.5, help='0.5 for BFR-256, 1.0 for BFR-512')
    parser.add_argument('--sr-scale', type=int, default=2, help='SR scale factor (1 or 2)')
    parser.add_argument('--no-compile', action='store_true')
    parser.add_argument('--no-fp16', action='store_true')
    args = parser.parse_args()

    sr_model_name = 'realesrnet' if args.sr_scale == 1 else 'realesrnet'

    print("=" * 70)
    print(f"GPEN Full Video: {args.model} + SR×{args.sr_scale} (optimized pipeline)")
    print("=" * 70)
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Cannot open {args.input}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Handle start time and duration
    start_frame = 0
    if args.start_time > 0:
        start_frame = int(args.start_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    if args.duration > 0:
        max_frames = int(args.duration * fps)
    elif args.max_frames > 0:
        max_frames = args.max_frames
    else:
        max_frames = total - start_frame

    print(f"  Video: {w}x{h} @ {fps:.1f}fps, {total} frames ({total/fps:.0f}s)")
    print(f"  Processing: frames {start_frame}-{start_frame + max_frames} ({max_frames} frames, {max_frames/fps:.1f}s)")
    print(f"  Config: {args.model}, in_size={args.in_size}, SR×{args.sr_scale}, compile={'off' if args.no_compile else 'on'}, FP16={'off' if args.no_fp16 else 'on'}")
    print()

    # Initialize
    enhancer = FastFaceEnhancement(
        base_dir='./',
        in_size=args.in_size,
        model=args.model,
        channel_multiplier=args.channel_multiplier,
        narrow=args.narrow,
        use_sr=True,
        sr_model='realesrnet',
        sr_scale=args.sr_scale,
        tile_size=0,
        alpha=1.0,
        device='cuda',
        use_compile=not args.no_compile,
        use_fp16=not args.no_fp16,
    )

    # Warmup
    enhancer.warmup(video_height=h, video_width=w, batch_sizes=[1])

    # Output video (scaled resolution based on SR factor)
    out_w = w * args.sr_scale
    out_h = h * args.sr_scale
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (out_w, out_h))

    torch.cuda.reset_peak_memory_stats()

    frame_times = []
    faces_total = 0
    n = 0
    t_start = time.time()

    pbar = tqdm(total=max_frames, desc="Processing", unit="frame",
                bar_format="{l_bar}{bar:30}{r_bar}")

    while n < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        torch.cuda.synchronize()
        t0 = time.time()

        result, orig_faces, enhanced_faces = enhancer.process(frame)

        torch.cuda.synchronize()
        elapsed = (time.time() - t0) * 1000

        writer.write(result)

        # Skip first 10 frames for warmup
        if n >= 10:
            frame_times.append(elapsed)
            faces_total += len(enhanced_faces)

            # Update progress bar with current FPS
            recent = frame_times[-50:] if len(frame_times) >= 50 else frame_times
            med = np.median(recent)
            fps_now = 1000.0 / med if med > 0 else 0
            pbar.set_postfix_str(f"{med:.0f}ms | {fps_now:.1f}FPS | faces={len(enhanced_faces)}")

        n += 1
        pbar.update(1)

    pbar.close()

    total_time = time.time() - t_start
    cap.release()
    writer.release()

    # Results
    ft = np.array(frame_times)
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Frames processed: {n}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Overall FPS: {n / total_time:.2f}")
    print()
    if len(ft) > 0:
        print(f"  Per-frame (excluding warmup):")
        print(f"    Median: {np.median(ft):.1f}ms ({1000/np.median(ft):.1f} FPS)")
        print(f"    Mean:   {np.mean(ft):.1f}ms ({1000/np.mean(ft):.1f} FPS)")
        print(f"    P5:     {np.percentile(ft, 5):.1f}ms")
        print(f"    P95:    {np.percentile(ft, 95):.1f}ms")
        print(f"    Min:    {ft.min():.1f}ms")
        print(f"    Max:    {ft.max():.1f}ms")
    print(f"  Faces enhanced: {faces_total}")
    print(f"  Avg faces/frame: {faces_total / max(1, len(ft)):.2f}")
    print(f"  GPU peak memory: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
    print(f"  Output: {args.output}")
    print("=" * 70)


if __name__ == '__main__':
    main()
