#!/usr/bin/env python3
"""
GPEN v4 Benchmark: Tests the optimized face_enhancement_fast.py pipeline.
- Skips baseline (as requested)
- Per-component timing breakdown
- 1000 frames for stable statistics
"""

import os
import sys
import cv2
import time
import numpy as np
import torch
import argparse
from collections import defaultdict

# Use native PyTorch ops (no custom CUDA that doesn't support FP16)
os.environ['GPEN_NATIVE_OPS'] = '1'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_enhancement_fast import FastFaceEnhancement


def percentile_str(values, name):
    if not values:
        return f"  {name}: no data"
    arr = np.array(values)
    return (f"  {name}: median={np.median(arr):.1f}ms, "
            f"mean={np.mean(arr):.1f}ms, "
            f"P5={np.percentile(arr,5):.1f}ms, "
            f"P95={np.percentile(arr,95):.1f}ms, "
            f"min={arr.min():.1f}ms, max={arr.max():.1f}ms")


def print_stats(all_timings, total_times, n_faces_total, n_frames):
    """Print comprehensive timing breakdown."""
    print("\n" + "="*70)
    print("TIMING BREAKDOWN (per frame, milliseconds)")
    print("="*70)
    
    # Per-component
    components = ['sr_ms', 'detect_ms', 'align_ms', 'facegan_ms', 'parse_blend_ms', 'composite_ms']
    component_names = ['SR background', 'Face detection', 'Face alignment', 'FaceGAN', 'Parse+blend', 'Composite']
    
    for comp, name in zip(components, component_names):
        vals = all_timings.get(comp, [])
        print(percentile_str(vals, name))
    
    print("\n" + "-"*70)
    print(percentile_str(total_times, "TOTAL per frame"))
    
    total = np.array(total_times)
    fps = 1000.0 / np.median(total) if np.median(total) > 0 else 0
    print(f"\n  Throughput: {fps:.1f} FPS (median)")
    print(f"  Frames processed: {n_frames}")
    print(f"  Total faces enhanced: {n_faces_total}")
    print(f"  Avg faces/frame: {n_faces_total/max(1,n_frames):.2f}")
    
    # Component contribution (median-based)
    comp_medians = {}
    total_comp_time = 0
    for comp, name in zip(components, component_names):
        vals = all_timings.get(comp, [])
        m = np.median(vals) if vals else 0
        comp_medians[name] = m
        total_comp_time += m
    
    if total_comp_time > 0:
        print("\n  Component breakdown (% of median total):")
        for name, m in sorted(comp_medians.items(), key=lambda x: -x[1]):
            pct = 100.0 * m / total_comp_time
            bar = "█" * int(pct / 2)
            print(f"    {name:20s}: {m:7.1f}ms ({pct:5.1f}%) {bar}")

    print("="*70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='../test_1min.mp4')
    parser.add_argument('--frames', type=int, default=1000)
    parser.add_argument('--warmup-frames', type=int, default=20,
                        help='Frames to skip for timing after warmup')
    parser.add_argument('--no-sr', action='store_true', help='Disable SR')
    parser.add_argument('--no-compile', action='store_true', help='Disable torch.compile')
    parser.add_argument('--no-fp16', action='store_true', help='Disable FP16')
    parser.add_argument('--sr-scale', type=int, default=2)
    parser.add_argument('--output', type=str, default=None,
                        help='Save output frames to this video file')
    args = parser.parse_args()

    print("="*70)
    print("GPEN v4 Optimized Pipeline Benchmark")
    print("="*70)
    print(f"  Video: {args.video}")
    print(f"  Frames: {args.frames}")
    print(f"  SR: {'off' if args.no_sr else f'on (scale={args.sr_scale})'}")
    print(f"  torch.compile: {'off' if args.no_compile else 'on'}")
    print(f"  FP16: {'off' if args.no_fp16 else 'on'}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.version.cuda}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video}")
        return
    
    fps_orig = cap.get(cv2.CAP_PROP_FPS)
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {w_orig}x{h_orig} @ {fps_orig}fps, {total_frames} frames")

    # Initialize optimized pipeline
    enhancer = FastFaceEnhancement(
        base_dir='./',
        in_size=512,
        model='GPEN-BFR-512',
        channel_multiplier=2,
        narrow=1,
        use_sr=not args.no_sr,
        sr_model='realesrnet',
        sr_scale=args.sr_scale,
        tile_size=0,
        alpha=1.0,
        device='cuda',
        use_compile=not args.no_compile,
        use_fp16=not args.no_fp16,
    )

    # Warmup JIT + CUDA
    enhancer.warmup(video_height=h_orig, video_width=w_orig)

    # Video writer for output
    writer = None
    if args.output:
        out_h = h_orig * (args.sr_scale if not args.no_sr else 1)
        out_w = w_orig * (args.sr_scale if not args.no_sr else 1)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps_orig, (out_w, out_h))

    # Process frames
    all_timings = defaultdict(list)
    total_times = []
    n_faces_total = 0
    n_timed = 0

    for frame_idx in range(args.frames + args.warmup_frames):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        timing = {}
        torch.cuda.synchronize()
        t0 = time.time()

        result, orig_faces, enhanced_faces = enhancer.process(frame, timing=timing)

        torch.cuda.synchronize()
        elapsed_ms = (time.time() - t0) * 1000

        # Write output
        if writer is not None:
            writer.write(result)

        # Record timing after warmup
        if frame_idx >= args.warmup_frames:
            total_times.append(elapsed_ms)
            for k, v in timing.items():
                all_timings[k].append(v)
            n_faces_total += len(enhanced_faces)
            n_timed += 1

            if (n_timed) % 100 == 0:
                med = np.median(total_times[-100:])
                fps = 1000.0 / med if med > 0 else 0
                faces = sum(1 for t in total_times[-100:] if True)  # count frames
                print(f"  Frame {n_timed}/{args.frames}: "
                      f"last-100 median={med:.1f}ms ({fps:.1f} FPS), "
                      f"faces={len(enhanced_faces)}")

    cap.release()
    if writer:
        writer.release()

    # Print results
    print_stats(all_timings, total_times, n_faces_total, n_timed)

    # GPU memory
    print(f"\n  GPU peak memory: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")
    print(f"  GPU current memory: {torch.cuda.memory_allocated()/1024**2:.0f} MB")


if __name__ == '__main__':
    main()
