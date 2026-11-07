#!/usr/bin/env python3
"""
GPEN v5 Benchmark: Cross-frame batching (DDColor-style).
Reads N frames at a time, batches all face crops through FaceGAN + ParseNet
in single forward passes, maximizing GPU utilization.
"""

import os
import sys
import cv2
import time
import numpy as np
import torch
import argparse
from collections import defaultdict

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


def print_stats(all_timings, total_times, n_faces_total, n_frames, batch_size):
    """Print comprehensive timing breakdown."""
    print("\n" + "="*70)
    print(f"TIMING BREAKDOWN (batch={batch_size}, per-frame averages in ms)")
    print("="*70)

    components = ['sr_ms', 'detect_ms', 'align_ms', 'facegan_ms', 'parse_ms', 'composite_ms']
    component_names = ['SR background', 'Face detection', 'Face alignment',
                       'FaceGAN (batched)', 'ParseNet (batched)', 'Composite+blend']

    for comp, name in zip(components, component_names):
        vals = all_timings.get(comp, [])
        print(percentile_str(vals, name))

    print("\n" + "-"*70)
    print(percentile_str(total_times, "TOTAL per frame"))

    total = np.array(total_times)
    fps = 1000.0 / np.mean(total) if np.mean(total) > 0 else 0
    fps_med = 1000.0 / np.median(total) if np.median(total) > 0 else 0
    print(f"\n  Throughput: {fps:.1f} FPS (mean), {fps_med:.1f} FPS (median)")
    print(f"  Frames processed: {n_frames}")
    print(f"  Total faces enhanced: {n_faces_total}")
    print(f"  Avg faces/frame: {n_faces_total/max(1,n_frames):.2f}")
    print(f"  Batch size: {batch_size}")

    # Component contribution
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
            print(f"    {name:22s}: {m:7.1f}ms ({pct:5.1f}%) {bar}")

    print("="*70)


def run_benchmark(enhancer, cap, batch_size, n_frames, warmup_frames, writer=None,
                  fps_orig=25.0):
    """Run the benchmark with given batch size."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    all_timings = defaultdict(list)
    total_times = []
    n_faces_total = 0
    n_timed = 0
    total_processed = 0

    while total_processed < n_frames + warmup_frames:
        # Read a batch of frames
        batch_frames = []
        for _ in range(batch_size):
            if total_processed + len(batch_frames) >= n_frames + warmup_frames:
                break
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            batch_frames.append(frame)

        if not batch_frames:
            break

        timing = {}
        torch.cuda.synchronize()
        t0 = time.time()

        results = enhancer.process_batch(batch_frames, timing=timing)

        torch.cuda.synchronize()
        elapsed_ms = (time.time() - t0) * 1000
        per_frame_ms = elapsed_ms / len(batch_frames)

        # Write output + record timing
        for result, orig_faces, enhanced_faces in results:
            if writer is not None:
                writer.write(result)

            if total_processed >= warmup_frames:
                total_times.append(per_frame_ms)
                # Distribute batch timing evenly per frame
                for k, v in timing.items():
                    all_timings[k].append(v / len(batch_frames))
                n_faces_total += len(enhanced_faces)
                n_timed += 1

            total_processed += 1

        if n_timed > 0 and n_timed % 100 < batch_size:
            recent = total_times[-100:] if len(total_times) >= 100 else total_times
            med = np.median(recent)
            fps = 1000.0 / med if med > 0 else 0
            print(f"  Frame {n_timed}: "
                  f"last-100 median={med:.1f}ms/frame ({fps:.1f} FPS)")

    return all_timings, total_times, n_faces_total, n_timed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='../test_1min.mp4')
    parser.add_argument('--frames', type=int, default=1000)
    parser.add_argument('--warmup-frames', type=int, default=20)
    parser.add_argument('--batch-sizes', type=str, default='1,4,8',
                        help='Comma-separated batch sizes to test')
    parser.add_argument('--no-sr', action='store_true', help='Disable SR')
    parser.add_argument('--no-compile', action='store_true')
    parser.add_argument('--no-fp16', action='store_true')
    parser.add_argument('--sr-scale', type=int, default=2)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]

    print("="*70)
    print("GPEN v5: Cross-Frame Batch Benchmark (DDColor-style)")
    print("="*70)
    print(f"  Video: {args.video}")
    print(f"  Frames: {args.frames}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  SR: {'off' if args.no_sr else f'on (scale={args.sr_scale})'}")
    print(f"  torch.compile: {'off' if args.no_compile else 'on'}")
    print(f"  FP16: {'off' if args.no_fp16 else 'on'}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.version.cuda}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open {args.video}")
        return

    fps_orig = cap.get(cv2.CAP_PROP_FPS)
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {w_orig}x{h_orig} @ {fps_orig}fps, {total_frames} frames")

    # Initialize pipeline
    enhancer = FastFaceEnhancement(
        base_dir='./',
        in_size=512, model='GPEN-BFR-512',
        channel_multiplier=2, narrow=1,
        use_sr=not args.no_sr, sr_model='realesrnet',
        sr_scale=args.sr_scale, tile_size=0,
        alpha=1.0, device='cuda',
        use_compile=not args.no_compile,
        use_fp16=not args.no_fp16,
    )

    # Warmup ALL batch sizes we'll test
    warmup_bs = sorted(set([1] + batch_sizes))
    enhancer.warmup(video_height=h_orig, video_width=w_orig, batch_sizes=warmup_bs)

    # Run each batch size
    all_results = {}
    for bs in batch_sizes:
        print(f"\n{'='*70}")
        print(f">>> BATCH SIZE = {bs}")
        print(f"{'='*70}")

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        writer = None
        if args.output:
            out_file = args.output.replace('.mp4', f'_b{bs}.mp4')
            out_h = h_orig * (args.sr_scale if not args.no_sr else 1)
            out_w = w_orig * (args.sr_scale if not args.no_sr else 1)
            writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'),
                                     fps_orig, (out_w, out_h))

        timings, frame_times, faces, n = run_benchmark(
            enhancer, cap, bs, args.frames, args.warmup_frames, writer, fps_orig)

        if writer:
            writer.release()

        print_stats(timings, frame_times, faces, n, bs)
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  GPU peak memory: {peak_mem:.0f} MB")

        avg_ms = np.mean(frame_times) if frame_times else 0
        med_ms = np.median(frame_times) if frame_times else 0
        all_results[bs] = {
            'avg_fps': 1000/avg_ms if avg_ms > 0 else 0,
            'med_fps': 1000/med_ms if med_ms > 0 else 0,
            'avg_ms': avg_ms,
            'med_ms': med_ms,
            'peak_mem_mb': peak_mem,
            'faces': faces,
        }

    # Summary comparison
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(">>> BATCH SIZE COMPARISON")
        print(f"{'='*70}")
        print(f"  {'Batch':>6} {'Avg ms/f':>10} {'Med ms/f':>10} {'Avg FPS':>10} "
              f"{'Med FPS':>10} {'Speedup':>10} {'VRAM MB':>10}")
        print(f"  {'-'*66}")
        base_fps = list(all_results.values())[0]['avg_fps']
        for bs, r in all_results.items():
            speedup = r['avg_fps'] / base_fps if base_fps > 0 else 0
            print(f"  {bs:>6} {r['avg_ms']:>10.1f} {r['med_ms']:>10.1f} "
                  f"{r['avg_fps']:>10.2f} {r['med_fps']:>10.2f} "
                  f"{speedup:>9.2f}x {r['peak_mem_mb']:>10.0f}")

    cap.release()


if __name__ == '__main__':
    main()
