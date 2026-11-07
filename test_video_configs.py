#!/usr/bin/env python3
"""
GPEN Configuration Comparison Benchmark.
Tests multiple model + SR combinations to find the fastest config.
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


def run_config(config, video_path, n_frames, warmup_frames):
    """Run benchmark for a single config dict."""
    cap = cv2.VideoCapture(video_path)
    fps_orig = cap.get(cv2.CAP_PROP_FPS)
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    name = config['name']
    print(f"\n{'='*70}")
    print(f">>> CONFIG: {name}")
    print(f"{'='*70}")
    print(f"  model={config['model']}, in_size={config['in_size']}, "
          f"channel_multiplier={config['channel_multiplier']}, narrow={config['narrow']}")
    print(f"  sr_scale={config['sr_scale']}, use_sr={config['use_sr']}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        enhancer = FastFaceEnhancement(
            base_dir='./',
            in_size=config['in_size'],
            model=config['model'],
            channel_multiplier=config['channel_multiplier'],
            narrow=config['narrow'],
            use_sr=config['use_sr'],
            sr_model='realesrnet',
            sr_scale=config['sr_scale'],
            tile_size=0,
            alpha=1.0,
            device='cuda',
            use_compile=True,
            use_fp16=True,
        )
    except Exception as e:
        print(f"  FAILED to load: {e}")
        cap.release()
        return None

    # Determine output dimensions
    if config['use_sr']:
        out_h = h_orig * config['sr_scale']
        out_w = w_orig * config['sr_scale']
    else:
        out_h, out_w = h_orig, w_orig

    # Warmup
    enhancer.warmup(video_height=h_orig, video_width=w_orig, batch_sizes=[1])

    # Process frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    all_timings = defaultdict(list)
    total_times = []
    n_faces = 0
    processed = 0
    timed = 0

    while timed < n_frames:
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

        processed += 1
        if processed > warmup_frames:
            total_times.append(elapsed_ms)
            for k, v in timing.items():
                all_timings[k].append(v)
            n_faces += len(enhanced_faces)
            timed += 1

            if timed % 50 == 0:
                med = np.median(total_times[-50:])
                fps = 1000.0 / med if med > 0 else 0
                print(f"  Frame {timed}: last-50 median={med:.1f}ms ({fps:.1f} FPS)")

    cap.release()

    if not total_times:
        return None

    avg_ms = np.mean(total_times)
    med_ms = np.median(total_times)
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    # Print component breakdown
    components = ['sr_ms', 'detect_ms', 'align_ms', 'facegan_ms',
                  'parse_blend_ms', 'parse_ms', 'composite_ms']
    comp_names = ['SR background', 'Face detection', 'Face alignment',
                  'FaceGAN', 'Parse+blend', 'ParseNet', 'Composite']

    print(f"\n  {'─'*50}")
    total_comp_time = 0
    comp_medians = {}
    for comp, cname in zip(components, comp_names):
        vals = all_timings.get(comp, [])
        if vals:
            m = np.median(vals)
            comp_medians[cname] = m
            total_comp_time += m
            print(f"  {cname:20s}: {m:7.1f}ms  (mean={np.mean(vals):.1f}ms)")

    if total_comp_time > 0:
        print(f"\n  Component shares:")
        for cname, m in sorted(comp_medians.items(), key=lambda x: -x[1]):
            pct = 100.0 * m / total_comp_time
            bar = "█" * int(pct / 2)
            print(f"    {cname:20s}: {pct:5.1f}% {bar}")

    print(f"\n  TOTAL: median={med_ms:.1f}ms, mean={avg_ms:.1f}ms")
    fps = 1000.0 / med_ms if med_ms > 0 else 0
    print(f"  FPS: {fps:.1f} (median), {1000/avg_ms:.1f} (mean)")
    print(f"  Output: {out_w}x{out_h}")
    print(f"  Faces: {n_faces} total, {n_faces/max(1,timed):.2f}/frame")
    print(f"  GPU peak: {peak_mem:.0f} MB")

    # Cleanup
    del enhancer
    torch.cuda.empty_cache()

    return {
        'name': name,
        'avg_ms': avg_ms,
        'med_ms': med_ms,
        'fps_med': fps,
        'fps_avg': 1000/avg_ms if avg_ms > 0 else 0,
        'peak_mem_mb': peak_mem,
        'output_res': f"{out_w}x{out_h}",
        'faces': n_faces,
        'comp_medians': comp_medians,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='../test_1min.mp4')
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--warmup-frames', type=int, default=10)
    args = parser.parse_args()

    # Define all configs to test
    configs = [
        {
            'name': '512 + SR×2 (current)',
            'model': 'GPEN-BFR-512',
            'in_size': 512,
            'channel_multiplier': 2,
            'narrow': 1,
            'use_sr': True,
            'sr_scale': 2,
        },
        {
            'name': '512 + SR×1',
            'model': 'GPEN-BFR-512',
            'in_size': 512,
            'channel_multiplier': 2,
            'narrow': 1,
            'use_sr': True,
            'sr_scale': 1,
        },
        {
            'name': '512 no SR',
            'model': 'GPEN-BFR-512',
            'in_size': 512,
            'channel_multiplier': 2,
            'narrow': 1,
            'use_sr': False,
            'sr_scale': 1,
        },
        {
            'name': '256 + SR×2',
            'model': 'GPEN-BFR-256',
            'in_size': 256,
            'channel_multiplier': 1,
            'narrow': 0.5,
            'use_sr': True,
            'sr_scale': 2,
        },
        {
            'name': '256 + SR×1',
            'model': 'GPEN-BFR-256',
            'in_size': 256,
            'channel_multiplier': 1,
            'narrow': 0.5,
            'use_sr': True,
            'sr_scale': 1,
        },
        {
            'name': '256 no SR',
            'model': 'GPEN-BFR-256',
            'in_size': 256,
            'channel_multiplier': 1,
            'narrow': 0.5,
            'use_sr': False,
            'sr_scale': 1,
        },
    ]

    print("="*70)
    print("GPEN Configuration Comparison Benchmark")
    print("="*70)
    print(f"  Video: {args.video}")
    print(f"  Frames per config: {args.frames}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA: {torch.version.cuda}")

    cap = cv2.VideoCapture(args.video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"  Input: {w}x{h} @ {fps}fps")

    results = []
    for config in configs:
        r = run_config(config, args.video, args.frames, args.warmup_frames)
        if r:
            results.append(r)

    # Summary table
    if results:
        print(f"\n\n{'='*90}")
        print(">>> CONFIGURATION COMPARISON")
        print(f"{'='*90}")
        print(f"  {'Config':<22s} {'Med ms':>8s} {'FPS':>6s} {'Output':>12s} "
              f"{'SR ms':>8s} {'Comp ms':>8s} {'VRAM':>8s}")
        print(f"  {'─'*78}")

        best_fps = max(r['fps_med'] for r in results)
        for r in results:
            sr_ms = r['comp_medians'].get('SR background', 0)
            comp_ms = (r['comp_medians'].get('Composite', 0) +
                       r['comp_medians'].get('Parse+blend', 0))
            marker = " ★" if r['fps_med'] == best_fps else ""
            print(f"  {r['name']:<22s} {r['med_ms']:>7.1f}ms {r['fps_med']:>5.1f} "
                  f"{r['output_res']:>12s} {sr_ms:>7.1f}ms {comp_ms:>7.1f}ms "
                  f"{r['peak_mem_mb']:>7.0f}MB{marker}")

        print(f"\n  ★ = fastest config")
        print(f"{'='*90}")


if __name__ == '__main__':
    main()
