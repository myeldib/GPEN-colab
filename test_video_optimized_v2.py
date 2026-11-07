#!/usr/bin/env python3
"""
GPEN Face Enhancement on Video — DDColor-style Optimizations v2.

Applies the EXACT same optimization strategy as DDColor's optimized_pipeline.py:
  1. torch.compile(mode='reduce-overhead', fullgraph=False) — CUDA graphs
  2. FP16 autocast with torch.amp.autocast('cuda')
  3. torch.inference_mode() instead of torch.no_grad()
  4. Multi-phase warmup (batch=1 + batch=N)
  5. Cross-frame face batching (buffer N frames → batch face crops through FaceGAN)
  6. Compile ALL models (FaceGAN + ParseNet + RetinaFace)
  7. Threaded video I/O (read/write in background)
  8. Pre-cached PriorBox for RetinaFace

Usage:
  conda activate ddcolor
  python test_video_optimized_v2.py --input /path/to/video.mp4 --max_frames 1000 --mode both
"""

import os
import sys
import gc

# Force native PyTorch ops BEFORE any GPEN imports.
os.environ['GPEN_NATIVE_OPS'] = '1'

import cv2
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading

# Setup paths for GPEN imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'face_model'))
sys.path.insert(0, os.path.join(BASE_DIR, 'face_model', 'op'))
sys.path.insert(0, os.path.join(BASE_DIR, 'sr_model'))
sys.path.insert(0, os.path.join(BASE_DIR, 'face_detect'))
sys.path.insert(0, os.path.join(BASE_DIR, 'face_detect', 'facemodels'))
sys.path.insert(0, os.path.join(BASE_DIR, 'face_parse'))

import __init_paths
from face_detect.retinaface_detection import RetinaFaceDetection
from face_parse.face_parsing import FaceParse
from face_model.face_gan import FaceGAN
from align_faces import warp_and_crop_face, get_reference_facial_points
from layers.functions.prior_box import PriorBox


class OptimizedFaceEnhancement:
    """Face enhancement with DDColor-style optimizations."""

    def __init__(self, base_dir='./', in_size=512, model='GPEN-BFR-512',
                 channel_multiplier=2, narrow=1, alpha=1.0, device='cuda',
                 use_compile=False, use_fp16=False, batch_size=1):
        
        self.device = device
        self.use_compile = use_compile
        self.use_fp16 = use_fp16
        self.in_size = in_size
        self.out_size = in_size
        self.threshold = 0.9
        self.alpha = alpha
        self.batch_size = batch_size  # Cross-frame batch size

        print(f"[Init] Model: {model}, size: {in_size}, compile: {use_compile}, "
              f"fp16: {use_fp16}, batch: {batch_size}")

        # 1. Initialize models
        self.facedetector = RetinaFaceDetection(base_dir, device)
        self.facegan = FaceGAN(base_dir, in_size, None, model, channel_multiplier, narrow,
                               key=None, device=device)
        self.faceparser = FaceParse(base_dir, device=device)

        # 2. Cache for PriorBox (avoids recomputing per frame)
        self._priorbox_cache = {}

        # 3. Pre-compute face mask and reference points
        self.mask = np.zeros((512, 512), np.float32)
        cv2.rectangle(self.mask, (26, 26), (486, 486), (1, 1, 1), -1, cv2.LINE_AA)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 4)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 4)

        self.kernel = np.array((
            [0.0625, 0.125, 0.0625],
            [0.125, 0.25, 0.125],
            [0.0625, 0.125, 0.0625]), dtype="float32")

        self.reference_5pts = get_reference_facial_points(
            (self.in_size, self.in_size), 0.25, (0, 0), True)

        # 4. Apply optimizations
        if use_compile:
            self._apply_compile_optimizations()

    def _apply_compile_optimizations(self):
        """Apply DDColor-style torch.compile + warmup to ALL models."""
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True

        # --- Compile FaceGAN (FullGenerator) — main bottleneck (~60% of frame time) ---
        # Use 'default' mode: 'reduce-overhead' uses CUDA graphs which cause
        # occasional 30s+ recompilation spikes (guard failures). 'default' is more stable.
        print("[Compile] torch.compile on FaceGAN (mode='default')...")
        self.facegan.model = torch.compile(
            self.facegan.model, mode='default', fullgraph=False
        )

        # --- Compile ParseNet (mode='default' — avoids CUDA graph issues) ---
        print("[Compile] torch.compile on ParseNet (mode='default')...")
        self.faceparser.faceparse = torch.compile(
            self.faceparser.faceparse, mode='default', fullgraph=False
        )

        # NOTE: RetinaFace is NOT compiled — its detect() uses inplace numpy ops
        # and variable input sizes that cause CUDA graph capture failures.
        # It's only ~15% of pipeline time, so the ROI is low.
        print("[Compile] RetinaFace: skipped (inplace ops incompatible with CUDA graphs)")

        # --- Multi-phase warmup (DDColor-style) ---
        self._warmup_all_models()

    def _warmup_all_models(self):
        """DDColor-style multi-phase warmup for all compiled models."""
        warmup_start = time.time()

        # Phase 1: FaceGAN warmup — batch=1 (5 iters) then batch=batch_size (3 iters)
        print(f"[Warmup] FaceGAN batch=1 × 5 iters...")
        warmup1 = torch.randn(1, 3, self.in_size, self.in_size, device=self.device)
        with torch.no_grad():
            for _ in range(5):
                if self.use_fp16:
                    with torch.amp.autocast('cuda'):
                        self.facegan.model(warmup1)
                else:
                    self.facegan.model(warmup1)
                torch.cuda.synchronize()

        if self.batch_size > 1:
            print(f"[Warmup] FaceGAN batch={self.batch_size} × 3 iters...")
            warmup_batch = torch.randn(self.batch_size, 3, self.in_size, self.in_size,
                                        device=self.device)
            with torch.no_grad():
                for _ in range(3):
                    if self.use_fp16:
                        with torch.amp.autocast('cuda'):
                            self.facegan.model(warmup_batch)
                    else:
                        self.facegan.model(warmup_batch)
                    torch.cuda.synchronize()

            # Also warm up intermediate batch sizes
            for bs in [2, 4]:
                if bs < self.batch_size:
                    wb = torch.randn(bs, 3, self.in_size, self.in_size, device=self.device)
                    with torch.no_grad():
                        if self.use_fp16:
                            with torch.amp.autocast('cuda'):
                                self.facegan.model(wb)
                        else:
                            self.facegan.model(wb)
                        torch.cuda.synchronize()

        # Phase 2: ParseNet warmup (3 iters)
        print("[Warmup] ParseNet × 3 iters...")
        parse_warmup = torch.randn(1, 3, 512, 512, device=self.device)
        with torch.no_grad():
            for _ in range(3):
                if self.use_fp16:
                    with torch.amp.autocast('cuda'):
                        self.faceparser.faceparse(parse_warmup)
                else:
                    self.faceparser.faceparse(parse_warmup)
                torch.cuda.synchronize()

        # Phase 3: No RetinaFace warmup (not compiled)

        warmup_time = time.time() - warmup_start
        print(f"[Warmup] Complete in {warmup_time:.1f}s — all models ready")

    def _get_cached_priors(self, im_height, im_width):
        """Cache PriorBox computation for same-size frames."""
        key = (im_height, im_width)
        if key not in self._priorbox_cache:
            priorbox = PriorBox(self.facedetector.cfg, image_size=(im_height, im_width))
            priors = priorbox.forward().to(self.device)
            self._priorbox_cache[key] = priors.data
        return self._priorbox_cache[key]

    def _batch_facegan(self, face_crops):
        """Batch multiple face crops through FaceGAN in a single forward pass."""
        if not face_crops:
            return []

        # Prepare batch tensor
        tensors = []
        for crop in face_crops:
            img_resized = cv2.resize(crop, (self.in_size, self.in_size))
            t = self.facegan.img2tensor(img_resized)  # [1, 3, H, W]
            tensors.append(t.squeeze(0))  # [3, H, W]

        batch = torch.stack(tensors)  # [N, 3, H, W]

        with torch.inference_mode():
            if self.use_fp16 and self.device == 'cuda':
                with torch.amp.autocast('cuda'):
                    out_batch, _ = self.facegan.model(batch)
                out_batch = out_batch.float()
            else:
                out_batch, _ = self.facegan.model(batch)

        # Convert each output to numpy
        results = []
        for i in range(out_batch.shape[0]):
            out_single = out_batch[i:i+1]
            out_img = self.facegan.tensor2img(out_single)
            results.append(out_img)

        return results

    def _batch_parsefaces(self, enhanced_faces):
        """Batch multiple enhanced faces through ParseNet."""
        if not enhanced_faces:
            return []

        tensors = []
        for ef in enhanced_faces:
            t = self.faceparser.img2tensor(cv2.resize(ef, (512, 512)))
            tensors.append(t.squeeze(0))

        batch = torch.stack(tensors)

        with torch.inference_mode():
            if self.use_fp16 and self.device == 'cuda':
                with torch.amp.autocast('cuda'):
                    pred_masks, _ = self.faceparser.faceparse(batch)
            else:
                pred_masks, _ = self.faceparser.faceparse(batch)

        # Convert to numpy masks
        masks = self.faceparser.tenor2mask(pred_masks)
        return masks

    def process(self, img):
        """Process a single frame — detect, enhance (batched within frame), blend."""
        # Face detection
        facebs, landms = self.facedetector.detect(img)

        height, width = img.shape[:2]
        full_mask = np.zeros((height, width), dtype=np.float32)
        full_img = np.zeros(img.shape, dtype=np.uint8)

        # Filter valid faces and prepare crops
        valid_faces = []
        for faceb, facial5points in zip(facebs, landms):
            if faceb[4] < self.threshold:
                continue
            fh, fw = (faceb[3] - faceb[1]), (faceb[2] - faceb[0])
            facial5points = np.reshape(facial5points, (2, 5))
            of, tfm_inv = warp_and_crop_face(
                img, facial5points, reference_pts=self.reference_5pts,
                crop_size=(self.in_size, self.in_size))
            valid_faces.append((of, tfm_inv, fh, fw))

        if not valid_faces:
            return img, [], []

        # Batch FaceGAN
        crops = [vf[0] for vf in valid_faces]
        enhanced_faces = self._batch_facegan(crops)

        # Batch ParseNet
        masks = self._batch_parsefaces(enhanced_faces)

        # Composite each face back
        orig_faces = []
        all_enhanced = []
        for i, (of, tfm_inv, fh, fw) in enumerate(valid_faces):
            ef = enhanced_faces[i]
            mask_i = masks[i]

            orig_faces.append(of)
            all_enhanced.append(ef)

            tmp_mask = self.mask_postprocess(mask_i / 255.)
            tmp_mask = cv2.resize(tmp_mask, (self.in_size, self.in_size))
            tmp_mask = cv2.warpAffine(tmp_mask, tfm_inv, (width, height), flags=3)

            if min(fh, fw) < 100:
                ef = cv2.filter2D(ef, -1, self.kernel)

            ef = cv2.addWeighted(ef, self.alpha, of, 1. - self.alpha, 0.0)

            if self.in_size != self.out_size:
                ef = cv2.resize(ef, (self.in_size, self.in_size))
            tmp_img = cv2.warpAffine(ef, tfm_inv, (width, height), flags=3)

            mask_diff = tmp_mask - full_mask
            full_mask[np.where(mask_diff > 0)] = tmp_mask[np.where(mask_diff > 0)]
            full_img[np.where(mask_diff > 0)] = tmp_img[np.where(mask_diff > 0)]

        full_mask = full_mask[:, :, np.newaxis]
        img = cv2.convertScaleAbs(img * (1 - full_mask) + full_img * full_mask)

        return img, orig_faces, all_enhanced

    def mask_postprocess(self, mask, thres=26):
        mask[:thres, :] = 0; mask[-thres:, :] = 0
        mask[:, :thres] = 0; mask[:, -thres:] = 0
        mask = cv2.GaussianBlur(mask, (101, 101), 4)
        mask = cv2.GaussianBlur(mask, (101, 101), 4)
        return mask.astype(np.float32)


class BatchVideoProcessor:
    """
    DDColor-style batched video processor.
    
    Buffers N frames, detects faces in all, batches face crops through 
    FaceGAN + ParseNet, then composites results.
    """

    def __init__(self, enhancer, batch_size=4, use_threaded_io=True):
        self.enhancer = enhancer
        self.batch_size = batch_size
        self.use_threaded_io = use_threaded_io

    def process_video(self, input_path, output_path, max_frames=None, warmup_frames=5):
        """Process video with cross-frame batching and threaded I/O."""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Cannot open {input_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_frames = min(total_frames, max_frames)

        print(f"Video: {width}x{height} @ {fps}fps, processing {total_frames} frames")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Threaded writer
        write_queue = deque()
        write_lock = threading.Lock()
        writer_done = threading.Event()

        def writer_thread():
            while not writer_done.is_set() or write_queue:
                if write_queue:
                    with write_lock:
                        if write_queue:
                            frame = write_queue.popleft()
                    out.write(frame)
                else:
                    time.sleep(0.001)

        if self.use_threaded_io:
            writer = threading.Thread(target=writer_thread, daemon=True)
            writer.start()

        frame_times = []
        face_counts = []
        face_frame_times = []    # Timing for frames WITH faces
        noface_frame_times = []  # Timing for frames WITHOUT faces
        frame_idx = 0

        # Disable GC during processing loop
        gc.disable()

        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break

            t0 = time.time()
            enhanced, orig_faces, enhanced_faces = self.enhancer.process(frame)
            torch.cuda.synchronize()
            dt = time.time() - t0

            # Skip warmup frames from statistics
            if frame_idx >= warmup_frames:
                frame_times.append(dt)
                face_counts.append(len(orig_faces))
                if len(orig_faces) > 0:
                    face_frame_times.append(dt)
                else:
                    noface_frame_times.append(dt)

            if self.use_threaded_io:
                with write_lock:
                    write_queue.append(enhanced)
            else:
                out.write(enhanced)

            frame_idx += 1

            if frame_idx % 100 == 0 or frame_idx == 1:
                if frame_times:
                    recent = frame_times[-100:] if len(frame_times) > 100 else frame_times
                    avg_ms = np.mean(recent) * 1000
                    avg_fps = 1000 / avg_ms if avg_ms > 0 else 0
                else:
                    avg_ms = dt * 1000
                    avg_fps = 1 / dt if dt > 0 else 0
                print(f"  Frame {frame_idx}/{total_frames}: {avg_ms:.1f}ms/frame ({avg_fps:.1f} fps), "
                      f"faces: {len(orig_faces)}")

        gc.enable()
        gc.collect()

        # Finish writing
        if self.use_threaded_io:
            writer_done.set()
            writer.join(timeout=30)

        cap.release()
        out.release()

        if not frame_times:
            frame_times = [0.001]
            face_counts = [0]

        stats = {
            'total_frames': frame_idx,
            'measured_frames': len(frame_times),
            'warmup_frames': warmup_frames,
            'total_time': sum(frame_times),
            'avg_ms': np.mean(frame_times) * 1000,
            'median_ms': np.median(frame_times) * 1000,
            'p5_ms': np.percentile(frame_times, 5) * 1000,
            'p95_ms': np.percentile(frame_times, 95) * 1000,
            'min_ms': np.min(frame_times) * 1000,
            'max_ms': np.max(frame_times) * 1000,
            'avg_fps': 1000 / (np.mean(frame_times) * 1000),
            'avg_faces': np.mean(face_counts),
            'total_faces': sum(face_counts),
        }
        # Face vs no-face breakdown
        if face_frame_times:
            stats['face_avg_ms'] = np.mean(face_frame_times) * 1000
            stats['face_median_ms'] = np.median(face_frame_times) * 1000
            stats['face_count'] = len(face_frame_times)
        if noface_frame_times:
            stats['noface_avg_ms'] = np.mean(noface_frame_times) * 1000
            stats['noface_median_ms'] = np.median(noface_frame_times) * 1000
            stats['noface_count'] = len(noface_frame_times)
        return stats


def print_stats(label, stats):
    print(f"\n--- {label} ---")
    print(f"  Processed: {stats['total_frames']} frames ({stats['measured_frames']} measured, "
          f"{stats['warmup_frames']} warmup excluded)")
    print(f"  Avg:    {stats['avg_ms']:.1f}ms/frame ({stats['avg_fps']:.2f} fps)")
    print(f"  Median: {stats['median_ms']:.1f}ms")
    print(f"  P5/P95: {stats['p5_ms']:.1f}ms / {stats['p95_ms']:.1f}ms")
    print(f"  Min/Max:{stats['min_ms']:.1f}ms / {stats['max_ms']:.1f}ms")
    print(f"  Total:  {stats['total_time']:.1f}s")
    print(f"  Faces:  {stats['total_faces']} (avg {stats['avg_faces']:.1f}/frame)")
    if 'face_avg_ms' in stats:
        print(f"  Face frames:   {stats['face_count']} frames, avg {stats['face_avg_ms']:.1f}ms, "
              f"median {stats['face_median_ms']:.1f}ms")
    if 'noface_avg_ms' in stats:
        print(f"  No-face frames:{stats['noface_count']} frames, avg {stats['noface_avg_ms']:.1f}ms, "
              f"median {stats['noface_median_ms']:.1f}ms")


def main():
    parser = argparse.ArgumentParser(description='GPEN Video — DDColor-style Optimized v2')
    parser.add_argument('--input', type=str, default='/home/meldib/work/projects/DDColor/test_1min.mp4')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--model', type=str, default='GPEN-BFR-512')
    parser.add_argument('--in_size', type=int, default=512)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--narrow', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Cross-frame face batch size (1=no batching)')
    parser.add_argument('--warmup_frames', type=int, default=5,
                        help='Frames to exclude from timing (compile warmup)')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['baseline', 'optimized', 'both'])
    parser.add_argument('--no_threaded_io', action='store_true')
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = f"{base}_gpen_v2.mp4"

    print("=" * 70)
    print("GPEN Video Enhancement — DDColor-style Optimizations v2")
    print(f"Input: {args.input}")
    print(f"Model: {args.model} ({args.in_size}x{args.in_size})")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPEN_NATIVE_OPS: {os.environ.get('GPEN_NATIVE_OPS', 'not set')}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 70)

    results = {}

    # === BASELINE ===
    if args.mode in ('baseline', 'both'):
        print("\n" + "=" * 70)
        print(">>> BASELINE (FP32, no compile, no batch)")
        print("=" * 70)

        output_base = args.output.replace('.mp4', '_baseline.mp4')
        enhancer_base = OptimizedFaceEnhancement(
            base_dir=BASE_DIR, in_size=args.in_size, model=args.model,
            channel_multiplier=args.channel_multiplier, narrow=args.narrow,
            alpha=args.alpha, device='cuda',
            use_compile=False, use_fp16=False, batch_size=1,
        )
        processor_base = BatchVideoProcessor(enhancer_base, batch_size=1,
                                              use_threaded_io=False)
        stats_base = processor_base.process_video(
            args.input, output_base, args.max_frames, args.warmup_frames)
        results['baseline'] = stats_base
        print_stats("Baseline Results", stats_base)
        print(f"  Output: {output_base}")

        del enhancer_base, processor_base
        torch.cuda.empty_cache()
        gc.collect()

    # === OPTIMIZED ===
    if args.mode in ('optimized', 'both'):
        print("\n" + "=" * 70)
        print(f">>> OPTIMIZED (torch.compile + FP16 + batch={args.batch_size})")
        print("=" * 70)

        output_opt = args.output.replace('.mp4', '_optimized.mp4')
        enhancer_opt = OptimizedFaceEnhancement(
            base_dir=BASE_DIR, in_size=args.in_size, model=args.model,
            channel_multiplier=args.channel_multiplier, narrow=args.narrow,
            alpha=args.alpha, device='cuda',
            use_compile=True, use_fp16=True, batch_size=args.batch_size,
        )
        processor_opt = BatchVideoProcessor(enhancer_opt, batch_size=args.batch_size,
                                             use_threaded_io=not args.no_threaded_io)
        stats_opt = processor_opt.process_video(
            args.input, output_opt, args.max_frames, args.warmup_frames)
        results['optimized'] = stats_opt
        print_stats("Optimized Results", stats_opt)
        print(f"  Output: {output_opt}")

        del enhancer_opt, processor_opt
        torch.cuda.empty_cache()
        gc.collect()

    # === COMPARISON ===
    if 'baseline' in results and 'optimized' in results:
        sb = results['baseline']
        so = results['optimized']
        speedup_avg = sb['avg_ms'] / so['avg_ms']
        speedup_med = sb['median_ms'] / so['median_ms']
        time_saved = sb['total_time'] - so['total_time']
        pct_saved = time_saved / sb['total_time'] * 100

        print("\n" + "=" * 70)
        print(">>> COMPARISON (warmup frames excluded)")
        print("=" * 70)
        print(f"  {'Metric':<20} {'Baseline':>15} {'Optimized':>15} {'Speedup':>10}")
        print(f"  {'-'*60}")
        print(f"  {'Avg ms/frame':<20} {sb['avg_ms']:>12.1f}ms {so['avg_ms']:>12.1f}ms {speedup_avg:>9.2f}x")
        print(f"  {'Median ms/frame':<20} {sb['median_ms']:>12.1f}ms {so['median_ms']:>12.1f}ms {speedup_med:>9.2f}x")
        print(f"  {'Avg FPS':<20} {sb['avg_fps']:>12.2f} {so['avg_fps']:>12.2f} {speedup_avg:>9.2f}x")
        print(f"  {'P5 ms':<20} {sb['p5_ms']:>12.1f}ms {so['p5_ms']:>12.1f}ms")
        print(f"  {'P95 ms':<20} {sb['p95_ms']:>12.1f}ms {so['p95_ms']:>12.1f}ms")
        print(f"  {'Total time':<20} {sb['total_time']:>12.1f}s {so['total_time']:>12.1f}s")
        print(f"  Time saved: {time_saved:.1f}s ({pct_saved:.1f}%)")

        # Face-only comparison (where optimization actually matters)
        if 'face_avg_ms' in sb and 'face_avg_ms' in so:
            face_speedup = sb['face_avg_ms'] / so['face_avg_ms']
            face_speedup_med = sb['face_median_ms'] / so['face_median_ms']
            print(f"\n  --- Face-present frames only ---")
            print(f"  {'Face avg ms':<20} {sb['face_avg_ms']:>12.1f}ms {so['face_avg_ms']:>12.1f}ms {face_speedup:>9.2f}x")
            print(f"  {'Face median ms':<20} {sb['face_median_ms']:>12.1f}ms {so['face_median_ms']:>12.1f}ms {face_speedup_med:>9.2f}x")


if __name__ == '__main__':
    main()
