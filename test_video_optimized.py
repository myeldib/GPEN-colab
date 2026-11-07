#!/usr/bin/env python3
"""
GPEN Face Enhancement on Video — torch.compile + FP16 optimized vs baseline.

Tests face enhancement on a video file, comparing:
  1. Baseline (FP32, no compile)
  2. torch.compile + FP16 (optimized)

Usage:
  conda activate ddcolor
  python test_video_optimized.py --input /path/to/video.mp4 --output /path/to/output.mp4
"""

import os
import sys
import cv2
import time
import argparse
import numpy as np
import torch

# Setup paths for GPEN imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'face_model'))
sys.path.insert(0, os.path.join(BASE_DIR, 'face_model', 'op'))
sys.path.insert(0, os.path.join(BASE_DIR, 'sr_model'))
sys.path.insert(0, os.path.join(BASE_DIR, 'face_detect'))
sys.path.insert(0, os.path.join(BASE_DIR, 'face_detect', 'facemodels'))
sys.path.insert(0, os.path.join(BASE_DIR, 'face_parse'))

# Force native PyTorch ops BEFORE any GPEN imports.
# The custom CUDA kernels (fused_bias_act, upfirdn2d) don't support FP16,
# and torch.compile can better optimize the native PyTorch equivalents.
os.environ['GPEN_NATIVE_OPS'] = '1'
print("[Init] GPEN_NATIVE_OPS=1 → using native PyTorch ops for FP16/compile compatibility")


import __init_paths
from face_detect.retinaface_detection import RetinaFaceDetection
from face_parse.face_parsing import FaceParse
from face_model.face_gan import FaceGAN
from sr_model.real_esrnet import RealESRNet
from align_faces import warp_and_crop_face, get_reference_facial_points


class OptimizedFaceEnhancement:
    """Face enhancement with torch.compile + FP16 support."""

    def __init__(self, base_dir='./', in_size=512, model='GPEN-BFR-512',
                 channel_multiplier=2, narrow=1, use_sr=False, sr_model='realesrnet',
                 sr_scale=2, tile_size=0, alpha=1.0, device='cuda',
                 use_compile=False, use_fp16=False):
        
        self.device = device
        self.use_compile = use_compile
        self.use_fp16 = use_fp16
        self.in_size = in_size
        self.out_size = in_size
        self.threshold = 0.9
        self.alpha = alpha
        self.use_sr = use_sr

        print(f"[Init] Model: {model}, size: {in_size}, compile: {use_compile}, fp16: {use_fp16}")

        # Initialize models
        self.facedetector = RetinaFaceDetection(base_dir, device)
        
        # Face GAN - main model to optimize
        self.facegan = FaceGAN(base_dir, in_size, None, model, channel_multiplier, narrow, 
                              key=None, device=device)
        
        # Face parser
        self.faceparser = FaceParse(base_dir, device=device)
        
        # SR model (optional)
        if use_sr:
            self.srmodel = RealESRNet(base_dir, sr_model, sr_scale, tile_size, device=device)
        else:
            self.srmodel = None

        # Apply torch.compile to the main models
        if use_compile:
            print("[Compile] Applying torch.compile to FaceGAN model...")
            torch.set_float32_matmul_precision('high')
            torch.backends.cudnn.benchmark = True
            
            self.facegan.model = torch.compile(
                self.facegan.model, mode='default', fullgraph=False
            )
            
            # Warmup the compiled model
            print("[Compile] Warming up compiled FaceGAN...")
            warmup_input = torch.randn(1, 3, in_size, in_size, device=device)
            for i in range(5):
                with torch.no_grad():
                    if use_fp16:
                        with torch.amp.autocast('cuda', dtype=torch.float16):
                            self.facegan.model(warmup_input)
                    else:
                        self.facegan.model(warmup_input)
                torch.cuda.synchronize()
            print("[Compile] Warmup complete")

            # Also compile SR model if used
            if use_sr and self.srmodel is not None:
                print("[Compile] Applying torch.compile to SR model...")
                self.srmodel.model = torch.compile(
                    self.srmodel.model, mode='default', fullgraph=False
                )
                sr_warmup = torch.randn(1, 3, 64, 64, device=device)
                for i in range(3):
                    with torch.no_grad():
                        if use_fp16:
                            with torch.amp.autocast('cuda', dtype=torch.float16):
                                self.srmodel.model(sr_warmup)
                        else:
                            self.srmodel.model(sr_warmup)
                    torch.cuda.synchronize()
                print("[Compile] SR warmup complete")

        # Face mask setup
        self.mask = np.zeros((512, 512), np.float32)
        cv2.rectangle(self.mask, (26, 26), (486, 486), (1, 1, 1), -1, cv2.LINE_AA)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 4)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 4)

        self.kernel = np.array((
            [0.0625, 0.125, 0.0625],
            [0.125, 0.25, 0.125],
            [0.0625, 0.125, 0.0625]), dtype="float32")

        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        self.reference_5pts = get_reference_facial_points(
            (self.in_size, self.in_size), inner_padding_factor, outer_padding, default_square)

    def mask_postprocess(self, mask, thres=26):
        mask[:thres, :] = 0; mask[-thres:, :] = 0
        mask[:, :thres] = 0; mask[:, -thres:] = 0
        mask = cv2.GaussianBlur(mask, (101, 101), 4)
        mask = cv2.GaussianBlur(mask, (101, 101), 4)
        return mask.astype(np.float32)

    def _process_face_gan(self, img):
        """Run FaceGAN with optional FP16."""
        img = cv2.resize(img, (self.in_size, self.in_size))
        img_t = self.facegan.img2tensor(img)
        
        with torch.no_grad():
            if self.use_fp16 and self.device == 'cuda':
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    out, _ = self.facegan.model(img_t)
                out = out.float()  # Cast back to FP32 for tensor2img
            else:
                out, _ = self.facegan.model(img_t)
        
        del img_t
        out = self.facegan.tensor2img(out)
        return out

    def process(self, img):
        """Process a single frame — detect faces, enhance, blend back."""
        orig_faces, enhanced_faces = [], []

        if self.use_sr and self.srmodel is not None:
            img_sr = self.srmodel.process(img)
            if img_sr is not None:
                img = cv2.resize(img, img_sr.shape[:2][::-1])
        else:
            img_sr = None

        facebs, landms = self.facedetector.detect(img)

        height, width = img.shape[:2]
        full_mask = np.zeros((height, width), dtype=np.float32)
        full_img = np.zeros(img.shape, dtype=np.uint8)

        for i, (faceb, facial5points) in enumerate(zip(facebs, landms)):
            if faceb[4] < self.threshold:
                continue
            fh, fw = (faceb[3] - faceb[1]), (faceb[2] - faceb[0])

            facial5points = np.reshape(facial5points, (2, 5))

            of, tfm_inv = warp_and_crop_face(
                img, facial5points, reference_pts=self.reference_5pts,
                crop_size=(self.in_size, self.in_size))

            # Enhanced face processing (optimized path)
            ef = self._process_face_gan(of)

            orig_faces.append(of)
            enhanced_faces.append(ef)

            tmp_mask = self.mask_postprocess(self.faceparser.process(ef)[0] / 255.)
            tmp_mask = cv2.resize(tmp_mask, (self.in_size, self.in_size))
            tmp_mask = cv2.warpAffine(tmp_mask, tfm_inv, (width, height), flags=3)

            if min(fh, fw) < 100:
                ef = cv2.filter2D(ef, -1, self.kernel)

            ef = cv2.addWeighted(ef, self.alpha, of, 1. - self.alpha, 0.0)

            if self.in_size != self.out_size:
                ef = cv2.resize(ef, (self.in_size, self.in_size))
            tmp_img = cv2.warpAffine(ef, tfm_inv, (width, height), flags=3)

            mask = tmp_mask - full_mask
            full_mask[np.where(mask > 0)] = tmp_mask[np.where(mask > 0)]
            full_img[np.where(mask > 0)] = tmp_img[np.where(mask > 0)]

        full_mask = full_mask[:, :, np.newaxis]
        if self.use_sr and img_sr is not None:
            img = cv2.convertScaleAbs(img_sr * (1 - full_mask) + full_img * full_mask)
        else:
            img = cv2.convertScaleAbs(img * (1 - full_mask) + full_img * full_mask)

        return img, orig_faces, enhanced_faces


def process_video(input_path, output_path, enhancer, max_frames=None):
    """Process video frame by frame with face enhancement."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_path}")
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

    frame_times = []
    face_counts = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frame_idx >= max_frames):
            break

        t0 = time.time()
        enhanced, orig_faces, enhanced_faces = enhancer.process(frame)
        torch.cuda.synchronize()
        dt = time.time() - t0

        frame_times.append(dt)
        face_counts.append(len(orig_faces))

        out.write(enhanced)
        frame_idx += 1

        if frame_idx % 25 == 0 or frame_idx == 1:
            avg_ms = np.mean(frame_times[-25:]) * 1000
            avg_fps = 1000 / avg_ms if avg_ms > 0 else 0
            print(f"  Frame {frame_idx}/{total_frames}: {avg_ms:.1f}ms/frame ({avg_fps:.1f} fps), "
                  f"faces: {face_counts[-1]}")

    cap.release()
    out.release()

    # Stats
    stats = {
        'total_frames': frame_idx,
        'total_time': sum(frame_times),
        'avg_ms': np.mean(frame_times) * 1000,
        'median_ms': np.median(frame_times) * 1000,
        'min_ms': np.min(frame_times) * 1000,
        'max_ms': np.max(frame_times) * 1000,
        'avg_fps': 1000 / (np.mean(frame_times) * 1000),
        'avg_faces': np.mean(face_counts),
        'total_faces': sum(face_counts),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description='GPEN Video Face Enhancement — Optimized')
    parser.add_argument('--input', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, default=None, help='Output video path')
    parser.add_argument('--model', type=str, default='GPEN-BFR-512', help='GPEN model name')
    parser.add_argument('--in_size', type=int, default=512, help='Input face size')
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--narrow', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--use_sr', action='store_true', help='Enable super-resolution')
    parser.add_argument('--sr_model', type=str, default='realesrnet')
    parser.add_argument('--sr_scale', type=int, default=2)
    parser.add_argument('--tile_size', type=int, default=0)
    parser.add_argument('--max_frames', type=int, default=None, help='Max frames to process')
    parser.add_argument('--mode', type=str, default='both', choices=['baseline', 'optimized', 'both'],
                        help='Run baseline, optimized, or both')
    parser.add_argument('--add_audio', action='store_true', help='Mux audio from input to output')
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = f"{base}_gpen_enhanced.mp4"

    print("=" * 70)
    print(f"GPEN Video Face Enhancement — torch.compile + FP16 Benchmark")
    print(f"Input: {args.input}")
    print(f"Model: {args.model} ({args.in_size}x{args.in_size})")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 70)

    results = {}

    # === BASELINE ===
    if args.mode in ('baseline', 'both'):
        print("\n" + "=" * 70)
        print(">>> BASELINE (FP32, no compile)")
        print("=" * 70)
        
        output_baseline = args.output.replace('.mp4', '_baseline.mp4')
        
        enhancer_base = OptimizedFaceEnhancement(
            base_dir=BASE_DIR, in_size=args.in_size, model=args.model,
            channel_multiplier=args.channel_multiplier, narrow=args.narrow,
            use_sr=args.use_sr, sr_model=args.sr_model, sr_scale=args.sr_scale,
            tile_size=args.tile_size, alpha=args.alpha, device='cuda',
            use_compile=False, use_fp16=False,
        )
        
        stats_base = process_video(args.input, output_baseline, enhancer_base, args.max_frames)
        results['baseline'] = stats_base
        
        del enhancer_base
        torch.cuda.empty_cache()
        
        print(f"\n--- Baseline Results ---")
        print(f"  Avg: {stats_base['avg_ms']:.1f}ms/frame ({stats_base['avg_fps']:.2f} fps)")
        print(f"  Median: {stats_base['median_ms']:.1f}ms")
        print(f"  Total: {stats_base['total_time']:.1f}s for {stats_base['total_frames']} frames")
        print(f"  Faces processed: {stats_base['total_faces']} (avg {stats_base['avg_faces']:.1f}/frame)")
        print(f"  Output: {output_baseline}")

    # === OPTIMIZED ===
    if args.mode in ('optimized', 'both'):
        print("\n" + "=" * 70)
        print(">>> OPTIMIZED (torch.compile + FP16)")
        print("=" * 70)
        
        output_opt = args.output.replace('.mp4', '_optimized.mp4')
        
        enhancer_opt = OptimizedFaceEnhancement(
            base_dir=BASE_DIR, in_size=args.in_size, model=args.model,
            channel_multiplier=args.channel_multiplier, narrow=args.narrow,
            use_sr=args.use_sr, sr_model=args.sr_model, sr_scale=args.sr_scale,
            tile_size=args.tile_size, alpha=args.alpha, device='cuda',
            use_compile=True, use_fp16=True,
        )
        
        stats_opt = process_video(args.input, output_opt, enhancer_opt, args.max_frames)
        results['optimized'] = stats_opt
        
        del enhancer_opt
        torch.cuda.empty_cache()
        
        print(f"\n--- Optimized Results ---")
        print(f"  Avg: {stats_opt['avg_ms']:.1f}ms/frame ({stats_opt['avg_fps']:.2f} fps)")
        print(f"  Median: {stats_opt['median_ms']:.1f}ms")
        print(f"  Total: {stats_opt['total_time']:.1f}s for {stats_opt['total_frames']} frames")
        print(f"  Faces processed: {stats_opt['total_faces']} (avg {stats_opt['avg_faces']:.1f}/frame)")
        print(f"  Output: {output_opt}")

    # === COMPARISON ===
    if 'baseline' in results and 'optimized' in results:
        speedup = results['baseline']['avg_ms'] / results['optimized']['avg_ms']
        time_saved = results['baseline']['total_time'] - results['optimized']['total_time']
        print("\n" + "=" * 70)
        print(">>> COMPARISON")
        print("=" * 70)
        print(f"  Baseline:  {results['baseline']['avg_ms']:.1f}ms/frame ({results['baseline']['avg_fps']:.2f} fps)")
        print(f"  Optimized: {results['optimized']['avg_ms']:.1f}ms/frame ({results['optimized']['avg_fps']:.2f} fps)")
        print(f"  Speedup:   {speedup:.2f}x")
        print(f"  Time saved: {time_saved:.1f}s ({time_saved/results['baseline']['total_time']*100:.1f}%)")

    # Mux audio if requested
    if args.add_audio:
        for key in results:
            if key == 'baseline':
                vid = args.output.replace('.mp4', '_baseline.mp4')
            else:
                vid = args.output.replace('.mp4', '_optimized.mp4')
            final = vid.replace('.mp4', '_audio.mp4')
            cmd = f'ffmpeg -y -i "{vid}" -i "{args.input}" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0? -shortest "{final}" 2>/dev/null'
            os.system(cmd)
            print(f"  Audio muxed: {final}")


if __name__ == '__main__':
    main()
