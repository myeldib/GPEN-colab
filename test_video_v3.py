#!/usr/bin/env python3
"""
GPEN Face Enhancement — Max GPU Utilization Benchmark
Tests torch.compile + FP16 + batched face processing + SR background enhancement.
Configs: batch 4, 8, 16 with all optimizations.
"""

import os
import sys
import cv2
import time
import argparse
import numpy as np
import torch

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'face_model'))
sys.path.insert(0, os.path.join(BASE_DIR, 'face_model', 'op'))
sys.path.insert(0, os.path.join(BASE_DIR, 'sr_model'))
sys.path.insert(0, os.path.join(BASE_DIR, 'face_detect'))
sys.path.insert(0, os.path.join(BASE_DIR, 'face_detect', 'facemodels'))
sys.path.insert(0, os.path.join(BASE_DIR, 'face_parse'))

# Force native PyTorch ops for FP16/compile compatibility
os.environ['GPEN_NATIVE_OPS'] = '1'

import __init_paths
from face_detect.retinaface_detection import RetinaFaceDetection
from face_parse.face_parsing import FaceParse
from face_model.face_gan import FaceGAN
from sr_model.real_esrnet import RealESRNet
from align_faces import warp_and_crop_face, get_reference_facial_points


class MaxOptimizedFaceEnhancement:
    """
    Face enhancement with ALL optimizations:
    - torch.compile on FaceGAN, SR, ParseNet
    - FP16 autocast
    - Batched face processing (process N faces in one GPU call)
    - SR background enhancement
    - cudnn.benchmark + TF32
    """

    def __init__(self, base_dir='./', in_size=512, model='GPEN-BFR-512',
                 channel_multiplier=2, narrow=1, use_sr=False, sr_model='realesrnet',
                 sr_scale=2, tile_size=0, alpha=1.0, device='cuda',
                 face_batch_size=4, video_height=720, video_width=960):
        
        self.device = device
        self.in_size = in_size
        self.out_size = in_size
        self.threshold = 0.9
        self.alpha = alpha
        self.use_sr = use_sr
        self.face_batch_size = face_batch_size
        self.video_height = video_height
        self.video_width = video_width

        # Global PyTorch optimizations
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        print(f"[Init] Model: {model}, size: {in_size}")
        print(f"[Init] face_batch_size: {face_batch_size}, SR: {use_sr} (scale={sr_scale})")
        print(f"[Init] torch.compile + FP16 + cudnn.benchmark + TF32")

        # ── Models ──────────────────────────────────────────────
        # Face Detector (NOT compiled — uses inplace ops, dynamic shapes)
        self.facedetector = RetinaFaceDetection(base_dir, device)
        
        # Face GAN — main model, biggest optimization target
        self.facegan = FaceGAN(base_dir, in_size, None, model, channel_multiplier, narrow,
                              key=None, device=device)
        
        # Face Parser
        self.faceparser = FaceParse(base_dir, device=device)

        # SR model (optional)
        if use_sr:
            self.srmodel = RealESRNet(base_dir, sr_model, sr_scale, tile_size, device=device)
        else:
            self.srmodel = None

        # ── Compile all models ──────────────────────────────────
        print("[Compile] Compiling FaceGAN...")
        self.facegan.model = torch.compile(
            self.facegan.model, mode='default', fullgraph=False
        )

        print("[Compile] Compiling ParseNet...")
        self.faceparser.faceparse = torch.compile(
            self.faceparser.faceparse, mode='default', fullgraph=False
        )

        if use_sr and self.srmodel is not None:
            print("[Compile] Compiling SR model...")
            self.srmodel.srmodel = torch.compile(
                self.srmodel.srmodel, mode='default', fullgraph=False
            )

        # ── Warmup ──────────────────────────────────────────────
        self._warmup()

        # ── Face mask setup ────────────────────────────────────
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

    def _warmup(self):
        """Warmup all compiled models with FP16 to trigger JIT compilation."""
        print("[Warmup] FaceGAN...")
        # Warmup at batch=1 and batch=face_batch_size
        for bs in [1, self.face_batch_size]:
            warmup = torch.randn(bs, 3, self.in_size, self.in_size, device=self.device)
            for _ in range(3):
                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        self.facegan.model(warmup)
                torch.cuda.synchronize()
            del warmup

        print("[Warmup] ParseNet...")
        parse_warmup = torch.randn(1, 3, 512, 512, device=self.device)
        for _ in range(3):
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    self.faceparser.faceparse(parse_warmup)
            torch.cuda.synchronize()
        del parse_warmup

        if self.use_sr and self.srmodel is not None:
            print(f"[Warmup] SR model at {self.video_height}x{self.video_width}...")
            # Warmup at actual video resolution to avoid recompilation on first frame
            sr_warmup = torch.randn(1, 3, self.video_height, self.video_width, device=self.device)
            for _ in range(3):
                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        self.srmodel.srmodel(sr_warmup)
                torch.cuda.synchronize()
            del sr_warmup

        torch.cuda.empty_cache()
        print("[Warmup] Complete!")

    def mask_postprocess(self, mask, thres=26):
        mask[:thres, :] = 0; mask[-thres:, :] = 0
        mask[:, :thres] = 0; mask[:, -thres:] = 0
        mask = cv2.GaussianBlur(mask, (101, 101), 4)
        mask = cv2.GaussianBlur(mask, (101, 101), 4)
        return mask.astype(np.float32)

    def _batch_process_faces(self, face_imgs):
        """
        Process multiple face crops in a SINGLE batched GPU call.
        face_imgs: list of numpy arrays (cropped faces, already resized to in_size)
        Returns: list of enhanced face numpy arrays
        """
        if not face_imgs:
            return []
        
        n = len(face_imgs)
        
        # Stack all faces into a batch tensor
        tensors = []
        for img in face_imgs:
            img_t = torch.from_numpy(img).to(self.device) / 255.
            if self.facegan.is_norm:
                img_t = (img_t - 0.5) / 0.5
            img_t = img_t.permute(2, 0, 1).flip(0)  # BGR->RGB, HWC->CHW
            tensors.append(img_t)
        
        batch = torch.stack(tensors, dim=0)  # (N, 3, H, W)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.float16):
                out_batch, _ = self.facegan.model(batch)
            out_batch = out_batch.float()
        
        del batch
        
        # Convert each output back to numpy
        results = []
        for i in range(n):
            out_t = out_batch[i]
            if self.facegan.is_norm:
                out_t = out_t * 0.5 + 0.5
            out_t = out_t.permute(1, 2, 0).flip(2)  # RGB->BGR, CHW->HWC
            out_np = np.clip(out_t.cpu().numpy(), 0, 1) * 255.0
            results.append(out_np.astype(np.uint8))
        
        del out_batch
        return results

    def _process_sr(self, img):
        """Run SR with FP16 autocast — only autocast around the forward pass."""
        img_f = img.astype(np.float32) / 255.
        img_t = torch.from_numpy(np.transpose(img_f[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_t = img_t.unsqueeze(0).to(self.device)

        scale = self.srmodel.scale
        mod_scale = 2 if scale == 2 else (4 if scale == 1 else None)

        if mod_scale is not None:
            _, _, h, w = img_t.size()
            h_pad = (mod_scale - h % mod_scale) % mod_scale
            w_pad = (mod_scale - w % mod_scale) % mod_scale
            if h_pad or w_pad:
                img_t = torch.nn.functional.pad(img_t, (0, w_pad, 0, h_pad), 'reflect')

        try:
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    output = self.srmodel.srmodel(img_t)
            del img_t

            output = output.float()
            if mod_scale is not None:
                _, _, h, w = output.size()
                output = output[:, :, 0:h - h_pad, 0:w - w_pad]
            output = output.data.squeeze().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            return output
        except Exception as e:
            print(f'SR failed: {e}')
            return None

    def process(self, img, timing=None):
        """Process a single frame — detect faces, batch enhance, blend back.
        
        Args:
            timing: if dict, will be filled with per-component times (ms)
        """
        orig_faces, enhanced_faces = [], []
        if timing is not None:
            timing.clear()

        # SR background enhancement
        t = time.time()
        if self.use_sr and self.srmodel is not None:
            img_sr = self._process_sr(img)
            if img_sr is not None:
                img = cv2.resize(img, img_sr.shape[:2][::-1])
        else:
            img_sr = None
        torch.cuda.synchronize()
        if timing is not None:
            timing['sr_ms'] = (time.time() - t) * 1000

        # Face detection
        t = time.time()
        facebs, landms = self.facedetector.detect(img)
        torch.cuda.synchronize()
        if timing is not None:
            timing['detect_ms'] = (time.time() - t) * 1000

        height, width = img.shape[:2]
        full_mask = np.zeros((height, width), dtype=np.float32)
        full_img = np.zeros(img.shape, dtype=np.uint8)

        # Collect all valid faces for batched processing
        t = time.time()
        valid_faces = []
        for i, (faceb, facial5points) in enumerate(zip(facebs, landms)):
            if faceb[4] < self.threshold:
                continue
            fh, fw = (faceb[3] - faceb[1]), (faceb[2] - faceb[0])
            facial5points = np.reshape(facial5points, (2, 5))
            of, tfm_inv = warp_and_crop_face(
                img, facial5points, reference_pts=self.reference_5pts,
                crop_size=(self.in_size, self.in_size))
            valid_faces.append({
                'orig': of,
                'tfm_inv': tfm_inv,
                'fh': fh, 'fw': fw,
                'resized': cv2.resize(of, (self.in_size, self.in_size))
            })
        if timing is not None:
            timing['align_ms'] = (time.time() - t) * 1000

        if valid_faces:
            # Process faces in batches
            t = time.time()
            all_enhanced = []
            for batch_start in range(0, len(valid_faces), self.face_batch_size):
                batch = valid_faces[batch_start:batch_start + self.face_batch_size]
                batch_imgs = [f['resized'] for f in batch]
                batch_results = self._batch_process_faces(batch_imgs)
                all_enhanced.extend(batch_results)
            torch.cuda.synchronize()
            if timing is not None:
                timing['facegan_ms'] = (time.time() - t) * 1000

            # Blend enhanced faces back
            t = time.time()
            for face_info, ef in zip(valid_faces, all_enhanced):
                of = face_info['orig']
                tfm_inv = face_info['tfm_inv']
                fh, fw = face_info['fh'], face_info['fw']

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
            torch.cuda.synchronize()
            if timing is not None:
                timing['parse_blend_ms'] = (time.time() - t) * 1000
        else:
            if timing is not None:
                timing['facegan_ms'] = 0
                timing['parse_blend_ms'] = 0

        full_mask = full_mask[:, :, np.newaxis]
        t = time.time()
        if self.use_sr and img_sr is not None:
            img = cv2.convertScaleAbs(img_sr * (1 - full_mask) + full_img * full_mask)
        else:
            img = cv2.convertScaleAbs(img * (1 - full_mask) + full_img * full_mask)
        if timing is not None:
            timing['composite_ms'] = (time.time() - t) * 1000

        return img, orig_faces, enhanced_faces


def process_video(input_path, output_path, enhancer, max_frames=None):
    """Process video with detailed timing breakdown."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # If SR is used with scale=2, output resolution doubles
    out_w, out_h = width, height
    if enhancer.use_sr and enhancer.srmodel is not None:
        out_w = width * enhancer.srmodel.scale
        out_h = height * enhancer.srmodel.scale

    if max_frames:
        total_frames = min(total_frames, max_frames)

    print(f"Video: {width}x{height} → {out_w}x{out_h} @ {fps}fps, {total_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    frame_times = []
    face_counts = []
    component_times = []  # list of timing dicts
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frame_idx >= max_frames):
            break

        timing = {}
        t0 = time.time()
        enhanced, orig_faces, enhanced_faces = enhancer.process(frame, timing=timing)
        torch.cuda.synchronize()
        dt = time.time() - t0

        frame_times.append(dt)
        face_counts.append(len(orig_faces))
        component_times.append(timing)

        out.write(enhanced)
        frame_idx += 1

        if frame_idx % 25 == 0 or frame_idx == 1:
            recent = frame_times[-25:]
            avg_ms = np.mean(recent) * 1000
            avg_fps = 1000 / avg_ms if avg_ms > 0 else 0
            face_frames = [i for i, c in enumerate(face_counts[-25:]) if c > 0]
            faces_pct = len(face_frames) / len(recent) * 100
            # Show component breakdown for latest frame
            t = timing
            parts = f"SR:{t.get('sr_ms',0):.0f} Det:{t.get('detect_ms',0):.0f} " \
                    f"GAN:{t.get('facegan_ms',0):.0f} Parse+Blend:{t.get('parse_blend_ms',0):.0f} " \
                    f"Comp:{t.get('composite_ms',0):.0f}"
            print(f"  Frame {frame_idx}/{total_frames}: {avg_ms:.1f}ms ({avg_fps:.1f}fps) | "
                  f"faces: {face_counts[-1]} | {parts}", flush=True)

    cap.release()
    out.release()

    # Detailed stats
    ft = np.array(frame_times)
    fc = np.array(face_counts)
    face_frame_mask = fc > 0
    
    # Component timing aggregation (skip first frame — compilation overhead)
    skip = min(1, len(component_times) - 1)
    ct_subset = component_times[skip:]
    component_keys = ['sr_ms', 'detect_ms', 'align_ms', 'facegan_ms', 'parse_blend_ms', 'composite_ms']
    component_avgs = {}
    for k in component_keys:
        vals = [t.get(k, 0) for t in ct_subset]
        if vals:
            component_avgs[k] = {
                'avg': np.mean(vals),
                'median': np.median(vals),
                'p95': np.percentile(vals, 95),
            }

    stats = {
        'total_frames': frame_idx,
        'total_time': ft.sum(),
        'avg_ms': ft.mean() * 1000,
        'median_ms': np.median(ft) * 1000,
        'p5_ms': np.percentile(ft, 5) * 1000,
        'p95_ms': np.percentile(ft, 95) * 1000,
        'min_ms': ft.min() * 1000,
        'max_ms': ft.max() * 1000,
        'avg_fps': 1000 / (ft.mean() * 1000),
        'avg_faces': fc.mean(),
        'total_faces': fc.sum(),
        'pct_face_frames': face_frame_mask.mean() * 100,
        'component_avgs': component_avgs,
    }
    
    if face_frame_mask.any():
        stats['avg_ms_face_frames'] = ft[face_frame_mask].mean() * 1000
        stats['avg_ms_noface_frames'] = ft[~face_frame_mask].mean() * 1000 if (~face_frame_mask).any() else 0
    
    return stats


def print_stats(name, stats):
    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")
    print(f"  Total: {stats['total_time']:.1f}s for {stats['total_frames']} frames")
    print(f"  Avg:    {stats['avg_ms']:.1f}ms/frame ({stats['avg_fps']:.2f} fps)")
    print(f"  Median: {stats['median_ms']:.1f}ms | P5: {stats['p5_ms']:.1f}ms | P95: {stats['p95_ms']:.1f}ms")
    print(f"  Min: {stats['min_ms']:.1f}ms | Max: {stats['max_ms']:.1f}ms")
    print(f"  Faces: {stats['total_faces']} total ({stats['avg_faces']:.1f}/frame, {stats['pct_face_frames']:.0f}% frames)")
    if 'avg_ms_face_frames' in stats:
        print(f"  Face frames: {stats['avg_ms_face_frames']:.1f}ms avg | No-face: {stats['avg_ms_noface_frames']:.1f}ms avg")
    
    # Component breakdown
    if 'component_avgs' in stats and stats['component_avgs']:
        print(f"\n  Component breakdown (median, excl. 1st frame):")
        ca = stats['component_avgs']
        total_median = sum(v['median'] for v in ca.values())
        labels = {
            'sr_ms': 'SR (background)',
            'detect_ms': 'Face detection',
            'align_ms': 'Face alignment',
            'facegan_ms': 'Face GAN',
            'parse_blend_ms': 'Parse + blend',
            'composite_ms': 'Composite',
        }
        for k, label in labels.items():
            if k in ca:
                pct = ca[k]['median'] / total_median * 100 if total_median > 0 else 0
                print(f"    {label:<20s}: {ca[k]['median']:>7.1f}ms ({pct:>5.1f}%) | P95: {ca[k]['p95']:.1f}ms")
        print(f"    {'TOTAL':<20s}: {total_median:>7.1f}ms")


def main():
    parser = argparse.ArgumentParser(description='GPEN Max GPU Optimization Benchmark')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--model', type=str, default='GPEN-BFR-512')
    parser.add_argument('--in_size', type=int, default=512)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--narrow', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--use_sr', action='store_true', help='Enable SR background enhancement')
    parser.add_argument('--sr_model', type=str, default='realesrnet')
    parser.add_argument('--sr_scale', type=int, default=2)
    parser.add_argument('--tile_size', type=int, default=0)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--batch_sizes', type=str, default='4,8,16',
                        help='Comma-separated face batch sizes to test')
    parser.add_argument('--add_audio', action='store_true')
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]

    # Get video dimensions for proper warmup
    cap = cv2.VideoCapture(args.input)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print("=" * 70)
    print("GPEN Max GPU Optimization Benchmark")
    print(f"  Input: {args.input}")
    print(f"  Video: {video_width}x{video_height}")
    print(f"  Model: {args.model} ({args.in_size}x{args.in_size})")
    print(f"  SR: {'ON (scale=' + str(args.sr_scale) + ')' if args.use_sr else 'OFF'}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  PyTorch: {torch.__version__}")
    print("=" * 70)

    all_results = {}

    for bs in batch_sizes:
        config_name = f"batch{bs}_compile_fp16" + ("_sr" if args.use_sr else "")
        
        if args.output:
            out_path = args.output.replace('.mp4', f'_{config_name}.mp4')
        else:
            base = os.path.splitext(args.input)[0]
            out_path = f"{base}_{config_name}.mp4"

        print(f"\n{'═' * 70}")
        print(f">>> Config: {config_name}")
        print(f"{'═' * 70}")

        try:
            enhancer = MaxOptimizedFaceEnhancement(
                base_dir=BASE_DIR, in_size=args.in_size, model=args.model,
                channel_multiplier=args.channel_multiplier, narrow=args.narrow,
                use_sr=args.use_sr, sr_model=args.sr_model, sr_scale=args.sr_scale,
                tile_size=args.tile_size, alpha=args.alpha, device='cuda',
                face_batch_size=bs, video_height=video_height, video_width=video_width,
            )

            stats = process_video(args.input, out_path, enhancer, args.max_frames)
            all_results[config_name] = stats
            print_stats(config_name, stats)

            # Mux audio if requested
            if args.add_audio:
                final = out_path.replace('.mp4', '_audio.mp4')
                cmd = f'ffmpeg -y -i "{out_path}" -i "{args.input}" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0? -shortest "{final}" 2>/dev/null'
                os.system(cmd)
                print(f"  Audio muxed: {final}")

            del enhancer
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()

    # ── Final comparison ───────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{'═' * 70}")
        print(">>> COMPARISON")
        print(f"{'═' * 70}")
        print(f"{'Config':<35} {'Avg ms':>8} {'FPS':>6} {'Median':>8} {'P95':>8} {'Face ms':>8}")
        print(f"{'─' * 35} {'─' * 8} {'─' * 6} {'─' * 8} {'─' * 8} {'─' * 8}")
        
        best_fps = 0
        best_name = ""
        for name, s in all_results.items():
            face_ms = s.get('avg_ms_face_frames', 0)
            print(f"{name:<35} {s['avg_ms']:>8.1f} {s['avg_fps']:>6.2f} {s['median_ms']:>8.1f} {s['p95_ms']:>8.1f} {face_ms:>8.1f}")
            if s['avg_fps'] > best_fps:
                best_fps = s['avg_fps']
                best_name = name
        
        print(f"\n  Best config: {best_name} ({best_fps:.2f} fps)")


if __name__ == '__main__':
    main()
