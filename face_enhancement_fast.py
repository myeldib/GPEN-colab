"""
Optimized GPEN Face Enhancement Pipeline.
Directly optimizes all bottlenecks in the original GPEN code:
1. SR: Avoid numpy<->tensor conversions, keep data on GPU
2. FaceGAN: Batched inference, FP16, keep tensors on GPU  
3. ParseNet: GPU-resident mask generation, skip CPU round-trip
4. Detection: Cached PriorBox, pre-allocated scales
5. Composite: GPU-accelerated alpha blending
6. All models: torch.compile + FP16 autocast
"""

import os
import sys
import cv2
import time
import numpy as np
import torch
import torch.nn.functional as F

# Setup paths exactly like the original GPEN code
import __init_paths

from align_faces import warp_and_crop_face, get_reference_facial_points
from face_detect.retinaface_detection import RetinaFaceDetection
from face_parse.face_parsing import FaceParse
from face_model.face_gan import FaceGAN
from sr_model.real_esrnet import RealESRNet

# These modules are importable because __init_paths adds face_detect/ to sys.path
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.box_utils import decode, decode_landm


class FastFaceEnhancement:
    """
    Maximum-throughput face enhancement pipeline.
    
    All GPU models are torch.compiled with FP16 autocast.
    Minimizes CPU<->GPU data transfers by keeping tensors on GPU across stages.
    """

    def __init__(self, base_dir='./', in_size=512, model='GPEN-BFR-512',
                 channel_multiplier=2, narrow=1, use_sr=False, sr_model='realesrnet',
                 sr_scale=2, tile_size=0, alpha=1.0, device='cuda',
                 use_compile=True, use_fp16=True):
        
        self.device = device
        self.in_size = in_size
        self.out_size = in_size
        self.threshold = 0.9
        self.alpha = alpha
        self.use_sr = use_sr
        self.use_compile = use_compile
        self.use_fp16 = use_fp16

        # Global PyTorch speed flags
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # ── Load models ─────────────────────────────────────────
        self.facedetector = RetinaFaceDetection(base_dir, device)
        self.facegan = FaceGAN(base_dir, in_size, None, model, channel_multiplier,
                               narrow, key=None, device=device)
        self.faceparser = FaceParse(base_dir, device=device)
        
        if use_sr:
            self.srmodel = RealESRNet(base_dir, sr_model, sr_scale, tile_size, device=device)
            self.sr_scale = sr_scale
        else:
            self.srmodel = None
            self.sr_scale = 1

        # ── Pre-compute constants ────────────────────────────────
        # Face mask (pre-blurred, reused every frame)
        mask_np = np.zeros((512, 512), np.float32)
        cv2.rectangle(mask_np, (26, 26), (486, 486), (1, 1, 1), -1, cv2.LINE_AA)
        mask_np = cv2.GaussianBlur(mask_np, (101, 101), 4)
        mask_np = cv2.GaussianBlur(mask_np, (101, 101), 4)
        self.mask_np = mask_np

        self.kernel = np.array([
            [0.0625, 0.125, 0.0625],
            [0.125, 0.25, 0.125],
            [0.0625, 0.125, 0.0625]], dtype="float32")

        self.reference_5pts = get_reference_facial_points(
            (in_size, in_size), 0.25, (0, 0), True)

        # Face parsing colormap as tensor for GPU mask generation
        self.mask_colormap = torch.tensor(
            [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 255, 255, 0],
            device=device, dtype=torch.uint8)

        # Cache for PriorBox (same frame size = same priors)
        self._prior_cache = {}
        # Pre-allocated scale tensors
        self._scale_cache = {}

        # ── Compile models ──────────────────────────────────────
        if use_compile:
            self._compile_models()
        
        print(f"[FastFaceEnhancement] Ready: compile={use_compile}, fp16={use_fp16}, sr={use_sr}")

    def _compile_models(self):
        """torch.compile all GPU models."""
        print("[Compile] FaceGAN (mode='default')...")
        self.facegan.model = torch.compile(
            self.facegan.model, mode='default', fullgraph=False)
        
        print("[Compile] ParseNet (mode='default')...")
        self.faceparser.faceparse = torch.compile(
            self.faceparser.faceparse, mode='default', fullgraph=False)
        
        if self.srmodel is not None:
            print("[Compile] SR model (mode='default')...")
            self.srmodel.srmodel = torch.compile(
                self.srmodel.srmodel, mode='default', fullgraph=False)

    def warmup(self, video_height=720, video_width=960, batch_sizes=None):
        """Trigger JIT compilation for all models at typical input sizes."""
        t0 = time.time()
        
        if batch_sizes is None:
            batch_sizes = [1]
        
        # Compute power-of-2 warmup sizes for face models (avoids recompilation)
        # Always warmup at 1, 2, 4, 8 — covering typical face counts per batch
        face_warmup_sizes = sorted(set([1, 2, 4, 8]))
        
        # FaceGAN warmup — power-of-2 face batch sizes
        for bs in face_warmup_sizes:
            print(f"[Warmup] FaceGAN batch={bs}...")
            w = torch.randn(bs, 3, self.in_size, self.in_size, device=self.device)
            n_iters = 5 if bs == 1 else 3
            for _ in range(n_iters):
                with torch.no_grad():
                    if self.use_fp16:
                        with torch.amp.autocast('cuda'):
                            self.facegan.model(w)
                    else:
                        self.facegan.model(w)
                torch.cuda.synchronize()
            del w

        # ParseNet warmup — power-of-2 face batch sizes
        for bs in face_warmup_sizes:
            print(f"[Warmup] ParseNet batch={bs}...")
            w = torch.randn(bs, 3, 512, 512, device=self.device)
            n_iters = 3
            for _ in range(n_iters):
                with torch.no_grad():
                    if self.use_fp16:
                        with torch.amp.autocast('cuda'):
                            self.faceparser.faceparse(w)
                    else:
                        self.faceparser.faceparse(w)
                torch.cuda.synchronize()
            del w

        # SR warmup at actual video resolution (batch=1 only — batched SR is slower)
        if self.srmodel is not None:
            h, w_dim = video_height, video_width
            mod = 2 if self.sr_scale == 2 else (4 if self.sr_scale == 1 else 1)
            h_pad = (mod - h % mod) % mod
            w_pad = (mod - w_dim % mod) % mod
            sr_h = h + h_pad
            sr_w = w_dim + w_pad
            print(f"[Warmup] SR model at {sr_h}x{sr_w}...")
            sw = torch.randn(1, 3, sr_h, sr_w, device=self.device)
            for _ in range(3):
                with torch.no_grad():
                    if self.use_fp16:
                        with torch.amp.autocast('cuda'):
                            self.srmodel.srmodel(sw)
                    else:
                        self.srmodel.srmodel(sw)
                torch.cuda.synchronize()
            del sw

        torch.cuda.empty_cache()
        print(f"[Warmup] Complete in {time.time()-t0:.1f}s")

    # ─────────────────────────────────────────────────────────────
    # Optimized SR: Keep tensor on GPU, minimize CPU<->GPU transfers
    # ─────────────────────────────────────────────────────────────
    def _sr_process_fast(self, img_np):
        """
        Super-resolution that minimizes data movement.
        Returns (sr_output_np, sr_output_tensor_gpu) — keeps GPU copy for composite blending.
        """
        # Numpy BGR -> tensor RGB, directly to GPU
        img_rgb = img_np[:, :, ::-1].copy()  # BGR -> RGB
        img_t = torch.from_numpy(img_rgb).to(self.device, dtype=torch.float32)
        img_t = img_t.permute(2, 0, 1).unsqueeze(0) / 255.0  # (1, 3, H, W)

        scale = self.srmodel.scale
        mod_scale = 2 if scale == 2 else (4 if scale == 1 else None)

        h_pad, w_pad = 0, 0
        if mod_scale is not None:
            _, _, h, w = img_t.shape
            h_pad = (mod_scale - h % mod_scale) % mod_scale
            w_pad = (mod_scale - w % mod_scale) % mod_scale
            if h_pad or w_pad:
                img_t = F.pad(img_t, (0, w_pad, 0, h_pad), 'reflect')

        with torch.no_grad():
            if self.use_fp16:
                with torch.amp.autocast('cuda'):
                    output = self.srmodel.srmodel(img_t)
                output = output.float()
            else:
                output = self.srmodel.srmodel(img_t)

        del img_t

        # Remove padding
        if mod_scale is not None and (h_pad or w_pad):
            _, _, h_out, w_out = output.shape
            output = output[:, :, :h_out - h_pad * scale, :w_out - w_pad * scale]

        # Keep GPU tensor for later blending (avoid CPU->GPU round-trip)
        output_gpu = output.squeeze(0).clamp_(0, 1)
        # Convert (3,H,W) RGB to (H,W,3) BGR on GPU for reuse
        sr_gpu_bhw3 = output_gpu.flip(0).permute(1, 2, 0).contiguous()
        
        # Also get numpy for face detection + warpAffine (CPU ops)
        output_np = (sr_gpu_bhw3.cpu().numpy() * 255.0).round().astype(np.uint8)
        
        # Scale GPU tensor to [0,255] float for later blending  
        sr_gpu_f32 = sr_gpu_bhw3 * 255.0
        del output, output_gpu, sr_gpu_bhw3

        return output_np, sr_gpu_f32

    def _sr_batch_fast(self, frames):
        """
        Batched super-resolution: stack N frames -> single RRDBNet forward pass.
        All frames must be the same resolution (true for video).
        
        Args:
            frames: list of BGR uint8 numpy arrays (all same H x W)
        Returns:
            list of BGR uint8 numpy arrays (upscaled)
        """
        if not frames:
            return []
        n = len(frames)
        
        # Stack all frames: BGR -> RGB, build (N, 3, H, W) tensor in one transfer
        imgs_rgb = np.stack([f[:, :, ::-1].copy() for f in frames], axis=0)  # (N, H, W, 3)
        batch = torch.from_numpy(imgs_rgb).to(self.device, dtype=torch.float32)
        batch = batch.permute(0, 3, 1, 2) / 255.0  # (N, 3, H, W)
        del imgs_rgb

        scale = self.srmodel.scale
        mod_scale = 2 if scale == 2 else (4 if scale == 1 else None)

        h_pad, w_pad = 0, 0
        if mod_scale is not None:
            _, _, h, w = batch.shape
            h_pad = (mod_scale - h % mod_scale) % mod_scale
            w_pad = (mod_scale - w % mod_scale) % mod_scale
            if h_pad or w_pad:
                batch = F.pad(batch, (0, w_pad, 0, h_pad), 'reflect')

        with torch.no_grad():
            if self.use_fp16:
                with torch.amp.autocast('cuda'):
                    output = self.srmodel.srmodel(batch)
                output = output.float()
            else:
                output = self.srmodel.srmodel(batch)
        del batch

        # Remove padding from all frames
        if mod_scale is not None and (h_pad or w_pad):
            _, _, h_out, w_out = output.shape
            output = output[:, :, :h_out - h_pad * scale, :w_out - w_pad * scale]

        # Batch convert to numpy: (N,3,H,W) -> (N,H,W,3), RGB->BGR
        output = output.clamp_(0, 1)
        # flip channel dim RGB->BGR, permute to NHWC
        output_np = output.flip(1).permute(0, 2, 3, 1).contiguous().cpu().numpy()
        output_np = (output_np * 255.0).round().astype(np.uint8)
        del output

        return [output_np[i] for i in range(n)]

    # ─────────────────────────────────────────────────────────────
    # Optimized face detection with cached PriorBox
    # ─────────────────────────────────────────────────────────────
    def _detect_fast(self, img_raw, confidence_threshold=0.9, nms_threshold=0.4):
        """Face detection with cached priors, GPU preprocessing, and pre-allocated scales."""
        im_height, im_width = img_raw.shape[:2]
        
        ss = 1.0
        if max(im_height, im_width) > 1500:
            ss = 1000.0 / max(im_height, im_width)
            img_raw = cv2.resize(img_raw, (0, 0), fx=ss, fy=ss)
            im_height, im_width = img_raw.shape[:2]

        # Cache scale tensors for this resolution
        size_key = (im_height, im_width)
        if size_key not in self._scale_cache:
            self._scale_cache[size_key] = {
                'scale': torch.tensor([im_width, im_height, im_width, im_height],
                                       device=self.device, dtype=torch.float32),
                'scale1': torch.tensor([im_width, im_height] * 5,
                                        device=self.device, dtype=torch.float32),
            }
        scales = self._scale_cache[size_key]

        # Transfer to GPU, subtract mean on GPU (faster for large images)
        img_t = torch.from_numpy(img_raw.transpose(2, 0, 1)).unsqueeze(0).to(
            self.device, dtype=torch.float32)
        img_t -= self.facedetector.mean

        # Forward pass
        with torch.no_grad():
            loc, conf, landms = self.facedetector.net(img_t)
        del img_t

        # Cache PriorBox computation
        if size_key not in self._prior_cache:
            priorbox = PriorBox(self.facedetector.cfg, image_size=size_key)
            self._prior_cache[size_key] = priorbox.forward().to(self.device).data
        prior_data = self._prior_cache[size_key]

        # Decode
        boxes = decode(loc.data.squeeze(0), prior_data, self.facedetector.cfg['variance'])
        boxes = (boxes * scales['scale']).cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.facedetector.cfg['variance'])
        landms = (landms * scales['scale1']).cpu().numpy()

        # Filter by confidence (vectorized)
        mask = scores > confidence_threshold
        boxes = boxes[mask]
        landms = landms[mask]
        scores = scores[mask]

        # Top-K + NMS
        if len(scores) > 0:
            order = scores.argsort()[::-1][:5000]
            boxes, landms, scores = boxes[order], landms[order], scores[order]

            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, nms_threshold)
            dets = dets[keep][:750]
            landms = landms[keep][:750]
        else:
            dets = np.empty((0, 5), dtype=np.float32)
            landms = np.empty((0, 10), dtype=np.float32)

        landms = landms.reshape((-1, 5, 2)).transpose((0, 2, 1)).reshape(-1, 10)
        return dets / ss, landms / ss

    # ─────────────────────────────────────────────────────────────
    # Optimized FaceGAN: batch inference, FP16, minimal transfers
    # ─────────────────────────────────────────────────────────────
    def _facegan_batch(self, face_imgs):
        """
        Batch-process face crops through FaceGAN.
        face_imgs: list of numpy BGR uint8 arrays (already in_size x in_size)
        Returns: list of numpy BGR uint8 enhanced face arrays
        
        Pads batch to next power-of-2 to avoid torch.compile recompilation
        on variable face counts across frames.
        """
        if not face_imgs:
            return []

        n = len(face_imgs)
        
        # Pad to next power-of-2 to avoid torch.compile recompilation
        pad_n = 1
        while pad_n < n:
            pad_n *= 2
        
        # Build batch tensor on GPU — fused conversion
        # Stack numpy first, then single transfer
        imgs_np = np.stack(face_imgs, axis=0)  # (N, H, W, 3) uint8
        if pad_n > n:
            # Pad with zeros (dummy faces that we'll discard)
            pad_shape = (pad_n - n,) + face_imgs[0].shape
            imgs_np = np.concatenate([imgs_np, np.zeros(pad_shape, dtype=np.uint8)], axis=0)
        
        batch = torch.from_numpy(imgs_np).to(self.device, dtype=torch.float32)
        batch = batch / 255.0
        if self.facegan.is_norm:
            batch = (batch - 0.5) / 0.5
        # (N, H, W, 3) -> (N, 3, H, W), BGR->RGB
        batch = batch.permute(0, 3, 1, 2).flip(1)

        with torch.no_grad():
            if self.use_fp16:
                with torch.amp.autocast('cuda'):
                    out, _ = self.facegan.model(batch)
                out = out.float()
            else:
                out, _ = self.facegan.model(batch)

        del batch

        # Denormalize
        if self.facegan.is_norm:
            out = out * 0.5 + 0.5
        # (N, 3, H, W) -> (N, H, W, 3), RGB->BGR
        out = out.permute(0, 2, 3, 1).flip(3).clamp_(0, 1)
        
        # Single GPU->CPU transfer, take only real faces (discard padding)
        out_np = (out[:n].cpu().numpy() * 255.0).round().astype(np.uint8)
        del out

        return [out_np[i] for i in range(n)]

    # ─────────────────────────────────────────────────────────────
    # Optimized ParseNet: GPU-resident mask, skip CPU round-trip
    # ─────────────────────────────────────────────────────────────
    def _parse_face_fast(self, ef_np):
        """
        Parse a face to get the binary mask, optimized path.
        ef_np: enhanced face numpy BGR uint8 (512x512)
        Returns: mask numpy float32 (512x512) after postprocessing
        """
        # Convert to tensor on GPU — fused
        # BGR->RGB, normalize to [-1, 1]
        img_rgb = ef_np[:, :, ::-1].copy()
        img_t = torch.from_numpy(img_rgb).to(self.device, dtype=torch.float32)
        img_t = img_t / 255.0 * 2.0 - 1.0
        img_t = img_t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 512, 512)

        with torch.no_grad():
            if self.use_fp16:
                with torch.amp.autocast('cuda'):
                    pred_mask, _ = self.faceparser.faceparse(img_t)
            else:
                pred_mask, _ = self.faceparser.faceparse(img_t)

        del img_t

        # GPU-side argmax + colormap lookup (avoid CPU loop)
        mask_idx = pred_mask.squeeze(0).argmax(dim=0)  # (512, 512) int64
        mask = self.mask_colormap[mask_idx]  # (512, 512) uint8 via indexing
        mask_np = mask.cpu().numpy().astype(np.float32) / 255.0

        # Post-process (thresholding + blur — must be CPU/OpenCV)
        mask_np[:26, :] = 0; mask_np[-26:, :] = 0
        mask_np[:, :26] = 0; mask_np[:, -26:] = 0
        mask_np = cv2.GaussianBlur(mask_np, (101, 101), 4)
        mask_np = cv2.GaussianBlur(mask_np, (101, 101), 4)

        return mask_np

    def _parse_batch_fast(self, ef_list):
        """
        Batch-parse multiple enhanced faces through ParseNet in one forward pass.
        ef_list: list of BGR uint8 numpy arrays (512x512)
        Returns: list of float32 numpy masks (512x512) after postprocessing
        
        Pads batch to next power-of-2 to avoid torch.compile recompilation.
        """
        if not ef_list:
            return []

        n = len(ef_list)
        
        # Pad to next power-of-2
        pad_n = 1
        while pad_n < n:
            pad_n *= 2
        
        # Stack + convert in one shot: BGR->RGB, normalize to [-1,1]
        imgs_np = np.stack([ef[:, :, ::-1].copy() for ef in ef_list], axis=0)  # (N,512,512,3)
        if pad_n > n:
            pad_shape = (pad_n - n, 512, 512, 3)
            imgs_np = np.concatenate([imgs_np, np.zeros(pad_shape, dtype=np.uint8)], axis=0)
        batch = torch.from_numpy(imgs_np).to(self.device, dtype=torch.float32)
        batch = batch / 255.0 * 2.0 - 1.0
        batch = batch.permute(0, 3, 1, 2)  # (N, 3, 512, 512)

        with torch.no_grad():
            if self.use_fp16:
                with torch.amp.autocast('cuda'):
                    pred_masks, _ = self.faceparser.faceparse(batch)
            else:
                pred_masks, _ = self.faceparser.faceparse(batch)
        del batch

        # GPU-side argmax + colormap for all faces at once (only real faces, not padding)
        mask_indices = pred_masks[:n].argmax(dim=1)  # (n, 512, 512)
        masks_gpu = self.mask_colormap[mask_indices]  # (n, 512, 512)
        masks_np = masks_gpu.cpu().numpy().astype(np.float32) / 255.0
        del pred_masks, mask_indices, masks_gpu

        # Post-process each mask (blur is CPU-only)
        results = []
        for i in range(n):
            m = masks_np[i]
            m[:26, :] = 0; m[-26:, :] = 0
            m[:, :26] = 0; m[:, -26:] = 0
            m = cv2.GaussianBlur(m, (101, 101), 4)
            m = cv2.GaussianBlur(m, (101, 101), 4)
            results.append(m)
        return results

    # ─────────────────────────────────────────────────────────────
    # Cross-frame batch processing (DDColor-style)
    # ─────────────────────────────────────────────────────────────
    def process_batch(self, frames, timing=None):
        """
        Process N frames together — DDColor-style cross-frame batching.
        
        Pipeline:
          1. Batch SR all frames (single forward pass through RRDBNet)
          2. Detect faces in each frame (sequential — variable output)
          3. Batch ALL face crops from all frames through FaceGAN
          4. Batch ALL enhanced faces through ParseNet
          5. GPU-accelerated composite per frame
        
        Args:
            frames: list of BGR uint8 numpy arrays (all same resolution for batched SR)
            timing: if dict, filled with per-component ms totals
        Returns:
            list of (result_img, orig_faces, enhanced_faces) tuples
        """
        n_frames = len(frames)
        if timing is not None:
            timing.clear()

        # ── 1. SR all frames (sequential — GPU already saturated per frame) ──
        t = time.time()
        sr_frames = []      # numpy for warpAffine
        sr_gpu_frames = []  # GPU tensors for final blending (avoid CPU->GPU)
        work_frames = []
        if self.srmodel is not None:
            for frame in frames:
                result = self._sr_process_fast(frame)
                if result is not None:
                    img_sr_np, img_sr_gpu = result
                    work = cv2.resize(frame, img_sr_np.shape[:2][::-1])
                    sr_frames.append(img_sr_np)
                    sr_gpu_frames.append(img_sr_gpu)
                    work_frames.append(work)
                else:
                    sr_frames.append(None)
                    sr_gpu_frames.append(None)
                    work_frames.append(frame)
        else:
            for frame in frames:
                sr_frames.append(None)
                sr_gpu_frames.append(None)
                work_frames.append(frame)
        torch.cuda.synchronize()
        if timing is not None:
            timing['sr_ms'] = (time.time() - t) * 1000

        # ── 2. Detect faces in all frames ───────────────────────
        t = time.time()
        all_detections = []
        for wf in work_frames:
            facebs, landms = self._detect_fast(wf)
            all_detections.append((facebs, landms))
        torch.cuda.synchronize()
        if timing is not None:
            timing['detect_ms'] = (time.time() - t) * 1000

        # ── 3. Align + collect ALL face crops ───────────────────
        t = time.time()
        # Per-frame face data: list of (frame_idx, of, tfm_inv, fh, fw)
        all_face_data = []
        frame_face_ranges = []  # (start, end) indices into all_face_data

        for fi, (facebs, landms) in enumerate(all_detections):
            start = len(all_face_data)
            wf = work_frames[fi]
            for faceb, facial5points in zip(facebs, landms):
                if faceb[4] < self.threshold:
                    continue
                fh = faceb[3] - faceb[1]
                fw = faceb[2] - faceb[0]
                facial5points = np.reshape(facial5points, (2, 5))
                of, tfm_inv = warp_and_crop_face(
                    wf, facial5points, reference_pts=self.reference_5pts,
                    crop_size=(self.in_size, self.in_size))
                all_face_data.append((fi, of, tfm_inv, fh, fw))
            frame_face_ranges.append((start, len(all_face_data)))
        if timing is not None:
            timing['align_ms'] = (time.time() - t) * 1000

        total_faces = len(all_face_data)

        # ── 4. Batch ALL face crops through FaceGAN ─────────────
        t = time.time()
        if total_faces > 0:
            all_crops = [cv2.resize(fd[1], (self.in_size, self.in_size))
                         for fd in all_face_data]
            all_enhanced = self._facegan_batch(all_crops)
        else:
            all_enhanced = []
        torch.cuda.synchronize()
        if timing is not None:
            timing['facegan_ms'] = (time.time() - t) * 1000

        # ── 5. Batch ALL enhanced faces through ParseNet ────────
        t = time.time()
        if total_faces > 0:
            all_masks = self._parse_batch_fast(all_enhanced)
        else:
            all_masks = []
        torch.cuda.synchronize()
        if timing is not None:
            timing['parse_ms'] = (time.time() - t) * 1000

        # ── 6. Composite each frame (GPU-accelerated blending) ──
        t = time.time()
        results = []
        for fi in range(n_frames):
            wf = work_frames[fi]
            height, width = wf.shape[:2]
            start, end = frame_face_ranges[fi]

            orig_faces = []
            enhanced_faces = []

            if start < end:
                full_mask = np.zeros((height, width), dtype=np.float32)
                full_img = np.zeros(wf.shape, dtype=np.uint8)

                for gi in range(start, end):
                    _, of, tfm_inv, fh, fw = all_face_data[gi]
                    ef = all_enhanced[gi]
                    mask_i = all_masks[gi]

                    orig_faces.append(of)
                    enhanced_faces.append(ef)

                    tmp_mask = cv2.resize(mask_i, (self.in_size, self.in_size))
                    tmp_mask = cv2.warpAffine(tmp_mask, tfm_inv, (width, height), flags=3)

                    if min(fh, fw) < 100:
                        ef = cv2.filter2D(ef, -1, self.kernel)

                    ef = cv2.addWeighted(ef, self.alpha, of, 1.0 - self.alpha, 0.0)

                    if self.in_size != self.out_size:
                        ef = cv2.resize(ef, (self.in_size, self.in_size))
                    tmp_img = cv2.warpAffine(ef, tfm_inv, (width, height), flags=3)

                    diff = tmp_mask - full_mask
                    pos = np.where(diff > 0)
                    full_mask[pos] = tmp_mask[pos]
                    full_img[pos] = tmp_img[pos]

                # GPU-accelerated final blending (reuse SR GPU tensor, avoid CPU->GPU)
                full_mask_t = torch.from_numpy(full_mask).to(self.device, dtype=torch.float32)
                full_mask_3d = full_mask_t.unsqueeze(2)  # (H, W, 1)
                full_img_t = torch.from_numpy(full_img).to(self.device, dtype=torch.float32)
                if sr_gpu_frames[fi] is not None:
                    # Reuse GPU tensor directly (no CPU->GPU transfer for SR base!)
                    base_t = sr_gpu_frames[fi]
                else:
                    base_t = torch.from_numpy(wf).to(self.device, dtype=torch.float32)
                result_t = base_t * (1.0 - full_mask_3d) + full_img_t * full_mask_3d
                result = result_t.clamp_(0, 255).to(torch.uint8).cpu().numpy()
                del full_mask_t, full_mask_3d, full_img_t, result_t
                if sr_gpu_frames[fi] is not None:
                    del sr_gpu_frames[fi]
            else:
                result = sr_frames[fi] if sr_frames[fi] is not None else wf.copy()
                if sr_gpu_frames[fi] is not None:
                    del sr_gpu_frames[fi]

            results.append((result, orig_faces, enhanced_faces))

        if timing is not None:
            timing['composite_ms'] = (time.time() - t) * 1000

        return results

    # ─────────────────────────────────────────────────────────────
    # Main pipeline
    # ─────────────────────────────────────────────────────────────
    def process(self, img, timing=None):
        """
        Process a single frame: SR → detect → crop → enhance → parse → blend.
        
        Args:
            img: BGR uint8 numpy array
            timing: if dict, filled with per-component ms timings
        Returns:
            (result_img, orig_faces, enhanced_faces)
        """
        if timing is not None:
            timing.clear()

        # ── SR background ──────────────────────────────────────
        t = time.time()
        img_sr = None
        img_sr_gpu = None
        if self.srmodel is not None:
            result = self._sr_process_fast(img)
            if result is not None:
                img_sr, img_sr_gpu = result
                img = cv2.resize(img, img_sr.shape[:2][::-1])
        torch.cuda.synchronize()
        if timing is not None:
            timing['sr_ms'] = (time.time() - t) * 1000

        # ── Face detection ─────────────────────────────────────
        t = time.time()
        facebs, landms = self._detect_fast(img)
        torch.cuda.synchronize()
        if timing is not None:
            timing['detect_ms'] = (time.time() - t) * 1000

        height, width = img.shape[:2]

        # ── Collect & align faces ──────────────────────────────
        t = time.time()
        valid_faces = []
        for faceb, facial5points in zip(facebs, landms):
            if faceb[4] < self.threshold:
                continue
            fh = faceb[3] - faceb[1]
            fw = faceb[2] - faceb[0]
            facial5points = np.reshape(facial5points, (2, 5))
            of, tfm_inv = warp_and_crop_face(
                img, facial5points, reference_pts=self.reference_5pts,
                crop_size=(self.in_size, self.in_size))
            valid_faces.append((of, tfm_inv, fh, fw))
        if timing is not None:
            timing['align_ms'] = (time.time() - t) * 1000

        orig_faces = []
        enhanced_faces = []

        if valid_faces:
            # ── Batch FaceGAN ──────────────────────────────────
            t = time.time()
            crops = [cv2.resize(vf[0], (self.in_size, self.in_size)) for vf in valid_faces]
            enhanced_list = self._facegan_batch(crops)
            torch.cuda.synchronize()
            if timing is not None:
                timing['facegan_ms'] = (time.time() - t) * 1000

            # ── Parse + blend each face back ───────────────────
            t = time.time()
            full_mask = np.zeros((height, width), dtype=np.float32)
            full_img = np.zeros(img.shape, dtype=np.uint8)

            for (of, tfm_inv, fh, fw), ef in zip(valid_faces, enhanced_list):
                orig_faces.append(of)
                enhanced_faces.append(ef)

                # Parse mask (optimized — GPU argmax + colormap)
                tmp_mask = self._parse_face_fast(ef)
                tmp_mask = cv2.resize(tmp_mask, (self.in_size, self.in_size))
                tmp_mask = cv2.warpAffine(tmp_mask, tfm_inv, (width, height), flags=3)

                # Small face smoothing
                if min(fh, fw) < 100:
                    ef = cv2.filter2D(ef, -1, self.kernel)

                # Alpha blend
                ef = cv2.addWeighted(ef, self.alpha, of, 1.0 - self.alpha, 0.0)

                if self.in_size != self.out_size:
                    ef = cv2.resize(ef, (self.in_size, self.in_size))
                tmp_img = cv2.warpAffine(ef, tfm_inv, (width, height), flags=3)

                # Accumulate mask and image
                diff = tmp_mask - full_mask
                pos = np.where(diff > 0)
                full_mask[pos] = tmp_mask[pos]
                full_img[pos] = tmp_img[pos]

            torch.cuda.synchronize()
            if timing is not None:
                timing['parse_blend_ms'] = (time.time() - t) * 1000

            # ── Final composite (GPU-accelerated, reuse SR tensor) ──
            t = time.time()
            full_mask_t = torch.from_numpy(full_mask).to(self.device, dtype=torch.float32)
            full_mask_3d = full_mask_t.unsqueeze(2)  # (H, W, 1)
            full_img_t = torch.from_numpy(full_img).to(self.device, dtype=torch.float32)
            if img_sr_gpu is not None:
                base_t = img_sr_gpu  # Reuse GPU tensor (no CPU->GPU transfer!)
            else:
                base_t = torch.from_numpy(img).to(self.device, dtype=torch.float32)
            
            result_t = base_t * (1.0 - full_mask_3d) + full_img_t * full_mask_3d
            result = result_t.clamp_(0, 255).to(torch.uint8).cpu().numpy()
            del full_mask_t, full_mask_3d, full_img_t, result_t
            if img_sr_gpu is not None:
                del img_sr_gpu
            if timing is not None:
                timing['composite_ms'] = (time.time() - t) * 1000
        else:
            if timing is not None:
                timing['facegan_ms'] = 0
                timing['parse_blend_ms'] = 0
                timing['composite_ms'] = 0
            if img_sr is not None:
                result = img_sr
            else:
                result = img.copy()
            if img_sr_gpu is not None:
                del img_sr_gpu

        return result, orig_faces, enhanced_faces
