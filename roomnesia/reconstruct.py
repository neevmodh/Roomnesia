"""Joint feed-forward reconstruction of both recordings with VGGT, made to fit on a 16 GB Mac.

Both videos go through the network *together*, so every frame - before and after - comes out
in one shared coordinate frame. No separate SfM runs, no fragile point-cloud registration.
"""

from __future__ import annotations

import contextlib
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .frames import model_size


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _patch_chunked_attention(q_chunk: int = 1024) -> None:
    """VGGT's global attention spans every token of every frame (~800 tokens x 48 frames).
    A plain SDPA call on MPS materialises the full N x N matrix (tens of GB), so we process queries
    in chunks. Exact same result, peak memory ~ q_chunk * N instead of N * N."""
    from vggt.layers import attention as A

    def forward(self, x, pos=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q, k = self.rope(q, pos), self.rope(k, pos)
        if N <= 4096:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            out = torch.cat(
                [F.scaled_dot_product_attention(q[:, :, i:i + q_chunk], k, v) for i in range(0, N, q_chunk)], dim=2
            )
        return self.proj(out.transpose(1, 2).reshape(B, N, C))

    A.Attention.forward = forward


def load_vggt(device: torch.device, agg_dtype: torch.dtype):
    """Lean loader for a 16 GB machine. `VGGT.from_pretrained` materialises a random fp32 model *and* a
    second 5 GB copy of the weights on CPU (minutes of swapping). Instead: build the skeleton on the meta
    device, memory-map the safetensors file, and stream each tensor straight to the GPU - the 24-layer
    aggregator in bf16 (it dominates memory and compute), the small camera/depth heads in fp32.
    The point-map and tracking heads are never loaded: depth + camera heads are the more accurate path."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from vggt.models.vggt import VGGT

    path = hf_hub_download("facebook/VGGT-1B", "model.safetensors")
    with _params_on_meta():
        model = VGGT(enable_point=False, enable_track=False)
    state = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            if k.startswith(("point_head", "track_head")):
                continue
            dt = agg_dtype if k.startswith("aggregator") else torch.float32
            state[k] = f.get_tensor(k).to(device=device, dtype=dt)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if unexpected or missing:
        raise RuntimeError(f"weight mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    for name, buf in model.named_buffers():  # small non-persistent buffers were built for real on CPU
        mod = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        mod._buffers[name.rsplit(".", 1)[-1]] = buf.to(device, agg_dtype if name.startswith("aggregator") else buf.dtype)
    return model.eval()


@contextlib.contextmanager
def _params_on_meta():
    """Create parameters on the meta device (no memory, no random init); buffers stay real."""
    old = torch.nn.Module.register_parameter

    def register(self, name, param):
        old(self, name, param)
        if param is not None:
            p = self._parameters[name]
            self._parameters[name] = torch.nn.Parameter(p.to("meta"), requires_grad=p.requires_grad)

    torch.nn.Module.register_parameter = register
    try:
        yield
    finally:
        torch.nn.Module.register_parameter = old


def fit_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Scale-to-cover then centre-crop, so frames from both videos share one tensor size."""
    s = max(h / img.shape[0], w / img.shape[1])
    r = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    y0, x0 = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
    return r[y0:y0 + h, x0:x0 + w]


@torch.no_grad()
def reconstruct(frames_a: list[np.ndarray], frames_b: list[np.ndarray], hires_scale: int = 2) -> dict:
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    _patch_chunked_attention()
    device = pick_device()
    H, W = model_size(*frames_a[0].shape[:2])
    imgs = [fit_to(f, H, W) for f in frames_a + frames_b]
    hires = [fit_to(f, H * hires_scale, W * hires_scale) for f in frames_a + frames_b]
    rec = np.array([0] * len(frames_a) + [1] * len(frames_b), dtype=np.int8)

    t0 = time.time()
    agg_dtype = torch.bfloat16 if device.type in ("mps", "cuda") else torch.float32
    model = load_vggt(device, agg_dtype)

    # Portrait phone video: VGGT was trained on landscape/square crops and mis-estimates focal length on
    # tall images, so run it on frames rotated 90 deg clockwise and rotate every output back afterwards.
    portrait = H > W
    net_imgs = [np.ascontiguousarray(np.rot90(im, k=-1)) for im in imgs] if portrait else imgs
    Hn, Wn = net_imgs[0].shape[:2]
    x = torch.from_numpy(np.stack(net_imgs)).permute(0, 3, 1, 2).float().div(255)[None].to(device)
    print(f"[vggt] {x.shape[1]} frames @ {H}x{W} on {device} (model load {time.time() - t0:.1f}s)")
    t0 = time.time()
    tokens, ps_idx = model.aggregator(x.to(agg_dtype))
    tokens = [t.float() if t is not None else None for t in tokens]
    pose_enc = model.camera_head(tokens)[-1]
    extr, intr = pose_encoding_to_extri_intri(pose_enc, (Hn, Wn))
    depth, conf = model.depth_head(tokens, images=x, patch_start_idx=ps_idx)
    print(f"[vggt] inference {time.time() - t0:.1f}s")

    depth = depth[0, ..., 0].float().cpu().numpy()
    conf = conf[0].float().cpu().numpy()
    extr = extr[0].float().cpu().numpy()
    intr = intr[0].float().cpu().numpy()
    if portrait:
        depth, conf, extr, intr = _unrotate(depth, conf, extr, intr, H)
    out = dict(images=np.stack(imgs), hires=np.stack(hires), depth=depth, conf=conf,
               extrinsic=extr, intrinsic=intr, rec=rec)
    del model, tokens, x
    if device.type == "mps":
        torch.mps.empty_cache()
    return out


def _unrotate(depth, conf, extr, intr, H):
    """Undo a 90 deg clockwise image rotation. Rotated pixel (u', v') = (H - v, u), i.e. the rotated
    camera is the original one rolled about its optical axis: x' = -y, y' = x  ->  X' = Rz X."""
    depth = np.ascontiguousarray(np.rot90(depth, k=1, axes=(1, 2)))
    conf = np.ascontiguousarray(np.rot90(conf, k=1, axes=(1, 2)))
    Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float32)
    extr = np.einsum("ij,sjk->sik", Rz.T, extr)  # E = Rz^T E'
    K = np.zeros_like(intr)
    K[:, 0, 0], K[:, 1, 1] = intr[:, 1, 1], intr[:, 0, 0]
    K[:, 0, 2], K[:, 1, 2] = intr[:, 1, 2], H - intr[:, 0, 2]
    K[:, 2, 2] = 1
    return depth, conf, extr, K


@torch.no_grad()
def metric_scale(images: np.ndarray, depth: np.ndarray, conf: np.ndarray, n: int = 8) -> float | None:
    """VGGT is scale-free. A monocular metric-depth network (Depth Anything V2, indoor) gives a rough
    metres-per-unit factor so distances can be reported in metres. Robust median over frames."""
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError:
        return None
    name = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
    try:
        proc = AutoImageProcessor.from_pretrained(name)
        net = AutoModelForDepthEstimation.from_pretrained(name).to(pick_device()).eval()
    except Exception as e:  # offline etc. - fall back to scene units
        print(f"[scale] metric depth unavailable ({e}); reporting scene units")
        return None
    ratios = []
    for i in np.linspace(0, len(images) - 1, min(n, len(images))).astype(int):
        inp = proc(images=images[i], return_tensors="pt").to(net.device)
        pred = net(**inp).predicted_depth[0].float()
        pred = F.interpolate(pred[None, None], size=depth.shape[1:], mode="bilinear")[0, 0].cpu().numpy()
        m = conf[i] > np.percentile(conf[i], 50)
        ratios.append(np.median(pred[m] / np.maximum(depth[i][m], 1e-6)))
    return float(np.median(ratios))
