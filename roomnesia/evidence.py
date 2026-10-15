"""Same-viewpoint evidence: 'what this camera SHOULD see if nothing had changed' vs 'what it DID see'.

The before-recording is used as a memory of the room. For an after-camera, the nearest before-frames are
forward-warped into that exact camera with their depth (z-buffered, at high resolution). Expected and
actual images then share one viewpoint, so camera motion cannot look like a change - and every detected
change can be shown as a picture a human can verify in a second.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .geometry import cam_centers, project

UNSEEN_RGB = np.array([70, 72, 80], np.uint8)


def _nearest_sources(scene: dict, j: int, k: int = 8, focus: np.ndarray | None = None) -> np.ndarray:
    """Before-frames to build the memory from: the most similar viewpoints (position + direction), and,
    when a focus region is given, the frames that actually had that region in view."""
    src = np.where(scene["rec"] == 0)[0]
    C = cam_centers(scene["extrinsic"])
    fwd = scene["extrinsic"][:, 2, :3]
    score = np.linalg.norm(C[src] - C[j], axis=1) + 1.5 * (1 - fwd[src] @ fwd[j])
    pick = list(src[np.argsort(score)[: (k // 2 if focus is not None else k)]])
    if focus is not None:
        H, W = scene["depth"].shape[1:]
        seen = []
        for i in src:
            uv, z = project(focus, scene["intrinsic"][i], scene["extrinsic"][i])
            seen.append(np.mean((z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)))
        for i in src[np.argsort(seen)[::-1]]:
            if len(pick) >= k:
                break
            if i not in pick:
                pick.append(i)
    return np.array(pick)


def _memory_points(scene: dict, j: int, s: int, focus=None):
    """World points + colours from the before-frames that look most like after-camera j."""
    hires = scene["hires"]
    H, W = scene["depth"].shape[1] * s, scene["depth"].shape[2] * s
    v, u = np.mgrid[0:H, 0:W]
    Ps, Cs = [], []
    for i in _nearest_sources(scene, j, focus=focus):
        D = cv2.resize(scene["depth"][i], (W, H), interpolation=cv2.INTER_NEAREST)
        conf = cv2.resize(scene["conf"][i], (W, H), interpolation=cv2.INTER_NEAREST)
        Ki = scene["intrinsic"][i].copy()
        Ki[:2] *= s
        x = (u + 0.5 - Ki[0, 2]) / Ki[0, 0] * D
        y = (v + 0.5 - Ki[1, 2]) / Ki[1, 1] * D
        E = scene["extrinsic"][i]
        P = (np.stack([x, y, D], -1).reshape(-1, 3) - E[:, 3]) @ E[:, :3]
        keep = conf.ravel() > np.percentile(scene["conf"][i], 20)
        Ps.append(P[keep])
        Cs.append(hires[i].reshape(-1, 3)[keep])
    return np.concatenate(Ps), np.concatenate(Cs)


def _splat(P, C, K, E, H, W):
    """z-buffered point splat; removes far surfaces leaking through gaps between near points."""
    from scipy.ndimage import minimum_filter

    uv, zz = project(P, K, E)
    uu, vv = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ok = (zz > 0.05) & (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
    lin, zz, c = vv[ok] * W + uu[ok], zz[ok], C[ok]
    order = np.argsort(-zz)  # far first, near last: the last write wins
    zbuf = np.full(H * W, np.inf)
    cols = np.zeros((H * W, 3), np.uint8)
    zbuf[lin[order]] = zz[order]
    cols[lin[order]] = c[order]
    zb, img = zbuf.reshape(H, W), cols.reshape(H, W, 3)
    zmin = minimum_filter(np.where(np.isfinite(zb), zb, 1e9), size=5)
    zb[np.isfinite(zb) & (zb > zmin * 1.06 + 0.02)] = np.inf
    return img, zb


def render_expected(scene: dict, j: int, scale: int | None = None, focus: np.ndarray | None = None):
    """Expected image + expected depth of after-camera j, rendered from the before-recording.
    Splatted at three resolutions; holes at full resolution (where the after-camera is much closer than
    any before-camera, so points spread thin) are filled from the coarser, denser levels."""
    s = scene["hires"].shape[1] // scene["depth"].shape[1] if scale is None else scale
    H, W = scene["depth"].shape[1] * s, scene["depth"].shape[2] * s
    P, C = _memory_points(scene, j, s, focus)
    img, zb = None, None
    for f in (4, 2, 1):  # coarse -> fine
        K = scene["intrinsic"][j].copy()
        K[:2] *= s / f
        h, w = H // f, W // f
        im, z = _splat(P, C, K, scene["extrinsic"][j], h, w)
        if img is not None:  # fill this level's holes from the coarser level
            up_i = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            up_z = cv2.resize(np.where(np.isfinite(zb), zb, 0).astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            hole = ~np.isfinite(z) & (up_z > 0)
            im[hole], z[hole] = up_i[hole], up_z[hole]
        img, zb = im, z
    hit = np.isfinite(zb)
    small = cv2.morphologyEx(hit.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool) & ~hit
    if small.any():
        img[small] = cv2.inpaint(img, small.astype(np.uint8), 3, cv2.INPAINT_TELEA)[small]
    unseen = ~(hit | small)
    img[unseen] = _hatch(unseen.shape)[unseen]
    return img, zb


def _hatch(shape) -> np.ndarray:
    v, u = np.mgrid[0:shape[0], 0:shape[1]]
    out = np.empty(shape + (3,), np.uint8)
    out[:] = UNSEEN_RGB
    out[((u + v) // 6) % 2 == 0] = (95, 97, 108)
    return out


def change_mask(scene: dict, j: int, P: np.ndarray, shape, depth=None, tol=0.06) -> np.ndarray:
    """Pixels of camera j covered by object points P (optionally only where P is the visible surface)."""
    H, W = shape
    s = H / scene["depth"].shape[1]
    K = scene["intrinsic"][j].copy()
    K[:2] *= s
    uv, z = project(P, K, scene["extrinsic"][j])
    uu, vv = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ok = (z > 0.05) & (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
    if depth is not None:
        ok[ok] &= z[ok] <= depth[vv[ok], uu[ok]] * (1 + tol) + 0.03
    m = np.zeros((H, W), np.uint8)
    m[vv[ok], uu[ok]] = 1
    r = max(3, int(4 * s))
    m = cv2.morphologyEx(cv2.dilate(m, np.ones((r, r), np.uint8)), cv2.MORPH_CLOSE, np.ones((3 * r, 3 * r), np.uint8))
    return m.astype(bool)


def _outline(img, mask, color, label=None):
    out = img.copy()
    tint = out.astype(np.float32)
    tint[mask] = 0.55 * tint[mask] + 0.45 * np.array(color, np.float32)
    out = tint.astype(np.uint8)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, max(2, img.shape[0] // 180), cv2.LINE_AA)
    if label and cnts:
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        _label(out, label, (x, max(18, y - 6)), color)
    return out


def _label(img, text, org, color):
    sc = img.shape[0] / 600
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * sc, (255, 255, 255), max(3, int(4 * sc)), cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * sc, color, max(1, int(2 * sc)), cv2.LINE_AA)


def change_evidence(scene: dict, c, colors: dict) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """For one change: [(caption, expected_crop, actual_crop)] from the same after-camera."""
    from .semantics import best_views

    pairs = []
    spots = []
    if c.kind in ("removed", "moved"):
        spots.append(("old spot", c.points, "through", colors["removed" if c.kind == "removed" else "moved"]))
    if c.kind in ("added", "moved"):
        spots.append(("new spot", c.points_to if c.kind == "moved" else c.points, "surface",
                      colors["added" if c.kind == "added" else "moved_to"]))
    if c.kind == "appearance":
        spots.append(("same surface", c.points, "surface", colors["appearance"]))
    for name, P, mode, col in spots:
        vs = best_views(P, scene, 1, mode, k=3)
        if not vs:
            continue
        # among the best after-views, keep the one the before-memory covers best around the spot
        best = None
        for v in vs:
            e_, d_ = render_expected(scene, v.frame, focus=P)
            m_ = change_mask(scene, v.frame, P, e_.shape[:2])
            ring = cv2.dilate(m_.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
            cover = np.isfinite(d_)[ring].mean() if ring.any() else 0
            if best is None or cover > best[0]:
                best = (cover, v.frame, e_, d_)
        _, j, exp, dexp = best
        act = scene["hires"][j]
        if mode == "through":  # object is in memory, gone in reality: outline it on the expected view
            m = change_mask(scene, j, P, exp.shape[:2], dexp)
            e, a = _outline(exp, m, col, "expected here"), _outline(act, m, col, "gone")
        else:  # object is new (memory shows the empty spot) or the surface looks different
            m = change_mask(scene, j, P, exp.shape[:2])
            labels = ("looked like this", "looks like this now") if c.kind == "appearance" else ("was empty", "new")
            e, a = _outline(exp, m, col, labels[0]), _outline(act, m, col, labels[1])
        pairs.append((name, _crop_around(e, m), _crop_around(a, m)))
    return pairs


def _crop_around(img, mask, pad=0.9, min_px=160):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return img
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(xs.max() - xs.min(), ys.max() - ys.min(), min_px) * (0.5 + pad / 2)
    x0, x1 = int(max(0, cx - half)), int(min(img.shape[1], cx + half))
    y0, y1 = int(max(0, cy - half)), int(min(img.shape[0], cy + half))
    return img[y0:y1, x0:x1]


def evidence_gif(scene: dict, changes, colors: dict, path: Path, n: int = 14, height: int = 300):
    """Walk through the after-video: [expected from memory | actual | surprise] at each viewpoint."""
    from PIL import Image

    frames = np.where(scene["rec"] == 1)[0]
    frames = frames[np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)]
    out = []
    for j in frames:
        exp, dexp = render_expected(scene, j)
        act = scene["hires"][j].copy()
        sur = (0.45 * act).astype(np.uint8)
        for c in changes:
            if c.kind == "appearance":
                m = change_mask(scene, j, c.points, exp.shape[:2], dexp)
                exp, act, sur = (_outline(x, m, colors["appearance"]) for x in (exp, act, sur))
                continue
            if c.kind in ("removed", "moved"):
                m = change_mask(scene, j, c.points, exp.shape[:2], dexp)
                col = colors["removed" if c.kind == "removed" else "moved"]
                exp, sur = _outline(exp, m, col), _outline(sur, m, col)
            if c.kind in ("added", "moved"):
                P = c.points_to if c.kind == "moved" else c.points
                m = change_mask(scene, j, P, exp.shape[:2])
                col = colors["added" if c.kind == "added" else "moved_to"]
                act, sur = _outline(act, m, col), _outline(sur, m, col)
        tiles = []
        for t, cap in ((exp, "EXPECTED (memory of video A)"), (act, "ACTUAL (video B)"), (sur, "SURPRISE")):
            t = cv2.resize(t, (int(t.shape[1] * height / t.shape[0]), height), interpolation=cv2.INTER_AREA)
            _label(t, cap, (8, 22), (30, 30, 30))
            tiles.append(t)
        out.append(Image.fromarray(np.hstack([tiles[0], np.full((height, 4, 3), 255, np.uint8), tiles[1],
                                              np.full((height, 4, 3), 255, np.uint8), tiles[2]])))
    out[0].save(path, save_all=True, append_images=out[1:], duration=450, loop=0, optimize=True)
