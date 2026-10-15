"""Camera maths + putting the reconstruction into a gravity-aligned, metric 'room frame'."""

from __future__ import annotations

import cv2
import numpy as np


def backproject(depth: np.ndarray, K: np.ndarray, E: np.ndarray, stride: int = 1):
    """Depth map (H, W) -> world points (h, w, 3) plus the pixel grid used. E is cam-from-world (3x4)."""
    H, W = depth.shape
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[::stride, ::stride]
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * z
    cam = np.stack([x, y, z], -1)
    R, t = E[:, :3], E[:, 3]
    return (cam - t) @ R, (v, u)


def project(P: np.ndarray, K: np.ndarray, E: np.ndarray):
    """World points (N, 3) -> pixel coords (N, 2) and camera depth (N,)."""
    cam = P @ E[:, :3].T + E[:, 3]
    z = cam[:, 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = cam[:, 0] / zs * K[0, 0] + K[0, 2] - 0.5
    v = cam[:, 1] / zs * K[1, 1] + K[1, 2] - 0.5
    return np.stack([u, v], 1), z


def cam_centers(E: np.ndarray) -> np.ndarray:
    return np.einsum("sji,sj->si", E[:, :, :3], -E[:, :, 3])


def valid_mask(depth: np.ndarray, conf: np.ndarray, conf_thr: float, edge_thr: float = 0.06) -> np.ndarray:
    """Confident pixels away from depth discontinuities (where 'flying pixels' live)."""
    logd = np.log(np.maximum(depth, 1e-6))
    gx = np.abs(cv2.Sobel(logd, cv2.CV_32F, 1, 0, ksize=3)) / 8
    gy = np.abs(cv2.Sobel(logd, cv2.CV_32F, 0, 1, ksize=3)) / 8
    edge = cv2.dilate(((gx + gy) > edge_thr).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return (conf >= conf_thr) & ~edge & (depth > 0)


def to_room_frame(scene: dict, scale: float | None) -> dict:
    """Re-express the scene in metres (if a scale is known) with y pointing up and the floor at y=0.

    Up is first guessed from how people hold phones (camera 'down' axis averaged over all frames),
    then refined by fitting the floor plane with RANSAC among roughly horizontal, low-lying points.
    """
    E, K, D = scene["extrinsic"].astype(np.float64), scene["intrinsic"], scene["depth"]
    s = scale or 1.0
    thr = np.percentile(scene["conf"], 40)
    pts = []
    for i in range(len(D)):
        P, (v, u) = backproject(D[i], K[i], E[i], stride=8)
        m = scene["conf"][i][v, u] >= thr
        pts.append(P[m])
    P = np.concatenate(pts)

    down = np.mean(E[:, 1, :3], 0)  # camera +y (down) axis expressed in world coordinates
    up = -down / np.linalg.norm(down)
    up = _ransac_floor_normal(P, up)
    h = P @ up
    floor_h = np.percentile(h, 2)
    inl = np.abs(h - floor_h) < 0.03 * np.ptp(h)
    if inl.sum() > 100:
        floor_h = np.median(h[inl])

    # x axis: first camera's right vector flattened onto the floor, then snapped to the walls
    right0 = E[0, 0, :3] - (E[0, 0, :3] @ up) * up
    right0 /= np.linalg.norm(right0)
    back0 = np.cross(right0, up)  # right-handed: x right, y up, z back
    theta = _manhattan_yaw(P, up, right0, back0, floor_h)
    right = np.cos(theta) * right0 + np.sin(theta) * back0
    back = np.cross(right, up)
    Rg = np.stack([right, up, back])  # world -> room rotation (rows are room axes)
    c0 = cam_centers(E[:1])[0]
    origin = c0 - ((c0 @ up) - floor_h) * up  # floor point under the first camera
    # first camera's heading in room coordinates, for "left / right / further" language
    fwd0 = E[0, 2, :3] - (E[0, 2, :3] @ up) * up
    fwd0 /= np.linalg.norm(fwd0)
    view0 = np.array([[right0 @ right, right0 @ back], [fwd0 @ right, fwd0 @ back]])

    out = dict(scene)
    R, t = E[:, :, :3], E[:, :, 3]
    out["extrinsic"] = np.concatenate(
        [R @ Rg.T, (s * (np.einsum("sij,j->si", R, origin) + t))[..., None]], axis=2
    ).astype(np.float32)
    out["depth"] = (D * s).astype(np.float32)
    out["metric"] = scale is not None
    out["scale"] = s
    out["view0"] = view0
    return out


def _manhattan_yaw(P, up, right, back, floor_h) -> float:
    """Rotation about 'up' that makes walls axis-aligned: the angle whose x/z histograms of wall-height
    points are peakiest (walls collapse into sharp spikes). Purely cosmetic: nicer maps and viewer."""
    h = P @ up - floor_h
    span = np.percentile(h, 98)
    W = P[(h > 0.2 * span) & (h < 0.8 * span)]
    if len(W) < 200:
        return 0.0
    W = W[np.random.default_rng(0).choice(len(W), min(len(W), 30000), replace=False)]
    x, z = W @ right, W @ back
    bins = max(20, int(np.ptp(np.r_[x, z]) / (0.01 * np.ptp(np.r_[x, z])) ))
    best, best_t = -1, 0.0
    for t in np.deg2rad(np.arange(0, 90, 1.0)):
        xr, zr = np.cos(t) * x + np.sin(t) * z, -np.sin(t) * x + np.cos(t) * z
        sc = (np.histogram(xr, bins)[0].astype(float) ** 2).sum() + (np.histogram(zr, bins)[0].astype(float) ** 2).sum()
        if sc > best:
            best, best_t = sc, t
    return float(best_t)


def _ransac_floor_normal(P: np.ndarray, up0: np.ndarray, iters: int = 400, max_angle_deg: float = 30) -> np.ndarray:
    """Find the dominant horizontal plane among the lowest points; return its (upward) normal."""
    rng = np.random.default_rng(0)
    h = P @ up0
    low = P[h < np.percentile(h, 30)]
    if len(low) < 50:
        return up0
    sub = low[rng.choice(len(low), min(len(low), 20000), replace=False)]
    tol = 0.01 * np.ptp(h)
    best, best_n, best_p = 0, up0, None
    for _ in range(iters):
        a, b, c = sub[rng.choice(len(sub), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if n @ up0 < 0:
            n = -n
        if np.degrees(np.arccos(np.clip(n @ up0, -1, 1))) > max_angle_deg:
            continue
        cnt = int((np.abs((sub - a) @ n) < tol).sum())
        if cnt > best:
            best, best_n, best_p = cnt, n, a
    if best_p is None:
        return up0
    # least-squares refit on the inliers of the best plane
    inl = sub[np.abs((sub - best_p) @ best_n) < tol]
    if len(inl) > 50:
        _, _, vt = np.linalg.svd(inl - inl.mean(0), full_matrices=False)
        n = vt[2] if vt[2] @ up0 > 0 else -vt[2]
        if np.degrees(np.arccos(np.clip(n @ up0, -1, 1))) < max_angle_deg:
            return n
    return best_n
