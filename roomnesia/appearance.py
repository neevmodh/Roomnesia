"""Appearance channel: changes that geometry cannot see (a rug taken away, a poster swapped, papers on
a desk). Only surfaces BOTH recordings agree are geometrically unchanged are compared, so this channel
never double-counts a moved object, and viewpoint/occlusion are already handled by the depth test.

1. For every static before-voxel, sample the colour the after-cameras saw at that exact 3D spot
   (only views where the depth agrees, i.e. the spot was really visible).
2. Fit one global colour transform (after -> before) on all those pairs: removes exposure / white
   balance differences between the two videos.
3. Compare in CIELAB, weighting chroma over lightness (shadows change lightness, objects change colour).
4. Cluster the voxels that changed colour; keep compact patches of real size.
"""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from .change import STATIC, ChangeCluster, Cloud
from .geometry import project


def _lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)


def _multiview_colour(scene, P, rec, valid, tol_rel, blur):
    """Average colour of each 3D point over every frame of `rec` that really sees it (depth agrees)."""
    acc = np.zeros((len(P), 3))
    cnt = np.zeros(len(P))
    H, W = scene["depth"].shape[1:]
    for j in np.where(scene["rec"] == rec)[0]:
        uv, z = project(P, scene["intrinsic"][j], scene["extrinsic"][j])
        u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        ok = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ok[ok] &= valid[j][v[ok], u[ok]]
        ok[ok] &= np.abs(scene["depth"][j][v[ok], u[ok]] - z[ok]) < tol_rel * z[ok]
        acc[ok] += blur[j][v[ok], u[ok]]
        cnt[ok] += 1
    return acc / np.maximum(cnt, 1)[:, None], cnt


def appearance_changes(scene: dict, a: Cloud, valid: np.ndarray, tol_rel: float, eps: float, exclude: list[np.ndarray],
                       min_points: int = 80, min_area: float = 0.05) -> tuple[list[ChangeCluster], dict]:
    idx = np.where(a.state == STATIC)[0]
    P = a.xyz[idx]
    # slight blur: a 1-2 px misalignment on fine texture must not look like a colour change
    blur = np.stack([cv2.GaussianBlur(im, (0, 0), 2.0) for im in scene["images"]]).astype(np.float32)
    before, nb = _multiview_colour(scene, P, 0, valid, tol_rel, blur)
    after, na = _multiview_colour(scene, P, 1, valid, tol_rel, blur)
    seen = (nb >= 2) & (na >= 2)
    if seen.sum() < 500:
        return [], dict(compared=int(seen.sum()))
    before, after = before[seen], after[seen]
    # global colour correction after -> before (affine per channel, refit on inliers)
    X = np.c_[after, np.ones(len(after))]
    M = np.linalg.lstsq(X, before, rcond=None)[0]
    for _ in range(2):
        r = np.linalg.norm(X @ M - before, axis=1)
        inl = r < np.percentile(r, 80)
        M = np.linalg.lstsq(X[inl], before[inl], rcond=None)[0]
    lab_b, lab_a = _lab(before), _lab(X @ M)
    d = np.sqrt(0.25 * (lab_b[:, 0] - lab_a[:, 0]) ** 2 + (lab_b[:, 1] - lab_a[:, 1]) ** 2 + (lab_b[:, 2] - lab_a[:, 2]) ** 2)
    # threshold from this run's own colour noise (same idea as the depth tolerance)
    mad = 1.4826 * np.median(np.abs(d - np.median(d)))
    thr = max(20.0, float(np.median(d) + 6 * mad))
    flagged = idx[seen][d > thr]
    dd = np.zeros(len(a.xyz))
    dd[idx[seen]] = d
    # ignore colour changes right next to geometric changes (shadows, partial occlusion edges)
    if exclude and len(flagged):
        from scipy.spatial import cKDTree

        ex = cKDTree(np.concatenate(exclude))
        flagged = flagged[ex.query(a.xyz[flagged])[0] > 2 * eps]
    out = []
    if len(flagged) >= min_points:
        labels = DBSCAN(eps=0.6 * eps, min_samples=10).fit_predict(a.xyz[flagged])
        for lab in set(labels) - {-1}:
            c = flagged[labels == lab]
            ext = np.ptp(np.percentile(a.xyz[c], [5, 95], axis=0), axis=0)
            area = np.sort(ext)[-1] * np.sort(ext)[-2]
            fill = len(c) * (0.01 * np.median(scene["depth"])) ** 2 / max(area, 1e-9)  # compact patch, not a scatter
            if len(c) < min_points or area < min_area or fill < 0.3 or np.median(dd[c]) < 1.5 * thr:
                continue
            out.append(ChangeCluster("appearance", c, c, float(min(1.0, np.median(dd[c]) / (2 * thr))), 0))
    stats = dict(compared=int(seen.sum()), colour_noise=float(np.median(d)), threshold=thr, flagged=int(len(flagged)))
    return out, stats
