"""Visibility-aware 3D change detection.

The key question for every surface point seen in one recording is not "is there a point near it
in the other cloud?" (that confuses *unobserved* with *removed*) but:

    "Did any camera of the other recording look *through* this spot and see something behind it?"

For a point p from recording A, rendered into every camera j of recording B:

    B depth at p's pixel  >  depth(p) + tol   ->  FREE   (B saw empty space where p was: evidence of change)
    B depth at p's pixel  ~= depth(p)         ->  SAME   (B saw the same surface: evidence of no change)
    B depth at p's pixel  <  depth(p) - tol   ->  OCCLUDED (something in front, no information)
    p outside B's view                       ->  UNSEEN (no information)

A point is 'changed' only with positive free-space evidence from several views, 'static' with
consistency evidence, and 'unobserved' otherwise - so occlusions and partial coverage are never
reported as changes. Run A->B for removals, B->A for additions; a removal + an addition of the
same-looking object is a move.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from .geometry import backproject, cam_centers, project, valid_mask

STATIC, CHANGED, UNOBSERVED = 0, 1, 2


@dataclass
class Cloud:
    """Per-recording point cloud with provenance and change evidence."""
    xyz: np.ndarray
    rgb: np.ndarray
    frame: np.ndarray  # source frame index
    pix: np.ndarray  # (v, u) in that frame
    n_free: np.ndarray = None
    n_same: np.ndarray = None
    state: np.ndarray = None
    fmask: np.ndarray = None  # after voxel fusion: bitmask of frames observing each voxel
    wmask: np.ndarray = None  # bitmask of OTHER-recording frames that saw through the point (witnesses)
    nrm: np.ndarray = None  # unit surface normal (world), from the depth map it came from


@dataclass
class ChangeCluster:
    kind: str  # "removed" | "added"
    idx: np.ndarray  # indices into the source Cloud
    core: np.ndarray  # indices with direct free-space evidence
    score: float
    n_frames: int = 0


def _normals(P: np.ndarray) -> np.ndarray:
    """Per-pixel surface normals of a back-projected depth map (H, W, 3) from its image-space gradients."""
    du = np.gradient(P, axis=1)
    dv = np.gradient(P, axis=0)
    n = np.cross(du, dv)
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)


def build_cloud(scene: dict, valid: np.ndarray, rec: int, stride: int) -> Cloud:
    xyz, rgb, frame, pix, nrm = [], [], [], [], []
    for i in np.where(scene["rec"] == rec)[0]:
        P, (v, u) = backproject(scene["depth"][i], scene["intrinsic"][i], scene["extrinsic"][i], stride)
        m = valid[i][v, u]
        xyz.append(P[m])
        nrm.append(_normals(P)[m])
        rgb.append(scene["images"][i][v[m], u[m]])
        frame.append(np.full(m.sum(), i))
        pix.append(np.stack([v[m], u[m]], 1))
    c = Cloud(np.concatenate(xyz).astype(np.float32), np.concatenate(rgb), np.concatenate(frame), np.concatenate(pix))
    c.nrm = np.concatenate(nrm).astype(np.float32)
    return c


def evidence(cloud: Cloud, scene: dict, valid: np.ndarray, target_rec: int, tol_rel: float, tol_abs: float, win: int,
             grazing_cos: float = 0.26):  # ~75 degrees from the surface normal
    """Count FREE / SAME votes for every point of `cloud` against all cameras of `target_rec`."""
    n_free = np.zeros(len(cloud.xyz), np.int32)
    n_same = np.zeros(len(cloud.xyz), np.int32)
    wmask = np.zeros(len(cloud.xyz), np.uint64)
    H, W = scene["depth"].shape[1:]
    for j in np.where(scene["rec"] == target_rec)[0]:
        D, ok = scene["depth"][j], valid[j]
        # Neighbourhood min/max make the test robust to small pose/depth misalignment:
        # FREE needs the *whole* window to lie behind p (invalid pixels block the claim).
        d_min_block = minimum_filter(np.where(ok, D, 0.0), size=win)
        d_min = minimum_filter(np.where(ok, D, np.inf), size=win)
        d_max = maximum_filter(np.where(ok, D, -np.inf), size=win)
        uv, z = project(cloud.xyz, scene["intrinsic"][j], scene["extrinsic"][j])
        u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        inside = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if cloud.nrm is not None:
            # A surface the other camera sees nearly edge-on gives unreliable depth (it smears into the
            # background): such a view casts no vote at all.
            ray = cloud.xyz - cam_centers(scene["extrinsic"][j:j + 1])[0]
            cos = np.abs(np.einsum("ij,ij->i", ray, cloud.nrm)) / np.maximum(np.linalg.norm(ray, axis=1), 1e-9)
            inside &= cos > grazing_cos
        ii = np.where(inside)[0]
        u, v, z = u[ii], v[ii], z[ii]
        tol = tol_rel * z + tol_abs
        free = d_min_block[v, u] > z + tol
        center_ok = ok[v, u]
        occl = center_ok & (D[v, u] < z - tol)
        same = center_ok & ~free & ~occl & (d_min[v, u] - tol <= z) & (z <= d_max[v, u] + tol)
        n_free[ii[free]] += 1
        n_same[ii[same]] += 1
        wmask[ii[free]] |= np.uint64(1) << np.uint64(j % 64)
    cloud.n_free, cloud.n_same, cloud.wmask = n_free, n_same, wmask
    return cloud


def fuse_voxels(cloud: Cloud, voxel: float) -> Cloud:
    """Merge all observations of the same ~2 cm cell (from every frame of the recording) and pool their
    votes. Removes multi-view redundancy and makes the decision per *place*, not per pixel."""
    keys = np.floor(cloud.xyz / voxel).astype(np.int64)
    _, first, inv, cnt = np.unique(keys, axis=0, return_index=True, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    n = len(cnt)
    xyz = np.zeros((n, 3))
    np.add.at(xyz, inv, cloud.xyz)
    rgb = np.zeros((n, 3))
    np.add.at(rgb, inv, cloud.rgb.astype(np.float64))
    n_free = np.bincount(inv, cloud.n_free, n)
    n_same = np.bincount(inv, cloud.n_same, n)
    fmask = np.zeros(n, np.uint64)  # which source frames observed this voxel (bitmask, <= 64 frames)
    np.bitwise_or.at(fmask, inv, (np.uint64(1) << (cloud.frame % 64).astype(np.uint64)))
    out = Cloud((xyz / cnt[:, None]).astype(np.float32), (rgb / cnt[:, None]).astype(np.uint8),
                cloud.frame[first], cloud.pix[first], n_free / cnt, n_same / cnt)
    out.fmask = fmask
    if cloud.nrm is not None:
        nr = np.zeros((n, 3))
        np.add.at(nr, inv, cloud.nrm)
        out.nrm = (nr / np.maximum(np.linalg.norm(nr, axis=1, keepdims=True), 1e-9)).astype(np.float32)
    wm = np.zeros(n, np.uint64)
    np.bitwise_or.at(wm, inv, cloud.wmask)
    out.wmask = wm
    seen = out.n_free + out.n_same
    ratio = out.n_free / np.maximum(seen, 1e-9)
    state = np.full(n, UNOBSERVED, np.int8)
    state[(seen > 0) & (ratio < 0.5)] = STATIC
    # changed: free-space evidence from >= 2 views on average, and dominant over agreement
    state[(out.n_free >= 1.5) & (ratio >= 0.6)] = CHANGED
    out.state = state
    return out


def popcount(x: np.ndarray) -> np.ndarray:
    """Number of set bits of each uint64."""
    return np.unpackbits(x.astype(np.uint64).view(np.uint8).reshape(-1, 8), axis=1).sum(1)


def _popcount(x: np.ndarray) -> int:
    v = np.bitwise_or.reduce(x) if len(x) else np.uint64(0)
    return bin(int(v)).count("1")


def _merge_fragments(xyz: np.ndarray, groups: list[np.ndarray], pad: float) -> list[np.ndarray]:
    """One object often breaks into pieces (one face verified from one side, another face from another).
    Merge clusters whose padded bounding boxes overlap - union-find over box intersections."""
    boxes = [(np.percentile(xyz[g], 2, 0) - pad, np.percentile(xyz[g], 98, 0) + pad) for g in groups]
    parent = list(range(len(groups)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if np.all(boxes[i][0] <= boxes[j][1]) and np.all(boxes[j][0] <= boxes[i][1]):
                parent[find(i)] = find(j)
    merged = {}
    for i, g in enumerate(groups):
        merged.setdefault(find(i), []).append(g)
    return [np.concatenate(v) for v in merged.values()]


def cluster_changes(cloud: Cloud, kind: str, eps: float, min_points: int, min_frames: int,
                    min_height: float = 0.0, min_witnesses: int = 4, floor_clear: float = -np.inf) -> list[ChangeCluster]:
    """Group changed voxels into object-level clusters, then grow each over adjacent *unverifiable*
    voxels (object parts the other recording never saw behind) - never over verified-static ones."""
    core = np.where(cloud.state == CHANGED)[0]
    if len(core) < min_points:
        return []
    labels = DBSCAN(eps=eps, min_samples=6).fit_predict(cloud.xyz[core])
    # tiny fragments are noise: they must not merge into (and inflate) real objects
    groups = [core[labels == lab] for lab in set(labels) - {-1} if (labels == lab).sum() >= max(8, min_points // 3)]
    groups = _merge_fragments(cloud.xyz, groups, pad=0.5 * eps)
    cand = np.where(cloud.state == UNOBSERVED)[0]
    out = []
    for c in groups:
        n_frames = _popcount(cloud.fmask[c])
        if len(c) < min_points or n_frames < min_frames:
            continue
        # an object has some thickness: a flat patch on a surface is depth noise, not a thing
        if np.ptp(np.percentile(cloud.xyz[c], [5, 95], axis=0), axis=0).max() < 2 * eps or \
                np.percentile(cloud.xyz[c][:, 1], 95) < min_height:
            continue
        # independent confirmation: the free space must be witnessed by several different cameras of the
        # other recording (edge artefacts are typically seen 'through' by only one or two)
        if cloud.wmask is not None and _popcount(cloud.wmask[c]) < min_witnesses:
            continue
        # an object has volume: a single thin vertical facet (an edge of furniture seen at a grazing
        # angle) is not a thing that appeared or disappeared
        ext = np.ptp(np.percentile(cloud.xyz[c], [5, 95], axis=0), axis=0)
        if min(ext[0], ext[2]) < 0.8 * eps and ext[1] > 2 * min(ext[0], ext[2]):
            continue
        # a 'change' spanning two dimensions of a wall (~1.5 m+) is the two reconstructions disagreeing
        # about where a piece of the room is, not an object that appeared or disappeared
        if np.sort(ext)[-2] > 20 * eps:
            continue
        idx, frontier = c, c
        lo, hi = cloud.xyz[c].min(0) - eps, cloud.xyz[c].max(0) + eps
        lo[1] = max(lo[1], floor_clear)  # grow into object parts, never into the floor around it
        box = cand[np.all((cloud.xyz[cand] >= lo) & (cloud.xyz[cand] <= hi), axis=1)]
        for _ in range(2):  # two hops, bounded to the cluster's (padded) bounding box
            if len(box) == 0:
                break
            d, _ = cKDTree(cloud.xyz[frontier]).query(cloud.xyz[box], distance_upper_bound=eps * 0.6)
            new = box[np.isfinite(d)]
            if len(new) == 0:
                break
            idx = np.concatenate([idx, new])
            box = np.setdiff1d(box, new)
            frontier = new
        ratio = cloud.n_free[c] / np.maximum(cloud.n_free[c] + cloud.n_same[c], 1e-9)
        score = float(np.mean(ratio) * (1 - np.exp(-n_frames / 3)) * (1 - np.exp(-len(c) / (3 * min_points))))
        out.append(ChangeCluster(kind, idx, c, score, n_frames))
    return out


def _sub_scene(scene: dict, frames: np.ndarray, rec: np.ndarray) -> dict:
    keys = ("images", "depth", "conf", "extrinsic", "intrinsic")
    out = {k: scene[k][frames] for k in keys}
    out["rec"] = rec.astype(np.int8)
    return out


def calibrate(scene: dict, valid: np.ndarray, stride: int = 4) -> dict:
    """Measure this run's own depth noise. Within ONE recording nothing changes, so comparing its even
    frames against its odd frames shows how much two views of the same surface disagree. The robust
    spread (MAD) of that relative disagreement sets the tolerance - no hand-tuned threshold."""
    res = []
    for r in (0, 1):
        fr = np.where(scene["rec"] == r)[0]
        src, tgt = fr[0::2], fr[1::2]
        for i in src:
            P, (v, u) = backproject(scene["depth"][i], scene["intrinsic"][i], scene["extrinsic"][i], stride)
            P = P[valid[i][v, u]]
            for j in tgt:
                uv, z = project(P, scene["intrinsic"][j], scene["extrinsic"][j])
                uu, vv = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
                H, W = valid.shape[1:]
                m = (z > 0.05) & (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
                m[m] &= valid[j][vv[m], uu[m]]
                rel = (scene["depth"][j][vv[m], uu[m]] - z[m]) / z[m]
                res.append(rel[np.abs(rel) < 0.15])  # same-surface pairs only (drop occlusion/free)
    res = np.concatenate(res) if res else np.zeros(1)
    sigma = float(1.4826 * np.median(np.abs(res - np.median(res))))
    return dict(sigma=sigma, tol_rel=float(np.clip(3.5 * sigma, 0.015, 0.08)), n=int(len(res)))


def refine_alignment(a: Cloud, b: Cloud, max_dist: float, iters: int = 20):
    """Point-to-plane ICP of the 'after' recording onto the 'before' one, using ONLY surfaces both
    recordings agree are static (moved furniture can't pull the fit). VGGT already puts both videos in
    one frame; this removes the last centimetres of drift between the two halves of the batch.
    Returns (R, t, rmse_before, rmse_after) mapping after-world -> before-world."""
    A = a.xyz[a.state == STATIC].astype(np.float64)
    B = b.xyz[b.state == STATIC].astype(np.float64)
    if len(A) < 500 or len(B) < 500:
        return np.eye(3), np.zeros(3), np.nan, np.nan
    rng = np.random.default_rng(0)
    B = B[rng.choice(len(B), min(len(B), 20000), replace=False)]
    tree = cKDTree(A)
    _, nn = tree.query(A, k=12)  # normals from local PCA
    Q = A[nn] - A[nn].mean(1, keepdims=True)
    normals = np.linalg.eigh(np.einsum("nki,nkj->nij", Q, Q))[1][:, :, 0]
    R, t = np.eye(3), np.zeros(3)

    def residual(R, t):
        X = B @ R.T + t
        d, j = tree.query(X)
        ok = d < max_dist
        return X[ok], j[ok], np.abs(np.einsum("ij,ij->i", X[ok] - A[j[ok]], normals[j[ok]]))

    _, _, r0 = residual(R, t)
    rmse0 = float(np.sqrt(np.mean(r0 ** 2))) if len(r0) else np.nan
    for _ in range(iters):
        X, j, r = residual(R, t)
        if len(X) < 100:
            break
        n, p = normals[j], A[j]
        w = 1.0 / np.maximum(r, np.median(r))  # robust (Huber-like) weights
        J = np.c_[np.cross(X, n), n]
        e = np.einsum("ij,ij->i", p - X, n)
        dx = np.linalg.solve((J * w[:, None]).T @ J + 1e-6 * np.eye(6), (J * w[:, None]).T @ e)
        w_ = dx[:3]
        th = np.linalg.norm(w_)
        K = np.array([[0, -w_[2], w_[1]], [w_[2], 0, -w_[0]], [-w_[1], w_[0], 0]]) / max(th, 1e-12)
        dR = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
        R, t = dR @ R, dR @ t + dx[3:]
        if np.linalg.norm(dx) < 1e-6:
            break
    _, _, r1 = residual(R, t)
    rmse1 = float(np.sqrt(np.mean(r1 ** 2))) if len(r1) else np.nan
    return R, t, rmse0, rmse1


def _apply_to_after(scene: dict, R: np.ndarray, t: np.ndarray):
    """Move the 'after' cameras by the world transform X_before = R X_after + t."""
    m = scene["rec"] == 1
    E = scene["extrinsic"][m].astype(np.float64)
    Re, te = E[:, :, :3], E[:, :, 3]
    scene["extrinsic"][m] = np.concatenate([Re @ R.T, (te - np.einsum("sij,j->si", Re @ R.T, t))[..., None]], 2)


def detect(scene: dict, stride: int = 2, conf_pct: float = 25, tol_rel: float | None = None, win: int = 5,
           min_points: int = 40, min_frames: int = 3, null_test: bool = True, min_confidence: float = 0.65,
           levels=(1.0, 1.5, 2.0, 3.0)):
    """Run bidirectional change detection. Returns (cloud_a, cloud_b, clusters, info).

    Operating point, all measured on this capture (nothing hand-tuned per scene):
      1. tol_rel from the within-video depth noise (even vs odd frames);
      2. tol_abs >= 2.5 x the before/after disagreement on agreed-static surfaces (after ICP);
      3. the smallest multiplier in `levels` for which the null test - the same detector on even vs odd
         frames of each video, where the truth is "nothing changed" - reports ZERO changes.
    """
    conf_thr = np.percentile(scene["conf"], conf_pct)
    valid = np.stack([valid_mask(d, c, conf_thr) for d, c in zip(scene["depth"], scene["conf"])])
    med = float(np.median(scene["depth"][valid]))
    cal = calibrate(scene, valid)
    base_rel = tol_rel or cal["tol_rel"]
    eps = 0.035 * med  # ~7 cm for a typical 2 m viewing distance
    voxel = 0.01 * med  # ~2 cm

    def run(sc, va, t_rel, t_abs):
        args = (t_rel, t_abs, win)
        a = fuse_voxels(evidence(build_cloud(sc, va, 0, stride), sc, va, 1, *args), voxel)
        b = fuse_voxels(evidence(build_cloud(sc, va, 1, stride), sc, va, 0, *args), voxel)
        fc = 2 * voxel  # room frame: floor at y = 0
        cl = cluster_changes(a, "removed", eps, min_points, min_frames, floor_clear=fc) + \
            cluster_changes(b, "added", eps, min_points, min_frames, floor_clear=fc)
        # low-confidence candidates are kept aside (reported as 'uncertain'), never as changes
        return a, b, [c for c in cl if c.score >= min_confidence], [c for c in cl if c.score < min_confidence]

    def null(t_rel, t_abs):
        nt = {}
        for r, name in ((0, "before"), (1, "after")):
            fr = np.where(scene["rec"] == r)[0]
            sub = _sub_scene(scene, fr, np.arange(len(fr)) % 2)
            na, nb, ncl, _ = run(sub, valid[fr], t_rel, t_abs)
            nt[name] = dict(false_changes=len(ncl), flagged_points=float(np.mean(np.r_[na.state, nb.state] == CHANGED)))
        return nt

    tol_abs = 0.01 * med
    a, b, clusters, uncertain = run(scene, valid, base_rel, tol_abs)
    # refine the before/after alignment on agreed-static surfaces, keep it only if it helps
    R, t, rmse0, rmse1 = refine_alignment(a, b, max_dist=3 * base_rel * med)
    icp = dict(rmse_before=rmse0, rmse_after=rmse1, shift=float(np.linalg.norm(t)),
               rotation_deg=float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))), applied=False)
    if np.isfinite(rmse1) and rmse1 < 0.95 * rmse0:
        _apply_to_after(scene, R, t)
        icp["applied"] = True
    resid = rmse1 if icp["applied"] else rmse0
    if np.isfinite(resid):  # the two videos never agree better than this on unchanged surfaces
        tol_abs = max(tol_abs, 2.5 * resid)

    level, nt = 1.0, None
    if null_test:
        # control the false-alarm rate: smallest tolerance level at which the null test is clean
        for level in levels:
            nt = null(base_rel * level, tol_abs * level)
            if sum(v["false_changes"] for v in nt.values()) == 0:
                break
    tol_rel, tol_abs = base_rel * level, tol_abs * level
    a, b, clusters, uncertain = run(scene, valid, tol_rel, tol_abs)
    info = dict(valid=valid, median_depth=med, eps=eps, voxel=voxel, tol_rel=tol_rel, tol_abs=tol_abs,
                depth_noise=cal["sigma"], icp=icp, tolerance_level=level,
                reconstruction_ok=nt is None or sum(v["false_changes"] for v in nt.values()) == 0,
                uncertain=[dict(kind=c.kind, score=round(c.score, 2),
                                position=np.round(np.median((a if c.kind == "removed" else b).xyz[c.idx], 0), 3).tolist())
                           for c in uncertain])
    if nt is not None:
        info["null_test"] = nt
    return a, b, clusters, info
