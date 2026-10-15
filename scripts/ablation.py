"""Ablation: why visibility reasoning matters.

Naive baseline = what most people try first: a point is 'removed' if the other recording has no point
within a few cm of it (and vice versa for 'added'), then cluster. Same geometry, same clustering, same
filters as roomnesia - only the change test differs.

    python scripts/ablation.py outputs/synthetic
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomnesia import change, geometry  # noqa: E402


def main(run: Path):
    info = json.loads((run / "run_info.json").read_text())
    scene = geometry.to_room_frame(dict(np.load(run / "scene.npz")), info.get("metric_scale"))
    a, b, clusters, det = change.detect(scene, null_test=False, min_confidence=0.0)
    eps, voxel = det["eps"], det["voxel"]
    rows = [("roomnesia (free-space evidence)", len(clusters))]
    for radius in (0.05, 0.10):
        na, nb = _copy(a), _copy(b)
        na.state = np.where(cKDTree(b.xyz).query(a.xyz)[0] > radius, change.CHANGED, change.STATIC).astype(np.int8)
        nb.state = np.where(cKDTree(a.xyz).query(b.xyz)[0] > radius, change.CHANGED, change.STATIC).astype(np.int8)
        na.n_free = nb.n_free = None
        n = len(_clusters(na, "removed", eps)) + len(_clusters(nb, "added", eps))
        rows.append((f"naive nearest-point diff (> {radius * 100:.0f} cm)", n))
    print(f"{'method':42s} change clusters reported")
    for name, n in rows:
        print(f"{name:42s} {n}")
    (run / "ablation.json").write_text(json.dumps(dict(rows), indent=2))


def _copy(c):
    return change.Cloud(c.xyz, c.rgb, c.frame, c.pix, c.n_free, c.n_same, c.state, c.fmask)


def _clusters(cloud, kind, eps):
    # naive clouds have no vote counts: give every changed voxel 'certain' votes so the scoring works
    cloud.n_free = (cloud.state == change.CHANGED).astype(float) * 3
    cloud.n_same = (cloud.state == change.STATIC).astype(float)
    return change.cluster_changes(cloud, kind, eps, 40, 3)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
