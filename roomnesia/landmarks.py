"""Open-vocabulary 3D landmarks: what the *unchanged* room is made of, so changes can be described
relative to it ("from next to the sofa to on the table") and labels get a second, independent opinion.

Grounding DINO boxes on a subset of keyframes -> lifted to 3D with the depth inside each box (the
dominant depth layer, so background behind the object is ignored) -> fused across views by label and
3D overlap. A landmark seen from several views is trusted; a one-off detection is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .geometry import backproject

PROMPT_CLASSES = ["sofa", "chair", "table", "desk", "bed", "shelf", "cabinet", "door", "window", "television",
                  "lamp", "plant", "bin", "box", "backpack", "bag", "suitcase", "bottle", "laptop", "pillow"]
LARGE = {"sofa", "table", "desk", "bed", "shelf", "cabinet", "door", "window", "television"}


@dataclass
class Landmark:
    label: str
    points: np.ndarray
    views: int
    recs: set
    score: float

    @property
    def lo(self):
        return np.percentile(self.points, 5, 0)

    @property
    def hi(self):
        return np.percentile(self.points, 95, 0)

    @property
    def center(self):
        return (self.lo + self.hi) / 2


def _normalise(label: str) -> str | None:
    for w in label.split():
        if w in PROMPT_CLASSES:
            return w
    return None


@torch.no_grad()
def detect_landmarks(scene: dict, per_rec: int = 10, threshold: float = 0.32) -> list[Landmark]:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    name = "IDEA-Research/grounding-dino-tiny"
    proc = AutoProcessor.from_pretrained(name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(name).eval()  # CPU is faster than MPS here
    text = ". ".join(PROMPT_CLASSES) + "."
    H, W = scene["depth"].shape[1:]
    dets = []  # (label, score, points, frame, rec)
    rng = np.random.default_rng(0)
    for r in (0, 1):
        frames = np.where(scene["rec"] == r)[0]
        for i in frames[np.linspace(0, len(frames) - 1, min(per_rec, len(frames))).astype(int)]:
            img = scene["hires"][i]
            inp = proc(images=img, text=text, return_tensors="pt")
            res = proc.post_process_grounded_object_detection(
                model(**inp), inp.input_ids, threshold=threshold, text_threshold=0.25, target_sizes=[img.shape[:2]])[0]
            s = H / img.shape[0]
            P, (v, u) = backproject(scene["depth"][i], scene["intrinsic"][i], scene["extrinsic"][i], 2)
            D = scene["depth"][i][v, u]
            for lab, score, box in zip(res["text_labels"], res["scores"], res["boxes"]):
                lab = _normalise(lab)
                if lab is None:
                    continue
                x0, y0, x1, y1 = (box.numpy() * s).tolist()
                inside = (u >= x0) & (u <= x1) & (v >= y0) & (v <= y1)
                if inside.sum() < 20 or (x1 - x0) * (y1 - y0) > 0.8 * H * W:
                    continue
                # dominant depth layer in the box centre = the object, not the wall behind it
                cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 4, (y1 - y0) / 4
                core = inside & (np.abs(u - cx) <= hw) & (np.abs(v - cy) <= hh)
                d0 = np.median(D[core]) if core.sum() > 5 else np.median(D[inside])
                keep = inside & (np.abs(D - d0) < 0.2 * d0)
                pts = P[keep]
                if len(pts) > 1500:
                    pts = pts[rng.choice(len(pts), 1500, replace=False)]
                if len(pts) >= 20:
                    dets.append((lab, float(score), pts, int(i), r))
    return _fuse(dets)


def _fuse(dets) -> list[Landmark]:
    """Greedy fusion: same label and overlapping 3D boxes -> one landmark."""
    marks: list[Landmark] = []
    for lab, score, pts, frame, r in sorted(dets, key=lambda d: -d[1]):
        lo, hi = np.percentile(pts, 10, 0), np.percentile(pts, 90, 0)
        for m in marks:
            if m.label != lab:
                continue
            mlo, mhi = m.lo, m.hi
            inter = np.clip(np.minimum(hi, mhi) - np.maximum(lo, mlo), 0, None)[[0, 2]].prod()
            small = min(np.prod((hi - lo)[[0, 2]]), np.prod((mhi - mlo)[[0, 2]])) + 1e-6
            if inter / small > 0.3 or np.linalg.norm(((lo + hi) / 2 - m.center)[[0, 2]]) < 0.25:
                m.points = np.concatenate([m.points, pts])[-6000:]
                m.views += 1
                m.recs.add(r)
                m.score = max(m.score, score)
                break
        else:
            marks.append(Landmark(lab, pts, 1, {r}, score))
    return [m for m in marks if m.views >= 2]


# ------------------------------------------------------------------------------------ relations ---

def _footprint_gap(a_lo, a_hi, b_lo, b_hi) -> float:
    d = np.maximum(0, np.maximum(a_lo - b_hi, b_lo - a_hi))[[0, 2]]
    return float(np.hypot(*d))


def place_phrase(P: np.ndarray, marks: list[Landmark], exclude=()) -> str:
    """'on the table', 'on the floor next to the sofa', ... for an object point set P."""
    lo, hi = np.percentile(P, 5, 0), np.percentile(P, 95, 0)
    ctr = (lo + hi) / 2
    static = [m for m in marks if id(m) not in exclude]
    # supported by a landmark: its top is just below our bottom and we sit over its footprint
    for m in sorted(static, key=lambda m: -m.hi[1]):
        if m.label in LARGE | {"chair", "box"} and abs(m.hi[1] - lo[1]) < 0.15 and \
                m.lo[0] - 0.1 <= ctr[0] <= m.hi[0] + 0.1 and m.lo[2] - 0.1 <= ctr[2] <= m.hi[2] + 0.1 and lo[1] > 0.15:
            return f"on the {m.label}"
    base = "on the floor" if lo[1] < 0.15 else ""
    near = sorted(((_footprint_gap(lo, hi, m.lo, m.hi), m) for m in static if m.label in LARGE | {"chair", "lamp", "plant", "bin"}),
                  key=lambda t: t[0])
    if near and near[0][0] < 1.0:
        rel = "next to" if near[0][0] < 0.4 else "near"
        return f"{base + ' ' if base else ''}{rel} the {near[0][1].label}"
    return base


def overlapping(P: np.ndarray, marks: list[Landmark], rec: int) -> Landmark | None:
    """Landmark (from the same recording) that covers the same 3D region as P - a detector's opinion
    of what the changed object is."""
    lo, hi = np.percentile(P, 5, 0), np.percentile(P, 95, 0)
    best, best_iou = None, 0.25
    for m in marks:
        if rec not in m.recs:
            continue
        inter = np.clip(np.minimum(hi, m.hi) - np.maximum(lo, m.lo), 0, None).prod()
        union = np.prod(hi - lo) + np.prod(m.hi - m.lo) - inter
        iou = inter / max(union, 1e-9)
        if iou > best_iou:
            best, best_iou = m, iou
    return best
