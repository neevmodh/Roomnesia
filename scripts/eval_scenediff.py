"""Evaluate Roomnesia on real video pairs from the SceneDiff benchmark (Wu et al., 2025; MIT licence).

SceneDiff annotates every object that was added, removed or moved between two videos with a mask in
every frame. Our output is 3D; to compare, each detected change is projected into the same video frames
(only where it is the visible surface) and overlapped with the annotated masks.

Protocol (object level, our own - SceneDiff's paper reports mask AP, which needs per-pixel scores):
  * a GT 'removed' object (only in video 1) is found if a detected removal (or the old position of a
    detected move) overlaps its mask in some video-1 keyframe with IoU >= 0.2;
  * a GT 'added' object (only in video 2): same with additions / new positions, in video 2;
  * a GT 'moved' object (in both videos): found if either its old or its new position is matched;
    'type correct' if the detection was a move;
  * a detection that matches no GT object is a false positive.

    python scripts/eval_scenediff.py data/scenediff/pairs/<pair> outputs/<run>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import safe_pickle  # noqa: E402

from roomnesia import frames  # noqa: E402
from roomnesia.geometry import project  # noqa: E402
from roomnesia.reconstruct import fit_to  # noqa: E402

IOU = 0.2
SAMPLE_FPS = 6.0


def rle_decode(rle: dict) -> np.ndarray:
    """COCO compressed RLE (string counts) -> binary mask, without pycocotools."""
    h, w = rle["size"]
    s, cnts, p = rle["counts"], [], 0
    while p < len(s):
        x, k, more = 0, 0, True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    flat = np.zeros(h * w, np.uint8)
    pos, val = 0, 0
    for c in cnts:
        flat[pos:pos + c] = val
        pos += c
        val ^= 1
    return flat.reshape(w, h).T.astype(bool)  # column-major


def gt_objects(seg: dict):
    out = []
    for o in seg["objects"]:
        kind = "moved" if o["in_video1"] and o["in_video2"] else "removed" if o["in_video1"] else "added"
        out.append(dict(label=o["label"], kind=kind, idx=o["original_obj_idx"]))
    return out


def pred_mask(P, K, E, D, H, W):
    uv, z = project(P, K, E)
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ok = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    ok[ok] &= z[ok] <= D[v[ok], u[ok]] * 1.08 + 0.03  # only where it is the visible surface
    m = np.zeros((H, W), np.uint8)
    m[v[ok], u[ok]] = 1
    m = cv2.morphologyEx(cv2.dilate(m, np.ones((5, 5), np.uint8)), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return m.astype(bool)


def evaluate(pair: Path, run: Path) -> dict:
    seg = safe_pickle.load(pair / "segments.pkl")
    ev = np.load(run / "eval_data.npz", allow_pickle=False)
    E, K, D, rec = ev["extrinsic"], ev["intrinsic"], ev["depth"].astype(np.float32), ev["rec"]
    H, W = D.shape[1:]
    kinds = list(ev["kinds"])
    kf = {}
    for r, key, vid in ((0, "kf_a", "video1.mp4"), (1, "kf_b", "video2.mp4")):
        if key in ev.files:
            kf[r] = ev[key]
        else:  # older runs: recompute (keyframe selection is deterministic)
            kf[r] = np.array(frames.load_recording(pair / vid, int((rec == r).sum()))[1])
    fps = {}
    for r, vid in ((0, "video1.mp4"), (1, "video2.mp4")):
        cap = cv2.VideoCapture(str(pair / vid))
        fps[r] = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

    gts = gt_objects(seg)
    # pred point sets per side: side 0 = video-1 state, side 1 = video-2 state
    preds = []
    for i, k in enumerate(kinds):
        if k == "appearance":
            preds.append(dict(i=i, kind=k, side={0: ev[f"p{i}_0"], 1: ev[f"p{i}_0"]}))
        elif k == "removed":
            preds.append(dict(i=i, kind=k, side={0: ev[f"p{i}_0"]}))
        elif k == "added":
            preds.append(dict(i=i, kind=k, side={1: ev[f"p{i}_0"]}))
        else:  # moved
            preds.append(dict(i=i, kind=k, side={0: ev[f"p{i}_0"], 1: ev[f"p{i}_1"]}))

    # best IoU between each (gt, pred) on each side, over that video's keyframes
    best = np.zeros((len(gts), len(preds), 2))
    for r, key in ((0, "video1_objects"), (1, "video2_objects")):
        frames_r = np.where(rec == r)[0]
        for local, j in enumerate(frames_r):
            raw = str(int(round(kf[r][local] * fps[r] / SAMPLE_FPS)))
            pm = {p["i"]: pred_mask(p["side"][r], K[j], E[j], D[j], H, W) for p in preds if r in p["side"]}
            for gi, g in enumerate(gts):
                rle = seg[key].get(g["idx"], {}).get(raw)
                if rle is None:
                    continue
                gm = fit_to(rle_decode(rle).astype(np.uint8) * 255, H, W) > 127
                if gm.sum() < 30:
                    continue
                for pi, p in enumerate(preds):
                    if p["i"] in pm:
                        inter = (gm & pm[p["i"]]).sum()
                        if inter:
                            best[gi, pi, r] = max(best[gi, pi, r], inter / (gm | pm[p["i"]]).sum())

    compatible = {"removed": {"removed", "moved", "appearance"}, "added": {"added", "moved", "appearance"},
                  "moved": {"removed", "added", "moved", "appearance"}}
    rows, used = [], set()
    for gi, g in enumerate(gts):
        sides = [0] if g["kind"] == "removed" else [1] if g["kind"] == "added" else [0, 1]
        cand = [(best[gi, pi, sides].max(), pi) for pi, p in enumerate(preds) if p["kind"] in compatible[g["kind"]]]
        cand = [c for c in cand if c[0] >= IOU]
        hit = max(cand) if cand else None
        if hit:
            used.add(hit[1])
        rows.append(dict(label=g["label"], kind=g["kind"], found=bool(hit),
                         iou=round(float(hit[0]), 2) if hit else None,
                         predicted_as=preds[hit[1]]["kind"] if hit else None,
                         type_correct=bool(hit) and preds[hit[1]]["kind"] == g["kind"]))
    # a prediction overlapping ANY annotated changed object (even an already-matched one) is not a false alarm
    fp = [p["kind"] for pi, p in enumerate(preds) if pi not in used and best[:, pi, :].max(initial=0) < IOU]
    fp_ids = [p["i"] for pi, p in enumerate(preds) if pi not in used and best[:, pi, :].max(initial=0) < IOU]
    return dict(pair=pair.name, gt=len(gts), found=sum(r["found"] for r in rows),
                type_correct=sum(r["type_correct"] for r in rows), predictions=len(preds), false_positives=len(fp),
                false_positive_kinds=fp, false_positive_ids=fp_ids, objects=rows)


if __name__ == "__main__":
    res = evaluate(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps(res, indent=2))
    (Path(sys.argv[2]) / "scenediff_eval.json").write_text(json.dumps(res, indent=2))
