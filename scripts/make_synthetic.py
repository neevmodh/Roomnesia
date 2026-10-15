"""Render a synthetic before/after room pair with known ground-truth changes.

A tiny numpy ray-caster (textured boxes in a textured room) - no external renderer needed.
Used as a sanity check / quantitative test of the pipeline; real phone videos are the main input.

Scenarios:
  standard  chair moved + turned, bin removed, backpack added
  hard      standard + two removals the after-video can NOT verify (correct answer: don't report):
            a plant in a spot the after-video never films, a crate hidden behind furniture in every
            after-frame. Spots are found automatically by rendering visibility.
  nochange  same room, filmed twice on different paths (correct answer: nothing changed)

    python scripts/make_synthetic.py --scenario standard --out data/synthetic
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOM = np.array([5.0, 2.6, 5.5])  # x, y(up), z in metres
RNG = np.random.default_rng(7)


def texture(h: int, w: int, base, n_shapes: int, seed: int) -> np.ndarray:
    """Procedural feature-rich texture: tinted noise + random shapes + text (posters, patterns...)."""
    r = np.random.default_rng(seed)
    img = np.empty((h, w, 3), np.float32)
    img[:] = base
    noise = cv2.resize(r.normal(0, 1, (h // 16 + 1, w // 16 + 1)).astype(np.float32), (w, h))
    img += 18 * noise[..., None]
    img += 6 * r.normal(0, 1, (h, w, 1)).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    for _ in range(n_shapes):
        c = tuple(int(v) for v in r.integers(0, 255, 3))
        p = (int(r.integers(0, w)), int(r.integers(0, h)))
        k = r.integers(0, 3)
        if k == 0:
            cv2.rectangle(img, p, (p[0] + int(r.integers(10, w // 6)), p[1] + int(r.integers(10, h // 6))), c, -1)
        elif k == 1:
            cv2.circle(img, p, int(r.integers(5, min(h, w) // 10)), c, -1)
        else:
            cv2.putText(img, "".join(r.choice(list("ABCDEFGHJKMNPRSTUVWXYZ0123456789"), 4)), p,
                        cv2.FONT_HERSHEY_SIMPLEX, float(r.uniform(0.6, 2.5)), c, int(r.integers(1, 4)))
    return img


class Box:
    def __init__(self, name, center_xz, size, yaw_deg=0.0, y0=0.0, color=(150, 120, 90), seed=0):
        self.name = name
        self.size = np.array(size, float)  # w(x), h(y), d(z)
        self.center = np.array([center_xz[0], y0 + size[1] / 2, center_xz[1]], float)
        a = np.deg2rad(yaw_deg)
        self.R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        self.tex = texture(256, 256, color, 12, seed)

    def to_json(self):
        return dict(name=self.name, center=self.center.round(3).tolist(), size=self.size.tolist())


def make_scene(variant: str, scenario: str = "standard", extras=()) -> list[Box]:
    table = Box("table", (2.5, 3.0), (1.2, 0.75, 0.7), 5, color=(120, 85, 55), seed=1)
    sofa = Box("sofa", (0.7, 3.5), (0.9, 0.8, 1.9), 0, color=(70, 90, 130), seed=2)
    shelf = Box("shelf", (4.5, 1.2), (0.6, 1.8, 1.0), 0, color=(200, 190, 170), seed=3)
    lamp = Box("lamp", (4.3, 4.6), (0.3, 1.5, 0.3), 0, color=(230, 220, 120), seed=4)
    books = Box("books", (2.6, 2.95), (0.35, 0.18, 0.25), -20, y0=0.75, color=(180, 40, 40), seed=5)
    objs = [table, sofa, shelf, lamp, books]
    if scenario == "hard":  # a tall cabinet that hides the space behind it from the after-video's path
        objs.append(Box("cabinet", (1.6, 2.5), (0.3, 1.6, 1.0), 0, color=(150, 150, 160), seed=14))
    if scenario == "nochange":
        variant = "A"
    if variant == "A":
        objs += list(extras)  # hard scenario: objects only the before-video can see
    if variant == "A":
        objs += [
            Box("chair", (1.6, 1.7), (0.5, 0.9, 0.5), 10, color=(40, 140, 60), seed=6),
            Box("bin", (4.0, 3.3), (0.35, 0.45, 0.35), 0, color=(90, 90, 95), seed=7),
        ]
    else:
        objs += [
            Box("chair", (3.3, 1.9), (0.5, 0.9, 0.5), 45, color=(40, 140, 60), seed=6),  # moved + rotated
            Box("backpack", (1.8, 4.3), (0.4, 0.5, 0.3), -15, color=(220, 110, 30), seed=8),  # added
            # bin removed
        ]
    return objs


ROOM_TEX = [texture(1024, 1024, c, 90, 100 + i) for i, c in enumerate(
    [(205, 200, 190), (190, 205, 200), (200, 190, 205), (210, 205, 185), (150, 115, 80), (240, 240, 235)])]


def look_at(eye, target):
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 1, 0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, -u, f])  # rows: camera x(right), y(down), z(forward) in world coords


def render(objs: list[Box], eye, target, W=960, H=540, fov_deg=62, gain=1.0, ids=False):
    R = look_at(np.asarray(eye, float), np.asarray(target, float))
    f = 0.5 * W / np.tan(np.deg2rad(fov_deg) / 2)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d_cam = np.stack([(u - W / 2) / f, (v - H / 2) / f, np.ones_like(u)], -1).reshape(-1, 3)
    d = d_cam @ R
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    o = np.asarray(eye, float)

    # room: inside of an axis-aligned box -> exit distance
    with np.errstate(divide="ignore", invalid="ignore"):
        t_lo, t_hi = (0 - o) / d, (ROOM - o) / d
    t_exit_axes = np.maximum(t_lo, t_hi)
    axis = np.argmin(t_exit_axes, 1)
    t_best = t_exit_axes[np.arange(len(d)), axis]
    hit_pt = o + t_best[:, None] * d
    side = (d[np.arange(len(d)), axis] > 0).astype(int)  # which wall on that axis
    surf = np.where(axis == 1, 4 + side, axis + side)  # x-walls 0/1, z-walls 2/3, floor 4, ceiling 5
    normal = np.zeros_like(d)
    normal[np.arange(len(d)), axis] = np.where(side == 1, -1.0, 1.0)
    uv = np.stack([np.where(axis == 0, hit_pt[:, 2], hit_pt[:, 0]), np.where(axis == 1, hit_pt[:, 2], hit_pt[:, 1])], 1)

    for k, b in enumerate(objs):
        ol = (o - b.center) @ b.R
        dl = d @ b.R
        half = b.size / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (-half - ol) / dl, (half - ol) / dl
        tn, tf = np.minimum(t1, t2), np.maximum(t1, t2)
        ax = np.argmax(tn, 1)
        t_in = tn[np.arange(len(d)), ax]
        hit = (t_in < tf.min(1)) & (t_in > 1e-4) & (t_in < t_best)
        if not hit.any():
            continue
        t_best[hit] = t_in[hit]
        p = ol + t_in[:, None] * dl
        n_l = np.zeros_like(d)
        n_l[np.arange(len(d)), ax] = -np.sign(dl[np.arange(len(d)), ax])
        normal[hit] = (n_l @ b.R.T)[hit]
        surf[hit] = 10 + k
        # face uv in metres (offset per face so faces look different)
        uu = np.where(ax == 0, p[:, 2], p[:, 0]) + ax * 0.37
        vv = np.where(ax == 1, p[:, 2], p[:, 1]) + ax * 0.21
        uv[hit] = np.stack([uu, vv], 1)[hit]

    color = np.zeros((len(d), 3), np.float32)
    for s in np.unique(surf):
        m = surf == s
        tex, ppm = (ROOM_TEX[s], 180) if s < 10 else (objs[s - 10].tex, 400)
        ix = (uv[m, 0] * ppm).astype(int) % tex.shape[1]
        iy = (uv[m, 1] * ppm).astype(int) % tex.shape[0]
        color[m] = tex[iy, ix]
    light = np.array([0.4, 0.8, 0.3])
    light /= np.linalg.norm(light)
    shade = 0.6 + 0.4 * np.clip(normal @ light, 0, 1)
    img = np.clip(color * shade[:, None] * gain, 0, 255).astype(np.uint8).reshape(H, W, 3)
    depth = (t_best * (d @ R[2])).reshape(H, W)
    if ids:
        return img, depth, surf.reshape(H, W)
    return img, depth


def trajectory(a0, a1, n, radius, height, seed):
    r = np.random.default_rng(seed)
    c = np.array([2.5, 0.0, 2.75])
    poses = []
    for a in np.linspace(np.deg2rad(a0), np.deg2rad(a1), n):
        eye = c + [radius * np.cos(a), height, radius * np.sin(a)] + r.normal(0, 0.02, 3)
        pan = 0.9 * np.sin(np.pi * len(poses) / 12)  # sweep the view left/right like a person filming
        tgt = c + [-0.5 * np.cos(a) - pan * np.sin(a), 0.45, -0.5 * np.sin(a) + pan * np.cos(a)] + r.normal(0, 0.05, 3)
        poses.append((eye, tgt))
    return poses


PATHS = {"A": (200, 340, 2.1, 1.45, 1, 1.0), "B": (250, 420, 1.9, 1.3, 2, 0.92)}  # B: other path, height, exposure
PATHS_HARD = {"A": PATHS["A"], "B": (295, 400, 1.9, 1.3, 2, 0.92)}  # after-video covers only part of the room


def paths_for(scenario: str) -> dict:
    return PATHS_HARD if scenario == "hard" else PATHS


def _visibility(box: Box, variant: str, n: int = 18):
    """(frames where the box is in the field of view, frames where any of it is actually visible)."""
    a0, a1, rad, hgt, seed, _ = PATHS_HARD[variant]
    objs = make_scene("A", "hard") + [box]
    k = len(objs) - 1
    in_view = seen = 0
    for eye, tgt in trajectory(a0, a1, 90, rad, hgt, seed)[:: 90 // n]:
        _, _, ids = render(objs, eye, tgt, W=192, H=108, ids=True)
        R = look_at(np.asarray(eye, float), np.asarray(tgt, float))
        cam = (box.center - eye) @ R.T
        f = 0.5 * 192 / np.tan(np.deg2rad(31))
        u, v = cam[0] / cam[2] * f + 96, cam[1] / cam[2] * f + 54
        in_view += bool(cam[2] > 0.1 and 0 <= u < 192 and 0 <= v < 108)
        seen += bool((ids == 10 + k).sum() > 4)
    return in_view, seen


def find_hidden_spots():
    """Search floor positions for (1) a spot the after-video never has in view, (2) a spot that is in
    the after-video's view but always occluded. Both must be clearly visible in the before-video."""
    unseen = occluded = None
    taken = make_scene("A", "hard") + make_scene("B", "hard")
    for x in np.arange(0.3, 4.75, 0.3):
        for z in np.arange(0.3, 5.25, 0.3):
            b = Box("probe", (x, z), (0.3, 0.3, 0.3), 0, seed=11)
            if any(np.all(np.abs(b.center[[0, 2]] - o.center[[0, 2]]) < (b.size[[0, 2]] + o.size[[0, 2]]) / 2 + 0.15) for o in taken):
                continue
            ia, sa = _visibility(b, "A")
            if sa < 3:
                continue
            ib, sb = _visibility(b, "B")
            if unseen is None and ib == 0 and sb == 0:
                unseen = (x, z, sa)
            if occluded is None and ib >= 3 and sb == 0:
                occluded = (x, z, sa)
            if unseen and occluded:
                return unseen, occluded
    return unseen, occluded


def check_spot(x, z, size=(0.3, 0.3, 0.3)):
    b = Box("probe", (x, z), size, 0, seed=11)
    return _visibility(b, "A"), _visibility(b, "B")


def hard_extras(verify: bool = False):
    """Objects only the before-video has: two the after-video cannot verify, plus a flat rug."""
    # spots found with find_hidden_spots() / check_spot(); re-verified by rendering when verify=True
    plant = Box("plant", (3.9, 1.2), (0.3, 0.3, 0.3), 0, color=(60, 150, 70), seed=12)
    crate = Box("crate", (1.32, 2.6), (0.3, 0.45, 0.3), 0, color=(160, 120, 70), seed=13)
    extras, abstain = [], []
    for b, why in ((plant, "removed where the after-video never looked"),
                   (crate, "removed behind the cabinet in every after-frame")):
        if verify:
            (ia, sa), (ib, sb) = _visibility(b, "A"), _visibility(b, "B")
            print(f"{b.name}: before-video sees it in {sa} sampled frames; after-video has it in view in {ib}, sees it in {sb}")
            assert sa > 0 and sb == 0
        extras.append(b)
        abstain.append(dict(b.to_json(), reason=why))
    # a flat rug (1 cm): below the depth tolerance, only the appearance channel can notice it went
    rug = Box("rug", (2.3, 2.15), (0.7, 0.01, 0.45), 0, color=(170, 30, 60), seed=15)
    extras.append(rug)
    return extras, abstain, rug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--scenario", default="standard", choices=["standard", "hard", "nochange"])
    args = ap.parse_args()
    out = Path(args.out)
    extras, abstain = [], []
    if args.scenario == "hard":
        extras, abstain, rug = hard_extras(verify=True)
    for variant, (a0, a1, rad, hgt, seed, gain) in paths_for(args.scenario).items():
        fdir = out / f"{variant}_frames"
        fdir.mkdir(parents=True, exist_ok=True)
        objs = make_scene(variant, args.scenario, extras)
        for i, (eye, tgt) in enumerate(trajectory(a0, a1, args.frames, rad, hgt, seed)):
            img, _ = render(objs, eye, tgt, gain=gain)
            cv2.imwrite(str(fdir / f"{i:05d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "15", "-i", str(fdir / "%05d.jpg"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out / f"{variant}.mp4")], check=True)
        print(f"wrote {out / f'{variant}.mp4'}")
    A, B = make_scene("A"), make_scene("B")
    gt = {
        "scenario": args.scenario,
        "moved": [{"name": "chair", "from": A[5].to_json()["center"], "to": B[5].to_json()["center"]}],
        "removed": [A[6].to_json()],
        "added": [B[6].to_json()],
        "must_not_report": abstain,
        "appearance": [rug.to_json()] if args.scenario == "hard" else [],
        "paths": paths_for(args.scenario),
        "units": "metres, y up",
    }
    if args.scenario == "nochange":
        gt.update(moved=[], removed=[], added=[])
    if args.scenario == "hard":
        # a change that the relevant video never actually sees cannot be detected: the right answer is to
        # not report it. Re-classify ground truth by rendered visibility instead of assuming.
        for key, var in (("removed", "A"), ("added", "B")):
            keep = []
            for g in gt[key]:
                obj = next(o for o in make_scene(var, "hard") if o.name == g["name"])
                _, seen = _visibility(obj, var)
                if seen == 0:
                    abstain.append(dict(g, reason=f"never visible in video {var}"))
                    print(f"{g['name']}: never visible in video {var} -> must not be reported")
                else:
                    keep.append(g)
            gt[key] = keep
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2))


if __name__ == "__main__":
    main()
