"""Evaluate change detection on the synthetic room against ground truth.

    python scripts/eval_synthetic.py --oracle         # perfect depth + poses: tests the change logic alone
    python scripts/eval_synthetic.py --run outputs/synthetic   # score a full pipeline run (VGGT geometry)

A detected change counts as correct if it has the right type and its centre lies within
0.4 m (floor-plane distance) of the ground-truth object. Pipeline runs are scored after aligning the
reconstruction's room frame to the synthetic world with the camera centres (Umeyama, with scale).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_synthetic as syn  # noqa: E402

from roomnesia import change, geometry, semantics, viz  # noqa: E402


def gt_cameras(n=24, scenario="standard"):
    cams = {}
    for variant, (a0, a1, rad, hgt, seed, gain) in syn.paths_for(scenario).items():
        poses = syn.trajectory(a0, a1, 90, rad, hgt, seed)
        idx = np.linspace(0, 89, n).astype(int)
        cams[variant] = ([poses[i] for i in idx], gain)
    return cams


def oracle_scene(n=24, W=518, H=294, fov=62, scenario="standard"):
    imgs, hires, depth, E, K, rec = [], [], [], [], [], []
    f = 0.5 * W / np.tan(np.deg2rad(fov) / 2)
    extras = syn.hard_extras()[0] if scenario == "hard" else ()
    for r, (variant, (poses, gain)) in enumerate(gt_cameras(n, scenario).items()):
        objs = syn.make_scene(variant, scenario, extras)
        for eye, tgt in poses:
            img, d = syn.render(objs, eye, tgt, W=2 * W, H=2 * H, fov_deg=fov, gain=gain)
            hires.append(img)
            imgs.append(img[::2, ::2])
            depth.append(d[::2, ::2])
            R = syn.look_at(np.asarray(eye, float), np.asarray(tgt, float))
            E.append(np.concatenate([R, (-R @ eye)[:, None]], 1))
            K.append(np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]]))
            rec.append(r)
    return dict(images=np.stack(imgs), hires=np.stack(hires), depth=np.stack(depth).astype(np.float32),
                conf=np.full((len(imgs), H, W), 5.0, np.float32), extrinsic=np.stack(E).astype(np.float32),
                intrinsic=np.stack(K).astype(np.float32), rec=np.array(rec, np.int8))


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    Sd = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Sd[2, 2] = -1
    R = U @ Sd @ Vt
    s = np.trace(np.diag(sig) @ Sd) / S.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def score(changes: list[dict], to_world, gt: dict) -> dict:
    targets = [("moved", g["name"], np.array(g["to"])) for g in gt["moved"]] + \
              [("removed", g["name"], np.array(g["center"])) for g in gt["removed"]] + \
              [("added", g["name"], np.array(g["center"])) for g in gt["added"]] + \
              [("appearance", g["name"], np.array(g["center"])) for g in gt.get("appearance", [])]
    hit, used = [], set()
    for kind, name, pos in targets:
        best = None
        for i, c in enumerate(changes):
            p = to_world(np.array(c.get("position_after") or c["position"]))
            d = float(np.linalg.norm((p - pos)[[0, 2]]))
            if i not in used and c["type"] == kind and d < 0.4 and (best is None or d < best[1]):
                best = (i, d)
        if best:
            used.add(best[0])
        hit.append(dict(gt=f"{kind} {name}", found=best is not None, error_m=round(best[1], 3) if best else None))
    # abstentions: changes the after-video could not see must NOT be reported
    abstain = []
    for g in gt.get("must_not_report", []):
        pos = np.array(g["center"])
        bad = [i for i, c in enumerate(changes)
               if np.linalg.norm((to_world(np.array(c["position"])) - pos)[[0, 2]]) < 0.4]
        abstain.append(dict(object=g["name"], reason=g["reason"], correctly_not_reported=not bad))
        used.update(bad)
    tp = sum(h["found"] for h in hit)
    return dict(recall=tp / len(targets) if targets else None, precision=tp / max(len(changes), 1) if changes else None,
                matches=hit, abstentions=abstain, false_positives=len(changes) - len(used) - 0,
                reported=len(changes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--run", type=Path)
    ap.add_argument("--gt", type=Path, default=Path("data/synthetic/ground_truth.json"))
    ap.add_argument("--out", type=Path, default=Path("outputs/synthetic_oracle"))
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--frames", type=int, default=24, help="keyframes per video used by the run")
    args = ap.parse_args()
    gt = json.loads(args.gt.read_text())

    if args.oracle:
        args.out.mkdir(parents=True, exist_ok=True)
        scene = oracle_scene()
        scene = geometry.to_room_frame(scene, 1.0)  # GT is metric already
        a, b, clusters, info = change.detect(scene)
        changes = semantics.build_changes(scene, a, b, clusters, use_vlm=not args.no_vlm)
        viz.write_json(changes, scene, args.out / "changes.json")
        cards = viz.write_cards(changes, scene, args.out)
        viz.topdown(changes, a, b, scene, args.out / "topdown.png")
        viz.summary_image(cards, args.out / "topdown.png", args.out / "summary.jpg")
        viz.viewer_html(changes, a, b, scene, args.out / "viewer.html")
        run = args.out
        # room frame -> synthetic world: recover from the camera centres
        Cw = np.concatenate([[p[0] for p in gt_cameras()[v][0]] for v in "AB"])
    else:
        run = args.run
        # the pipeline picks the sharpest frame per time bin: recover exactly which frames it used
        from roomnesia import frames
        paths = gt.get("paths", syn.PATHS)
        Cw = []
        for v in "AB":
            _, idx = frames.load_recording(args.gt.parent / f"{v}.mp4", args.frames)
            a0, a1, rad, hgt, seed, _ = paths[v]
            poses = syn.trajectory(a0, a1, 90, rad, hgt, seed)
            Cw += [poses[min(89, round(i * 15 / 6))][0] for i in idx]  # video is 15 fps, sampled at 6 fps
        Cw = np.array(Cw)

    scene_npz = np.load(run / "scene.npz") if (run / "scene.npz").exists() else None
    res = json.loads((run / "changes.json").read_text())
    if scene_npz is not None:
        # the pipeline's room frame is metric only approximately: align its camera centres to GT
        info = json.loads((run / "run_info.json").read_text())
        sc = dict(scene_npz)
        sc = geometry.to_room_frame(sc, info.get("metric_scale"))
        Cr = geometry.cam_centers(sc["extrinsic"])
        s, R, t = umeyama(Cr, Cw)
        print(f"alignment: scale {s:.3f} (1.0 = metric estimate was exact)")
        to_world = lambda p: s * R @ p + t
    else:
        Cr = geometry.cam_centers(geometry.to_room_frame(oracle_scene(), 1.0)["extrinsic"])
        s, R, t = umeyama(Cr, Cw)
        to_world = lambda p: s * R @ p + t
    report = score(res["changes"], to_world, gt)
    print(json.dumps(report, indent=2))
    (run / "eval.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
