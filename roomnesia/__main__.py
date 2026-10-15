"""roomnesia: what changed in the room?

    python -m roomnesia BEFORE.mp4 AFTER.mp4 --out outputs/my_room
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import appearance, change, evidence, frames, geometry, landmarks, nav, reconstruct, semantics, viz


def main(argv=None):
    ap = argparse.ArgumentParser(description="Detect moved / added / removed objects between two room recordings.")
    ap.add_argument("before", type=Path, help="video (or folder of images) of the room before")
    ap.add_argument("after", type=Path, help="video (or folder of images) of the room after")
    ap.add_argument("--out", type=Path, default=Path("outputs/run"))
    ap.add_argument("--frames", type=int, default=24, help="keyframes per recording (memory scales with total)")
    ap.add_argument("--no-metric", action="store_true", help="skip metric scale estimation (report scene units)")
    ap.add_argument("--camera-height", type=float, default=None,
                    help="typical phone height above the floor while filming, in metres (e.g. 1.4); "
                         "sets the metric scale from the floor instead of a monocular depth network")
    ap.add_argument("--no-vlm", action="store_true", help="skip the local VLM captioner (SigLIP labels only)")
    ap.add_argument("--force", action="store_true", help="recompute the reconstruction even if cached")
    ap.add_argument("--gif", action="store_true", help="also render a rotating 3D GIF (slower)")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    T = time.time()

    cache = args.out / "scene.npz"
    if cache.exists() and not args.force:
        print(f"[1/6] loading cached reconstruction {cache}")
        scene = dict(np.load(cache))
    else:
        print("[1/6] selecting keyframes")
        fa, ia = frames.load_recording(args.before, args.frames)
        fb, ib = frames.load_recording(args.after, args.frames)
        print(f"      before: {len(fa)} keyframes, after: {len(fb)} keyframes")
        print("[2/6] joint 3D reconstruction (VGGT)")
        scene = reconstruct.reconstruct(fa, fb)
        np.savez_compressed(cache, **scene)

    scale, scale_src = None, "none (scene units)"
    if args.camera_height:
        # the floor is known after the room-frame fit: scale = stated phone height / median camera height
        h = np.median(geometry.cam_centers(geometry.to_room_frame(scene, None)["extrinsic"])[:, 1])
        scale, scale_src = args.camera_height / h, f"camera height {args.camera_height} m"
    elif not args.no_metric:
        scale = reconstruct.metric_scale(scene["images"], scene["depth"], scene["conf"])
        scale_src = "Depth Anything V2 (metric, indoor)" if scale else scale_src
    if scale:
        print(f"      metric scale: {scale:.3f} m per unit (from {scale_src})")
    scene = geometry.to_room_frame(scene, scale)

    print("[3/6] visibility-aware change detection")
    a, b, clusters, info = change.detect(scene)
    for name, c in (("before", a), ("after", b)):
        n = len(c.state)
        print(f"      {name}: {n} pts | static {np.mean(c.state == 0):.0%} changed {np.mean(c.state == 1):.1%} "
              f"unverifiable {np.mean(c.state == 2):.0%}")
    print(f"      depth noise {info['depth_noise']:.1%} -> tolerance {info['tol_rel']:.1%} of depth | "
          f"{len(clusters)} confident change clusters, {len(info['uncertain'])} uncertain (not reported)")
    icp = info["icp"]
    print(f"      before/after alignment refined on static surfaces: rmse {icp['rmse_before'] * 100:.1f} -> "
          f"{icp['rmse_after'] * 100:.1f} cm ({'applied' if icp['applied'] else 'not needed'})")
    nt = info.get("null_test", {})
    print("      null test (even vs odd frames of one video, truth = no change): "
          + ", ".join(f"{k} {v['false_changes']} false changes" for k, v in nt.items()))

    excl = [a.xyz[c.idx] for c in clusters if c.kind == "removed"] + [b.xyz[c.idx] for c in clusters if c.kind == "added"]
    app, app_stats = appearance.appearance_changes(scene, a, info["valid"], info["tol_rel"], info["eps"], excl)
    print(f"      appearance channel: {app_stats.get('compared', 0)} static voxels compared after colour correction, "
          f"{len(app)} flat change(s)")
    clusters = clusters + app

    print("[4/6] semantics: labels, move matching, descriptions")
    marks = landmarks.detect_landmarks(scene)
    print(f"      {len(marks)} landmarks: " + ", ".join(sorted(m.label for m in marks)))
    changes = semantics.build_changes(scene, a, b, clusters, use_vlm=not args.no_vlm, marks=marks)
    for i, c in enumerate(changes, 1):
        print(f"      {i}. [{c.kind}] {c.sentence}" + (f"  -- {c.nav}" if c.nav else ""))

    print("[5/6] robot view: map update, re-planned routes, where to look next")
    grid = nav.build_grid(a, b, changes, scene)
    routes = nav.plan_routes(grid, changes, scene)
    views = nav.next_best_views(a, b, grid, scene, changes)
    for r in routes:
        print(f"      {r.sentence}")
    for n, v in enumerate(views, 1):
        print(f"      look #{n}: {v.sentence}")

    print("[6/6] writing outputs (same-viewpoint evidence, maps, viewer)")
    for c in changes:
        c.evidence = evidence.change_evidence(scene, c, viz.COLORS)
    evidence.evidence_gif(scene, changes, viz.COLORS, args.out / "evidence.gif")
    nav.write_ros_maps(grid, args.out / "maps")
    viz.navigation_figure(grid, routes, views, changes, args.out / "navigation.png")
    viz.write_json(changes, scene, args.out / "changes.json", routes, views, info)
    cards = viz.write_cards(changes, scene, args.out)
    viz.topdown(changes, a, b, scene, args.out / "topdown.png")
    viz.summary_image(cards, args.out / "topdown.png", args.out / "summary.jpg")
    viz.viewer_html(changes, a, b, scene, args.out / "viewer.html", routes=routes, views=views)
    try:
        from . import rerun_export

        rerun_export.export(args.out / "scene.rrd", scene, a, b, changes, routes, views, viz.COLORS)
    except Exception as e:  # optional output; never fail the run for it
        print(f"      (rerun export skipped: {e})")
    if args.gif:
        viz.orbit_gif(changes, a, b, args.out / "orbit.gif")
    (args.out / "run_info.json").write_text(json.dumps(dict(
        frames=int(len(scene["rec"])), image_size=list(scene["depth"].shape[1:]), metric_scale=scale, scale_source=scale_src,
        median_depth=info["median_depth"], depth_noise=info["depth_noise"], tolerance=info["tol_rel"],
        null_test=info.get("null_test"), alignment_refinement=info["icp"], seconds=round(time.time() - T, 1)), indent=2))
    print(f"done in {time.time() - T:.0f}s -> {args.out}/ (open viewer.html)")


if __name__ == "__main__":
    main()
