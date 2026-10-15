"""Export everything to a Rerun recording (`scene.rrd`), the 3D/time viewer common in robotics.

    rerun outputs/<run>/scene.rrd

Timeline 'frame' steps through both recordings' cameras (with their photos); the room, the change
boxes/arrows, the robot routes and the next-best views are static entities you can toggle.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .change import STATIC, UNOBSERVED
from .geometry import cam_centers


def _set_frame(rr, k: int):
    try:
        rr.set_time("frame", sequence=k)
    except (AttributeError, TypeError):  # older rerun
        rr.set_time_sequence("frame", k)


def export(path: Path, scene: dict, a, b, changes, routes, views, colors: dict, budget: int = 150_000):
    import rerun as rr

    rr.init("roomnesia")
    rr.save(str(path))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    rng = np.random.default_rng(0)

    def sub(n):
        return rng.choice(n, min(n, budget), replace=False)

    for name, cloud in (("before", a), ("after", b)):
        i = sub(len(cloud.xyz))
        rr.log(f"world/{name}/room", rr.Points3D(cloud.xyz[i], colors=cloud.rgb[i], radii=0.006), static=True)
        un = np.where(cloud.state == UNOBSERVED)[0]
        if len(un):
            un = un[sub(len(un))]
            rr.log(f"world/{name}/unverifiable", rr.Points3D(cloud.xyz[un], colors=[180, 140, 255], radii=0.007), static=True)
    for k, c in enumerate(changes, 1):
        sets = [("", c.points, colors["moved" if c.kind == "moved" else c.kind])]
        if c.kind == "moved":
            sets.append(("_after", c.points_to, colors["moved_to"]))
        for suffix, P, col in sets:
            lo, hi = np.percentile(P, 3, 0), np.percentile(P, 97, 0)
            rr.log(f"world/changes/{k}_{c.kind}{suffix}", rr.Boxes3D(centers=[(lo + hi) / 2], half_sizes=[(hi - lo) / 2],
                   colors=[col], labels=[f"{k}. {c.kind} {c.label}"]), static=True)
            rr.log(f"world/changes/{k}_{c.kind}{suffix}/points", rr.Points3D(P[sub(len(P))[:4000]], colors=col, radii=0.01), static=True)
        if c.kind == "moved":
            p0 = np.array(c.centroid)
            rr.log(f"world/changes/{k}_arrow", rr.Arrows3D(origins=[p0], vectors=[np.array(c.centroid_to) - p0],
                   colors=[[255, 255, 255]]), static=True)
        rr.log(f"world/changes/{k}_{c.kind}", rr.TextDocument(c.sentence), static=True)
    for r in routes:
        for key, col in (("path_before", [140, 140, 150]), ("path_after", [0, 209, 178] if r.status != "blocked" else [230, 57, 70])):
            P = np.array(getattr(r, key))
            if len(P):
                rr.log(f"world/robot/{r.name.replace(' ', '_')}/{key}",
                       rr.LineStrips3D([np.c_[P[:, 0], np.full(len(P), 0.03), P[:, 1]]], colors=[col]), static=True)
    for n, v in enumerate(views, 1):
        yaw = np.deg2rad(v.yaw_deg)
        rr.log(f"world/robot/look_{n}", rr.Arrows3D(origins=[v.position], vectors=[[0.6 * np.sin(yaw), 0, -0.6 * np.cos(yaw)]],
               colors=[[255, 159, 67]], labels=[f"look #{n}"]), static=True)

    H, W = scene["depth"].shape[1:]
    C = cam_centers(scene["extrinsic"])
    for k in range(len(scene["rec"])):
        _set_frame(rr, k)
        name = "before" if scene["rec"][k] == 0 else "after"
        E, K = scene["extrinsic"][k], scene["intrinsic"][k]
        rr.log(f"world/{name}/camera", rr.Transform3D(translation=C[k], mat3x3=E[:, :3].T))
        rr.log(f"world/{name}/camera", rr.Pinhole(image_from_camera=K, resolution=[W, H], camera_xyz=rr.ViewCoordinates.RDF))
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(scene["images"][k], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
        rr.log(f"world/{name}/camera", rr.EncodedImage(contents=jpg.tobytes(), media_type="image/jpeg"))
