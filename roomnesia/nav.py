"""What the changes mean for a robot: map updates, re-planned routes, and where to look next.

1. One consistent world model with two layers. Everything not involved in a change is shared, the removed
   objects live only in the 'before' layer and the added ones only in the 'after' layer, so both maps
   have the same coverage and differ *only* by the detected changes (exactly how a robot would patch
   its map). Exported as ROS map_server files.
2. Routes: the paths people actually walked while filming are known to be traversable. They are
   re-planned (A*, robot-radius inflation) in both layers; a route that gets longer or impossible is
   reported together with the change that caused it.
3. Next-best-view: the 'can't tell' areas are where a change could still hide. Candidate standing spots
   on the free floor are scored by how much unverified surface a camera there would actually see
   (panoramic z-buffer, so occluded surfaces don't count). Greedy selection gives a short list of
   "stand here, look there" actions that turn uncertainty into a plan.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from .change import CHANGED, STATIC, UNOBSERVED, Cloud
from .geometry import cam_centers

FREE, OCC, UNKNOWN = 0, 100, -1


@dataclass
class Grid:
    origin: np.ndarray  # world (x, z) of cell (0, 0) corner
    res: float
    before: np.ndarray  # int8 occupancy, FREE / OCC / UNKNOWN, indexed [row=z, col=x]
    after: np.ndarray

    def cell(self, xz: np.ndarray) -> np.ndarray:
        return np.floor((np.asarray(xz) - self.origin) / self.res).astype(int)[..., ::-1]  # (row, col)

    def world(self, rc: np.ndarray) -> np.ndarray:
        rc = np.asarray(rc)
        return self.origin + (rc[..., ::-1] + 0.5) * self.res


@dataclass
class Route:
    name: str
    start: list
    goal: list
    before_len: float | None
    after_len: float | None
    status: str  # unchanged | detour | blocked | opened
    cause: list = field(default_factory=list)  # change ids
    path_before: list = field(default_factory=list)
    path_after: list = field(default_factory=list)
    sentence: str = ""


@dataclass
class ViewSuggestion:
    position: list  # room-frame x, y, z
    yaw_deg: float
    gain: float  # unverified area it would reveal (m^2, approx.)
    sentence: str = ""


def build_grid(a: Cloud, b: Cloud, changes, scene: dict, res: float = 0.05, body=(0.08, 1.8)) -> Grid:
    in_change_a = np.zeros(len(a.xyz), bool)  # voxels that belong to a change (not part of the shared map)
    in_change_b = np.zeros(len(b.xyz), bool)
    for c in changes:
        if c.kind in ("removed", "moved"):
            _mark(a.xyz, c.points, in_change_a)
        if c.kind == "added":
            _mark(b.xyz, c.points, in_change_b)
        if c.kind == "moved":
            _mark(b.xyz, c.points_to, in_change_b)
    lo_h, hi_h = body if scene.get("metric") else (0.04 * _height_span(a), 0.8 * _height_span(a))
    every = np.concatenate([a.xyz, b.xyz])
    lo = every[:, [0, 2]].min(0) - 0.3
    hi = every[:, [0, 2]].max(0) + 0.3
    shape = tuple((np.ceil((hi - lo) / res).astype(int))[::-1])
    g = Grid(lo, res, None, None)

    def raster(P, val, grid):
        rc = g.cell(P[:, [0, 2]])
        ok = (rc[:, 0] >= 0) & (rc[:, 0] < shape[0]) & (rc[:, 1] >= 0) & (rc[:, 1] < shape[1])
        grid[rc[ok, 0], rc[ok, 1]] = val

    base = np.full(shape, UNKNOWN, np.int8)
    floor = every[every[:, 1] < lo_h]
    raster(floor, FREE, base)
    # camera positions are free space too (someone stood there)
    raster(np.c_[cam_centers(scene["extrinsic"])[:, 0], np.zeros(len(scene["rec"])), cam_centers(scene["extrinsic"])[:, 2]], FREE, base)
    body_pts = lambda P: P[(P[:, 1] >= lo_h) & (P[:, 1] <= hi_h)]
    raster(body_pts(np.concatenate([a.xyz[~in_change_a], b.xyz[~in_change_b]])), OCC, base)
    before, after = base.copy(), base.copy()
    for c in changes:
        if c.kind in ("removed", "moved"):
            raster(body_pts(c.points), OCC, before)
            _clear(c.points, after, g, lo_h, hi_h, base)
        if c.kind in ("added", "moved"):
            P = c.points_to if c.kind == "moved" else c.points
            raster(body_pts(P), OCC, after)
            _clear(P, before, g, lo_h, hi_h, base)
    # small holes in the observed floor are free, not unknown
    for m in (before, after):
        unk = m == UNKNOWN
        closed = ndimage.binary_closing(m == FREE, iterations=2)
        m[unk & closed] = FREE
    g.before, g.after = before, after
    return g


def _height_span(c: Cloud) -> float:
    return float(np.ptp(c.xyz[:, 1]))


def _mark(xyz, P, mask, tol=1e-4):
    from scipy.spatial import cKDTree

    d, i = cKDTree(xyz).query(P, distance_upper_bound=tol)
    mask[i[np.isfinite(d)]] = True


def _clear(P, grid, g: Grid, lo_h, hi_h, base):
    """Footprint of an object that is absent in this layer becomes free floor (where it was occupied)."""
    rc = g.cell(P[(P[:, 1] >= lo_h) & (P[:, 1] <= hi_h)][:, [0, 2]])
    ok = (rc[:, 0] >= 0) & (rc[:, 0] < grid.shape[0]) & (rc[:, 1] >= 0) & (rc[:, 1] < grid.shape[1])
    rc = rc[ok]
    grid[rc[:, 0], rc[:, 1]] = np.where(base[rc[:, 0], rc[:, 1]] == OCC, OCC, FREE)


def astar(occ: np.ndarray, start, goal, inflate_cells: int, unknown_cost: float = 3.0):
    """8-connected A* on an occupancy grid; obstacles inflated by the robot radius, unknown is allowed
    but expensive (a robot may explore, but prefers known floor)."""
    blocked = ndimage.binary_dilation(occ == OCC, iterations=inflate_cells)
    # the start/goal are where a person stood; never let inflation swallow them
    for p in (start, goal):
        blocked[max(0, p[0] - inflate_cells):p[0] + inflate_cells + 1, max(0, p[1] - inflate_cells):p[1] + inflate_cells + 1] &= occ[
            max(0, p[0] - inflate_cells):p[0] + inflate_cells + 1, max(0, p[1] - inflate_cells):p[1] + inflate_cells + 1] == OCC
    cost = np.where(occ == UNKNOWN, unknown_cost, 1.0)
    H, W = occ.shape
    start, goal = tuple(start), tuple(goal)
    if blocked[start] or blocked[goal]:
        return None
    nbrs = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
    h = lambda p: np.hypot(p[0] - goal[0], p[1] - goal[1])
    best = {start: 0.0}
    prev = {}
    pq = [(h(start), 0.0, start)]
    while pq:
        _, g, cur = heapq.heappop(pq)
        if cur == goal:
            path = [cur]
            while cur in prev:
                cur = prev[cur]
                path.append(cur)
            return path[::-1]
        if g > best.get(cur, np.inf):
            continue
        for dr, dc, w in nbrs:
            nr, nc = cur[0] + dr, cur[1] + dc
            if 0 <= nr < H and 0 <= nc < W and not blocked[nr, nc]:
                ng = g + w * cost[nr, nc]
                if ng < best.get((nr, nc), np.inf):
                    best[(nr, nc)] = ng
                    prev[(nr, nc)] = cur
                    heapq.heappush(pq, (ng + h((nr, nc)), ng, (nr, nc)))
    return None


def _length(path, res):
    if not path:
        return None
    p = np.array(path, float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() * res)


def plan_routes(g: Grid, changes, scene: dict, robot_radius: float = 0.25) -> list[Route]:
    inflate = max(1, int(round(robot_radius / g.res)))
    routes = []
    for r, name in ((0, "route walked in A"), (1, "route walked in B")):
        C = cam_centers(scene["extrinsic"][scene["rec"] == r])[:, [0, 2]]
        s, t = g.cell(C[0]), g.cell(C[-1])
        pb, pa = astar(g.before, s, t, inflate), astar(g.after, s, t, inflate)
        lb, la = _length(pb, g.res), _length(pa, g.res)
        if lb is None and la is None:
            status = "unreachable"
        elif la is None:
            status = "blocked"
        elif lb is None:
            status = "opened"
        elif la > lb + 0.25:
            status = "detour"
        elif la < lb - 0.25:
            status = "shorter"
        else:
            status = "unchanged"
        route = Route(name, g.world(s).tolist(), g.world(t).tolist(), lb, la, status,
                      path_before=g.world(np.array(pb)).round(3).tolist() if pb else [],
                      path_after=g.world(np.array(pa)).round(3).tolist() if pa else [])
        if status in ("blocked", "detour") and pb:
            # which change sits on the old path?
            P = np.array(route.path_before)
            for i, c in enumerate(changes, 1):
                if c.kind in ("added", "moved"):
                    Q = (c.points_to if c.kind == "moved" else c.points)[:, [0, 2]]
                    from scipy.spatial import cKDTree

                    if cKDTree(Q).query(P)[0].min() < robot_radius + g.res:
                        route.cause.append(i)
        if status in ("shorter", "opened") and pa:
            P = np.array(route.path_after)
            for i, c in enumerate(changes, 1):
                if c.kind in ("removed", "moved"):
                    from scipy.spatial import cKDTree

                    if cKDTree(c.points[:, [0, 2]]).query(P)[0].min() < robot_radius + g.res:
                        route.cause.append(i)
        route.sentence = _route_sentence(route, changes)
        routes.append(route)
    # per-change navigation impact, taken from the re-planned routes
    for i, c in enumerate(changes, 1):
        notes = []
        for r in routes:
            if i in r.cause:
                notes.append({"blocked": f"blocks the {r.name}",
                              "detour": f"adds {(r.after_len or 0) - (r.before_len or 0):.1f} m to the {r.name}",
                              "shorter": f"shortens the {r.name}", "opened": f"opens the {r.name}"}[r.status])
        c.nav = "; ".join(notes)
    return routes


def _route_sentence(r: Route, changes) -> str:
    who = ", ".join(f"#{i} {changes[i - 1].label}" for i in r.cause)
    by = f" by {who}" if who else ""
    if r.status == "blocked":
        return f"The {r.name} is now BLOCKED{by}; a robot following it would have to stop."
    if r.status == "detour":
        return f"The {r.name} now needs a {r.after_len - r.before_len:.1f} m detour{by} ({r.before_len:.1f} m -> {r.after_len:.1f} m)."
    if r.status == "shorter":
        return f"The {r.name} got {r.before_len - r.after_len:.1f} m shorter{' thanks to ' + who if who else ''}."
    if r.status == "opened":
        return f"The {r.name} is now passable{' thanks to ' + who if who else ''}."
    if r.status == "unchanged":
        return f"The {r.name} is unaffected ({r.after_len:.1f} m)."
    return f"The {r.name} could not be planned in either map."


# ------------------------------------------------------------------------------- next best view ---

def next_best_views(a: Cloud, b: Cloud, g: Grid, scene: dict, changes, k: int = 3, step: float = 0.3,
                    fov_deg: float = 65, max_range: float = 4.5) -> list[ViewSuggestion]:
    """Where should a robot stand, and which way should it look, to check the most 'can't tell' surface?"""
    rng = np.random.default_rng(0)
    metric = bool(scene.get("metric"))
    lo_h, hi_h = (0.05, 2.0) if metric else (0.03 * _height_span(a), 0.9 * _height_span(a))
    every = np.concatenate([a.xyz, b.xyz])
    occl = every[rng.choice(len(every), min(len(every), 60000), replace=False)]
    tgt = np.concatenate([a.xyz[a.state == UNOBSERVED], b.xyz[b.state == UNOBSERVED]])
    tgt = tgt[(tgt[:, 1] > lo_h) & (tgt[:, 1] < hi_h)]
    if len(tgt) < 50:
        return []
    tgt = tgt[rng.choice(len(tgt), min(len(tgt), 8000), replace=False)]
    # weight: unverified surface near a detected change matters more (a change may extend into it)
    w = np.ones(len(tgt))
    for c in changes:
        for P in (c.points, c.points_to):
            if P is not None:
                ctr = np.median(P, 0)
                w += 2.0 * (np.linalg.norm((tgt - ctr)[:, [0, 2]], axis=1) < 1.0)
    cell_area = (0.01 * np.median(scene["depth"])) ** 2  # ~voxel face area, to report m^2
    voxel_scale = len(np.concatenate([a.xyz[a.state == UNOBSERVED], b.xyz[b.state == UNOBSERVED]])) / max(len(tgt), 1)

    # candidate standing spots: known free floor, away from obstacles
    free = (g.after == FREE) & ~ndimage.binary_dilation(g.after == OCC, iterations=max(1, int(0.3 / g.res)))
    stride = max(1, int(step / g.res))
    rr, cc = np.where(free)
    keep = (rr % stride == 0) & (cc % stride == 0)
    cand = g.world(np.stack([rr[keep], cc[keep]], 1))
    if len(cand) == 0:
        return []
    eye_h = float(np.median(cam_centers(scene["extrinsic"])[:, 1]))
    n_az, n_el = 180, 45
    half = int(round(fov_deg / 2 / (360 / n_az)))
    covered = np.zeros(len(tgt), bool)
    out = []
    for _ in range(k):
        best = (0, None, None, None)
        for xz in cand:
            eye = np.array([xz[0], eye_h, xz[1]])
            vis_w, az_t = _panorama_visible(eye, occl, tgt, n_az, n_el, max_range)
            if vis_w is None:
                continue
            gain = np.zeros(n_az)
            np.add.at(gain, az_t[vis_w], (w * ~covered)[vis_w])
            # best yaw = best window of width fov over azimuth bins (circular)
            win = np.convolve(np.r_[gain, gain[: 2 * half]], np.ones(2 * half + 1), "valid")[:n_az]
            j = int(np.argmax(win))
            if win[j] > best[0]:
                center = (j + half) % n_az
                best = (win[j], eye, center, (vis_w & (np.abs(((az_t - center + n_az // 2) % n_az) - n_az // 2) <= half)))
        if best[1] is None or best[0] < 5:
            break
        gain_w, eye, center, newly = best
        covered |= newly
        yaw = (center + 0.5) * 360 / n_az - 180
        area = float(newly.sum() * voxel_scale * cell_area)
        out.append(ViewSuggestion(eye.round(3).tolist(), float(yaw), area))
    for v in out:
        v.sentence = _view_sentence(v, scene, metric)
    return out


def _panorama_visible(eye, occl, tgt, n_az, n_el, max_range):
    """Panoramic z-buffer of the scene around `eye`; which targets are the nearest surface in their bin."""
    def polar(P):
        d = P - eye
        r = np.linalg.norm(d, axis=1)
        az = np.arctan2(d[:, 0], -d[:, 2])  # 0 = towards -z ('forward' on the map)
        el = np.arcsin(np.clip(d[:, 1] / np.maximum(r, 1e-9), -1, 1))
        ai = ((az + np.pi) / (2 * np.pi) * n_az).astype(int) % n_az
        ei = np.clip(((el + np.pi / 2) / np.pi * n_el).astype(int), 0, n_el - 1)
        return r, ai, ei

    ro, ao, eo = polar(occl)
    m = (ro > 0.2) & (ro < max_range)
    zbuf = np.full((n_az, n_el), np.inf)
    np.minimum.at(zbuf, (ao[m], eo[m]), ro[m])
    rt, at, et = polar(tgt)
    vis = (rt > 0.3) & (rt < max_range) & (rt <= zbuf[at, et] * 1.04 + 0.05)
    # cameras pitch roughly level: ignore what is far above/below the view
    vis &= (et > n_el * 0.2) & (et < n_el * 0.65)
    return vis, at


def _view_sentence(v: ViewSuggestion, scene: dict, metric: bool) -> str:
    unit = "m" if metric else " units"
    view0 = scene["view0"]  # rows: first camera right / forward in room (x, z)
    d = np.array(v.position)[[0, 2]]
    lr, fb = view0 @ d
    look = np.array([np.sin(np.deg2rad(v.yaw_deg)), -np.cos(np.deg2rad(v.yaw_deg))])
    llr, lfb = view0 @ look
    facing = ("ahead" if lfb > 0.7 else "back towards the start" if lfb < -0.7 else
              "to the right" if llr > 0 else "to the left")
    return (f"Stand {abs(fb):.1f}{unit} {'ahead of' if fb >= 0 else 'behind'} and {abs(lr):.1f}{unit} "
            f"{'right' if lr >= 0 else 'left'} of where recording A started, facing {facing}: "
            f"this would check ~{v.gain:.1f} m² of surface neither video could verify.")


# ------------------------------------------------------------------------------------- export ---

def write_ros_maps(g: Grid, out: Path):
    """ROS map_server format (.pgm + .yaml). Map x = room x, map y = -room z (top view, y up)."""
    out.mkdir(parents=True, exist_ok=True)
    for name, m in (("before", g.before), ("after", g.after)):
        img = np.full(m.shape, 205, np.uint8)  # unknown
        img[m == FREE] = 254
        img[m == OCC] = 0
        # grid row 0 is min z = max map-y, which is exactly the image top row map_server expects
        with open(out / f"{name}.pgm", "wb") as f:
            f.write(f"P5\n{img.shape[1]} {img.shape[0]}\n255\n".encode())
            f.write(img.tobytes())
        # map_server: image row 0 is the TOP (max y). Our row index grows with z, i.e. y = -z decreases,
        # so row 0 (min z) is max y: origin (bottom-left) is at (x_min, -z_max).
        oy = -(g.origin[1] + m.shape[0] * g.res)
        (out / f"{name}.yaml").write_text(
            f"image: {name}.pgm\nresolution: {g.res}\norigin: [{g.origin[0]:.3f}, {oy:.3f}, 0.0]\n"
            "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")
    change = np.zeros(g.before.shape, np.uint8)
    change[(g.before == OCC) & (g.after != OCC)] = 1  # freed
    change[(g.after == OCC) & (g.before != OCC)] = 2  # newly blocked
    np.save(out / "change_layer.npy", change)
