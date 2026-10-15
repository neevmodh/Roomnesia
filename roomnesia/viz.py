"""Outputs: changes.json, before/after cards, a top-down 'navigation change map', and a 3D web viewer."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np

from .change import STATIC, Cloud
from .geometry import cam_centers
from .semantics import Change, View, crop, draw_box

COLORS = {"removed": (230, 57, 70), "added": (42, 157, 92), "moved": (240, 140, 0), "moved_to": (40, 110, 220),
          "appearance": (155, 89, 182)}


def _hex(c):
    return "#%02x%02x%02x" % c


# ------------------------------------------------------------------------------------------ json ---

def write_json(changes: list[Change], scene: dict, path: Path, routes=(), views=(), info=None):
    unit = "m" if scene.get("metric") else "scene units"
    out = {
        "units": unit,
        "frame": "room frame: y up, floor at y=0, x/z aligned to the walls (right-handed), origin on the floor under the first camera of recording A",
        "changes": [
            {k: v for k, v in dict(
                id=i + 1, type=c.kind, label=c.label, label_confidence=round(c.label_conf, 3), label_source=c.label_source,
                place=c.place_from or None, place_after=c.place_to or None,
                description=c.phrase, sentence=c.sentence, confidence=round(c.score, 3),
                position=np.round(c.centroid, 3).tolist(),
                position_after=np.round(c.centroid_to, 3).tolist() if c.centroid_to else None,
                size_xyz=np.round(c.size, 3).tolist(), bottom_height=round(c.height, 3),
                displacement=round(c.displacement, 3) if c.kind == "moved" else None,
                rotation_deg=round(c.yaw_deg, 1) if c.kind == "moved" else None,
                navigation=c.nav or None,
            ).items() if v is not None}
            for i, c in enumerate(changes)
        ],
        "routes": [dict(name=r.name, status=r.status, sentence=r.sentence, caused_by=r.cause,
                        length_before=None if r.before_len is None else round(r.before_len, 2),
                        length_after=None if r.after_len is None else round(r.after_len, 2)) for r in routes],
        "next_best_views": [dict(position=v.position, yaw_deg=round(v.yaw_deg, 1), unverified_area_m2=round(v.gain, 2),
                                 sentence=v.sentence) for v in views],
    }
    if info:
        out["self_check"] = dict(depth_noise=round(info["depth_noise"], 4), tolerance=round(info["tol_rel"], 4),
                                 null_test=info.get("null_test"), alignment_refinement=info.get("icp"))
        out["uncertain_candidates"] = info.get("uncertain", [])
    path.write_text(json.dumps(out, indent=2))


# ----------------------------------------------------------------------------------------- cards ---

def _fit(img, h):
    return cv2.resize(img, (max(1, int(img.shape[1] * h / img.shape[0])), h), interpolation=cv2.INTER_AREA)


def _put(img, text, org, scale, color, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def change_card(c: Change, i: int, scene: dict, h: int = 240) -> np.ndarray:
    """BEFORE photo (video A) + for each spot: EXPECTED (memory of A rendered from a B camera) next to
    ACTUAL (that same B camera). Same viewpoint, so the difference *is* the change. Sentence underneath."""
    tiles = []
    v = c.views.get("before")
    if v is not None:
        col = COLORS["moved" if c.kind == "moved" else c.kind]
        boxed = {"hires": draw_box(scene, v, col)[None], "images": scene["images"][:1]}
        t = _fit(crop(boxed, View(0, v.box, v.score), pad=0.6), h)
        _put(t, "BEFORE (video A)", (8, 24), 0.6, (30, 30, 30), 2)
        tiles.append(t)
    for name, exp, act in getattr(c, "evidence", []) or []:
        for img, cap in ((exp, f"EXPECTED {name}"), (act, f"ACTUAL {name}")):
            t = _fit(img, h)
            _put(t, cap, (8, 24), 0.6, (30, 30, 30), 2)
            tiles.append(t)
    if not tiles:
        tiles = [np.full((h, h, 3), 235, np.uint8)]
    gap = np.full((h, 6, 3), 255, np.uint8)
    row = tiles[0]
    for t in tiles[1:]:
        row = np.hstack([row, gap, t])
    W = max(row.shape[1], 640)
    row = cv2.copyMakeBorder(row, 0, 0, 0, W - row.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
    lines = _wrap(c.sentence.replace("°", " deg"), W - 20) + ([f"robot: {c.nav}"] if c.nav else [])
    text = np.full((46 + 21 * len(lines), W, 3), 255, np.uint8)
    col = COLORS[c.kind]
    cv2.rectangle(text, (0, 0), (W, 6), col, -1)
    _put(text, f"#{i} {c.kind.upper()}  {c.label}  (confidence {c.score:.2f})", (10, 32), 0.65, col, 2)
    for k, line in enumerate(lines):
        _put(text, line, (10, 56 + 21 * k), 0.5, (40, 40, 40))
    return np.vstack([row, text])


def _wrap(sentence: str, width: int) -> list[str]:
    lines, line = [], ""  # Hershey fonts are ASCII-only, hence the caller's deg substitution
    for w in sentence.split():
        if cv2.getTextSize(line + " " + w, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] > width:
            lines.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    return lines + [line]


def write_cards(changes: list[Change], scene: dict, out: Path) -> list[np.ndarray]:
    (out / "cards").mkdir(exist_ok=True)
    for old in (out / "cards").glob("*.jpg"):  # cards from a previous run would be misleading
        old.unlink()
    cards = []
    for i, c in enumerate(changes, 1):
        card = change_card(c, i, scene)
        cv2.imwrite(str(out / "cards" / f"{i:02d}_{c.kind}.jpg"), cv2.cvtColor(card, cv2.COLOR_RGB2BGR))
        cards.append(card)
    return cards


# --------------------------------------------------------------------------------------- top-down ---

def topdown(changes: list[Change], a: Cloud, b: Cloud, scene: dict, path: Path):
    """Bird's-eye 'what changed for navigation' map: static obstacles, walked paths, changes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from scipy.spatial import ConvexHull

    metric = scene.get("metric")
    lo_h, hi_h = (0.08, 1.8) if metric else (0.05 * _ptp_h(a), 0.8 * _ptp_h(a))
    P = np.concatenate([a.xyz[a.state == STATIC], b.xyz[b.state == STATIC]])
    floor = P[P[:, 1] < lo_h]
    obst = P[(P[:, 1] >= lo_h) & (P[:, 1] <= hi_h)]
    fig, ax = plt.subplots(figsize=(9, 9), dpi=130)
    ax.set_facecolor("#fafafa")
    rng = np.random.default_rng(0)
    for Q, col, s in ((floor, "#d9d9d9", 0.3), (obst, "#555555", 0.4)):
        if len(Q):
            Q = Q[rng.choice(len(Q), min(len(Q), 60000), replace=False)]
            ax.scatter(Q[:, 0], -Q[:, 2], s=s, c=col, linewidths=0, rasterized=True)
    for r, col, name in ((0, "#3a6ea5", "path A (before)"), (1, "#8e44ad", "path B (after)")):
        C = cam_centers(scene["extrinsic"][scene["rec"] == r])
        ax.plot(C[:, 0], -C[:, 2], "--", color=col, lw=1.5, label=name)
        ax.scatter(C[0, 0], -C[0, 2], color=col, marker="o", s=30)

    def hull(Q, col, alpha=0.35, hatch=None):
        if len(Q) < 4:
            return
        xy = Q[:, [0, 2]] * [1, -1]
        try:
            hv = xy[ConvexHull(xy).vertices]
        except Exception:
            return
        ax.fill(hv[:, 0], hv[:, 1], color=col, alpha=alpha, hatch=hatch, lw=0)
        ax.plot(*np.vstack([hv, hv[:1]]).T, color=col, lw=2)

    for i, c in enumerate(changes, 1):
        col = np.array(COLORS[c.kind]) / 255
        if c.kind == "moved":
            hull(c.points, col, 0.25, "//")
            hull(c.points_to, np.array(COLORS["moved_to"]) / 255, 0.4)
            p0, p1 = np.array(c.centroid)[[0, 2]] * [1, -1], np.array(c.centroid_to)[[0, 2]] * [1, -1]
            ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=18, color="k", lw=1.8))
            at = p1
        else:
            hull(c.points, col, 0.4, "//" if c.kind == "removed" else ".." if c.kind == "appearance" else None)
            at = np.array(c.centroid)[[0, 2]] * [1, -1]
        ax.annotate(f"{i}. {c.label}", at, xytext=(6, 6), textcoords="offset points", fontsize=10,
                    weight="bold", color=col * 0.8, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.9))
    for k, lab in (("removed", "removed"), ("added", "added"), ("moved", "moved (before)"), ("moved_to", "moved (after)"),
                   ("appearance", "looks different (flat)")):
        ax.scatter([], [], color=np.array(COLORS[k]) / 255, marker="s", s=60, label=lab)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_aspect("equal")
    unit = "m" if metric else "units"
    ax.set_xlabel(f"x ({unit})")
    ax.set_ylabel(f"forward, -z ({unit})")
    ax.set_title("Top-down change map (grey = obstacles at robot height)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def navigation_figure(g, routes, views, changes: list[Change], path: Path):
    """Two panels: the robot's map before and after, with the walked routes re-planned in each,
    plus 'where to look next' suggestions for the areas no one could verify."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    from .nav import FREE, OCC

    cmap = ListedColormap(["#c9c9cf", "#ffffff", "#2b2d33"])  # unknown, free, occupied
    ext = [g.origin[0], g.origin[0] + g.before.shape[1] * g.res, -(g.origin[1] + g.before.shape[0] * g.res), -g.origin[1]]
    fig, axs = plt.subplots(1, 2, figsize=(14, 7.2), dpi=120)
    for ax, layer, key, title in ((axs[0], g.before, "path_before", "BEFORE: robot map + planned routes"),
                                  (axs[1], g.after, "path_after", "AFTER: updated map, routes re-planned")):
        img = np.where(layer == OCC, 2, np.where(layer == FREE, 1, 0))
        ax.imshow(img, cmap=cmap, extent=ext, origin="upper", interpolation="nearest", vmin=0, vmax=2)
        for r, col in zip(routes, ("#1f77b4", "#9b59b6")):
            P = np.array(getattr(r, key))
            if not len(P) and key == "path_after" and r.status == "blocked":
                ax.plot([r.start[0], r.goal[0]], [-r.start[1], -r.goal[1]], "--", color="#d62728", lw=2, label=f"{r.name}: BLOCKED")
                ax.plot(*np.mean([[r.start[0], -r.start[1]], [r.goal[0], -r.goal[1]]], 0), "x", color="#d62728", ms=16, mew=3)
            if len(P):
                ax.plot(P[:, 0], -P[:, 1], color=col, lw=2.5, label=f"{r.name}: {getattr(r, 'before_len' if key == 'path_before' else 'after_len'):.1f} m")
            ax.plot(r.start[0], -r.start[1], "o", color=col, ms=7)
            ax.plot(r.goal[0], -r.goal[1], "s", color=col, ms=7)
        for i, c in enumerate(changes, 1):
            if key == "path_before" and c.kind in ("removed", "moved"):
                P, col = c.points, np.array(COLORS[c.kind]) / 255
            elif key == "path_after" and c.kind in ("added", "moved"):
                P, col = (c.points_to if c.kind == "moved" else c.points), np.array(COLORS["moved_to" if c.kind == "moved" else c.kind]) / 255
            else:
                continue
            ax.scatter(P[:, 0], -P[:, 2], s=2, color=col, alpha=0.6)
            ax.annotate(f"#{i} {c.label}", (np.median(P[:, 0]), -np.median(P[:, 2])), fontsize=9, weight="bold",
                        color=col * 0.75, xytext=(5, 5), textcoords="offset points")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    for n, v in enumerate(views, 1):
        x, z = v.position[0], v.position[2]
        yaw = np.deg2rad(v.yaw_deg)
        for da in (-0.55, 0.55):
            axs[1].plot([x, x + 0.9 * np.sin(yaw + da)], [-z, -z + 0.9 * np.cos(yaw + da)], color="#e67e22", lw=1.5)
        axs[1].plot(x, -z, marker="^", color="#e67e22", ms=11)
        axs[1].annotate(f"look #{n}", (x, -z), xytext=(6, -12), textcoords="offset points", color="#c0640a", fontsize=9, weight="bold")
    lines = [r.sentence for r in routes] + [f"look #{n}: {v.sentence}" for n, v in enumerate(views, 1)]
    fig.text(0.01, 0.01, "\n".join(lines), fontsize=9, va="bottom", family="sans-serif")
    fig.tight_layout(rect=(0, 0.04 + 0.018 * len(lines), 1, 1))
    fig.savefig(path)
    plt.close(fig)


def _ptp_h(c: Cloud):
    return float(np.ptp(c.xyz[:, 1]))


# ----------------------------------------------------------------------------------------- viewer ---

def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _jpg64(img: np.ndarray, h: int = 220) -> str:
    img = _fit(img, h)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def viewer_html(changes: list[Change], a: Cloud, b: Cloud, scene: dict, path: Path, budget: int = 160_000,
                routes=(), views=()):
    rng = np.random.default_rng(0)

    def sub(cloud: Cloud, mask=None):
        idx = np.arange(len(cloud.xyz)) if mask is None else np.where(mask)[0]
        if len(idx) > budget:
            idx = rng.choice(idx, budget, replace=False)
        return cloud.xyz[idx], cloud.rgb[idx], cloud.state[idx]

    Pa, Ca, Sa = sub(a)
    Pb, Cb, Sb = sub(b)
    allP = np.concatenate([Pa, Pb])
    lo, hi = np.percentile(allP, 0.5, 0), np.percentile(allP, 99.5, 0)
    span = np.maximum(hi - lo, 1e-6)

    def q(P):  # int16 quantisation keeps the page small
        return np.clip((P - lo) / span * 65535 - 32768, -32768, 32767).astype(np.int16)

    objs = []
    for i, c in enumerate(changes, 1):
        o = dict(id=i, kind=c.kind, label=c.label, sentence=c.sentence, nav=c.nav, score=round(c.score, 2),
                 c0=np.round(c.centroid, 3).tolist(), c1=np.round(c.centroid_to, 3).tolist() if c.centroid_to else None,
                 color=_hex(COLORS[c.kind]))
        P = c.points[rng.choice(len(c.points), min(len(c.points), 6000), replace=False)]
        o["p0"] = _b64(q(P))
        if c.kind == "moved":
            P = c.points_to[rng.choice(len(c.points_to), min(len(c.points_to), 6000), replace=False)]
            o["p1"] = _b64(q(P))
        for key in ("before", "after"):
            v = c.views.get(key)
            if v is not None:
                o[key] = _jpg64(crop(scene, v, pad=0.5))
        o["evidence"] = [dict(name=n, expected=_jpg64(e, 200), actual=_jpg64(a_, 200))
                         for n, e, a_ in (getattr(c, "evidence", []) or [])]
        if c.kind == "moved":
            o["yaw"] = round(c.yaw_deg, 1)
        objs.append(o)

    cams = {r: cam_centers(scene["extrinsic"][scene["rec"] == r]).round(3).tolist() for r in (0, 1)}
    data = dict(lo=lo.tolist(), span=span.tolist(), unit="m" if scene.get("metric") else "units",
                a=dict(p=_b64(q(Pa)), c=_b64(Ca), s=_b64(Sa.astype(np.uint8))),
                b=dict(p=_b64(q(Pb)), c=_b64(Cb), s=_b64(Sb.astype(np.uint8))),
                cams=cams, changes=objs,
                routes=[dict(name=r.name, status=r.status, sentence=r.sentence, before=r.path_before[::2],
                             after=r.path_after[::2]) for r in routes],
                views=[dict(p=v.position, yaw=v.yaw_deg, sentence=v.sentence) for v in views])
    html = (Path(__file__).parent / "viewer_template.html").read_text()
    path.write_text(html.replace("/*__DATA__*/null", json.dumps(data)))


def summary_image(cards: list[np.ndarray], topdown_png: Path, path: Path):
    td = cv2.cvtColor(cv2.imread(str(topdown_png)), cv2.COLOR_BGR2RGB)
    if not cards:
        cv2.imwrite(str(path), cv2.cvtColor(td, cv2.COLOR_RGB2BGR))
        return
    W = max(c.shape[1] for c in cards)
    cards = [cv2.copyMakeBorder(c, 6, 6, 6, 6 + W - c.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for c in cards]
    col = np.vstack(cards)
    td = _fit(td, col.shape[0]) if col.shape[0] < 2 * td.shape[0] else _fit(td, td.shape[0])
    H = max(col.shape[0], td.shape[0])
    pad = lambda x: cv2.copyMakeBorder(x, 0, H - x.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.imwrite(str(path), cv2.cvtColor(np.hstack([pad(td), pad(col)]), cv2.COLOR_RGB2BGR))


def orbit_gif(changes: list[Change], a: Cloud, b: Cloud, path: Path, frames: int = 40, n_static: int = 40000):
    """A rotating 3D view of the room with the changes highlighted (for READMEs and slides)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    rng = np.random.default_rng(0)
    S = np.concatenate([a.xyz[a.state == STATIC], b.xyz[b.state == STATIC]])
    C = np.concatenate([a.rgb[a.state == STATIC], b.rgb[b.state == STATIC]]).astype(float) / 255
    S = S[S[:, 1] < np.percentile(S[:, 1], 92)]  # drop the ceiling so we can see in
    idx = rng.choice(len(S), min(len(S), n_static), replace=False)
    S, C = S[idx], C[idx] if len(C) == len(idx) else C[: len(idx)]
    grey = (0.35 + 0.4 * C[:, :3].mean(1, keepdims=True)).repeat(3, 1) if len(C) else C
    lo, hi = np.percentile(S, 1, 0), np.percentile(S, 99, 0)
    imgs = []
    for k in range(frames):
        fig = plt.figure(figsize=(6.4, 4.8), dpi=100)
        ax = fig.add_subplot(projection="3d", computed_zorder=False)
        ax.set_facecolor("#15171c")
        fig.patch.set_facecolor("#15171c")
        ax.scatter(S[:, 0], -S[:, 2], S[:, 1], s=0.4, c=grey, linewidths=0, depthshade=False)
        for i, c in enumerate(changes, 1):
            sets = [(c.points, COLORS[c.kind])] + ([(c.points_to, COLORS["moved_to"])] if c.kind == "moved" else [])
            for P, col in sets:
                P = P[rng.choice(len(P), min(len(P), 2500), replace=False)]
                ax.scatter(P[:, 0], -P[:, 2], P[:, 1], s=2.5, color=np.array(col) / 255, linewidths=0, depthshade=False)
            if c.kind == "moved":
                p0, p1 = np.array(c.centroid), np.array(c.centroid_to)
                ax.plot([p0[0], p1[0]], [-p0[2], -p1[2]], [p0[1] + 0.5, p1[1] + 0.5], color="white", lw=1.5)
            top = (c.points_to if c.kind == "moved" else c.points)
            t = np.median(top, 0)
            ax.text(t[0], -t[2], np.percentile(top[:, 1], 98) + 0.25, f"{i}. {c.kind} {c.label}", color=np.array(COLORS[c.kind]) / 255,
                    fontsize=8, weight="bold", ha="center")
        ax.view_init(elev=38, azim=-60 + 360 * k / frames)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(-hi[2], -lo[2])
        ax.set_zlim(0, hi[1])
        ax.set_box_aspect((hi[0] - lo[0], hi[2] - lo[2], hi[1]), zoom=1.45)
        ax.set_axis_off()
        fig.subplots_adjust(0, 0, 1, 1)
        fig.canvas.draw()
        imgs.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=90, loop=0, optimize=True)
