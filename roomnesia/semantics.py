"""From geometric change clusters to semantic objects: views, crops, labels, moves, and language."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .change import ChangeCluster, Cloud
from .landmarks import overlapping, place_phrase
from .geometry import project
from .reconstruct import pick_device

VOCAB = [
    "chair", "office chair", "stool", "armchair", "sofa", "cushion", "pillow", "blanket", "table", "desk",
    "coffee table", "side table", "bed", "shelf", "bookshelf", "cabinet", "drawer", "wardrobe", "box",
    "cardboard box", "storage box", "basket", "laundry basket", "bin", "trash can", "bucket", "backpack",
    "bag", "handbag", "suitcase", "shoe", "shoes", "boots", "jacket", "coat", "towel", "clothes", "hat",
    "book", "stack of books", "notebook", "laptop", "computer monitor", "keyboard", "mouse", "phone",
    "tablet", "television", "speaker", "headphones", "game controller", "cable", "lamp", "desk lamp",
    "floor lamp", "plant", "potted plant", "flower vase", "vase", "bottle", "water bottle", "cup", "mug",
    "glass", "bowl", "plate", "kettle", "toy", "teddy bear", "ball", "guitar", "picture frame", "poster",
    "mirror", "clock", "fan", "heater", "vacuum cleaner", "umbrella", "helmet", "skateboard", "bicycle",
    "rug", "mat", "door", "window", "curtain", "radiator", "tool box", "printer", "water jug", "candle",
    "tissue box", "paper", "folder", "pen holder", "fruit", "food container", "yoga mat", "dumbbell",
]


@dataclass
class View:
    frame: int
    box: tuple  # (x0, y0, x1, y1) in model-resolution pixels
    score: float


@dataclass
class Change:
    kind: str  # removed | added | moved
    label: str = ""
    label_conf: float = 0.0
    phrase: str = ""
    sentence: str = ""
    score: float = 0.0
    centroid: list = field(default_factory=list)  # room frame (metres if metric)
    centroid_to: list = field(default_factory=list)  # for moves
    size: list = field(default_factory=list)  # gravity-aligned extent (x, y, z)
    height: float = 0.0  # bottom of object above floor
    displacement: float = 0.0
    yaw_deg: float = 0.0
    nav: str = ""
    place_from: str = ""  # relation to unchanged landmarks, e.g. "on the floor next to the sofa"
    place_to: str = ""
    label_source: str = "siglip"
    views: dict = field(default_factory=dict)  # name -> View
    points: np.ndarray = None  # (N, 3) object points (before-state for moves)
    points_to: np.ndarray = None
    colors: np.ndarray = None
    colors_to: np.ndarray = None
    emb: np.ndarray = None


# --------------------------------------------------------------------------------------- views ---

def best_views(P: np.ndarray, scene: dict, rec: int, mode: str, k: int = 3, tol: float = 0.05) -> list[View]:
    """Frames of recording `rec` that best show the 3D region P.
    mode='surface': P is visible there (depth agrees). mode='through': the camera sees *behind* P,
    i.e. it shows the now-empty space (used for 'after' shots of removals / 'before' shots of additions)."""
    H, W = scene["depth"].shape[1:]
    views = []
    for j in np.where(scene["rec"] == rec)[0]:
        uv, z = project(P, scene["intrinsic"][j], scene["extrinsic"][j])
        u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        ins = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if ins.sum() < 0.5 * len(P):
            continue
        D = scene["depth"][j][v[ins], u[ins]]
        zz = z[ins]
        ok = np.abs(D - zz) < tol * zz + 0.02 if mode == "surface" else D > zz + tol * zz
        if ok.sum() < 10:
            continue
        uu, vv = uv[ins][:, 0], uv[ins][:, 1]
        x0, x1, y0, y1 = np.percentile(uu, 2), np.percentile(uu, 98), np.percentile(vv, 2), np.percentile(vv, 98)
        frac_in = ins.mean()
        centrality = 1 - np.hypot((x0 + x1) / 2 / W - 0.5, (y0 + y1) / 2 / H - 0.5)
        area = (x1 - x0) * (y1 - y0) / (W * H)
        size_pref = min(area / 0.08, 1.0)  # prefer views where the object is reasonably large
        views.append(View(int(j), (x0, y0, x1, y1), float(ok.mean() * frac_in * centrality * (0.3 + size_pref))))
    return sorted(views, key=lambda v: -v.score)[:k]


def crop(scene: dict, view: View, pad: float = 0.35, min_px: int = 40) -> np.ndarray:
    img = scene["hires"][view.frame]
    s = img.shape[0] / scene["images"].shape[1]
    x0, y0, x1, y1 = view.box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0, min_px) * (0.5 + pad)
    X0, X1 = int(max(0, (cx - half) * s)), int(min(img.shape[1], (cx + half) * s))
    Y0, Y1 = int(max(0, (cy - half) * s)), int(min(img.shape[0], (cy + half) * s))
    return img[Y0:Y1, X0:X1].copy()


def draw_box(scene: dict, view: View, color) -> np.ndarray:
    img = scene["hires"][view.frame].copy()
    s = img.shape[0] / scene["images"].shape[1]
    x0, y0, x1, y1 = (int(c * s) for c in view.box)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, max(2, img.shape[0] // 200))
    return img


# ------------------------------------------------------------------------------------- labeling ---

def _feat(out):
    """transformers >= 5 returns a model output object from get_*_features; older versions a tensor."""
    return out if torch.is_tensor(out) else out.pooler_output


class Labeler:
    """Open-vocabulary labels + image embeddings with SigLIP (runs fine on Apple MPS)."""

    def __init__(self, name: str = "google/siglip-base-patch16-224"):
        from transformers import AutoModel, AutoProcessor

        self.dev = pick_device()
        self.proc = AutoProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.dev).eval()
        with torch.no_grad():
            t = self.proc(text=[f"a photo of a {c}." for c in VOCAB], padding="max_length", return_tensors="pt").to(self.dev)
            self.text = torch.nn.functional.normalize(_feat(self.model.get_text_features(**t)), dim=-1)

    @torch.no_grad()
    def __call__(self, crops: list[np.ndarray]):
        x = self.proc(images=crops, return_tensors="pt").to(self.dev)
        e = torch.nn.functional.normalize(_feat(self.model.get_image_features(**x)), dim=-1)
        logits = e @ self.text.T * self.model.logit_scale.exp() + self.model.logit_bias
        probs = torch.sigmoid(logits).mean(0)  # average over views
        top = int(probs.argmax())
        emb = torch.nn.functional.normalize(e.mean(0), dim=-1).cpu().numpy()
        return VOCAB[top], float(probs[top]), emb


class Captioner:
    """Short noun phrase for an object crop from a small local VLM (SmolVLM2), optional."""

    def __init__(self, name: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.dev = pick_device()
        self.proc = AutoProcessor.from_pretrained(name)
        self.model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16).to(self.dev).eval()

    @torch.no_grad()
    def __call__(self, crop_img: np.ndarray, hint: str) -> str:
        from PIL import Image

        msgs = [{"role": "user", "content": [
            {"type": "image", "image": Image.fromarray(crop_img)},
            {"type": "text", "text": f"What is the object in the centre of this photo (probably a {hint})? "
                                     "Answer with a short noun phrase of 2-6 words including its colour, "
                                     "e.g. 'a red plastic chair'."},
        ]}]
        inp = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                            return_tensors="pt").to(self.dev)
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        out = self.model.generate(**inp, max_new_tokens=16, do_sample=False)
        txt = self.proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        txt = txt.strip().strip(".").split("\n")[0]
        if txt.lower().startswith("assistant:"):
            txt = txt[10:].strip()
        return txt[:60] or f"a {hint}"


# ------------------------------------------------------------------------------------ geometry ---

def extent(P: np.ndarray):
    lo, hi = np.percentile(P, 3, 0), np.percentile(P, 97, 0)
    return (lo + hi) / 2, hi - lo, float(lo[1])


def color_hist(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    h = np.histogram2d(hsv[:, 0], hsv[:, 1], bins=(12, 4), range=((0, 180), (0, 256)))[0].ravel()
    return h / max(h.sum(), 1)


def fit_move(Pa: np.ndarray, Pb: np.ndarray, iters: int = 25) -> tuple[float, np.ndarray]:
    """Rigid fit of an object's before/after points constrained to yaw about the vertical axis
    (furniture slides and turns on the floor). Grid over yaw + point-to-point ICP; returns yaw, translation."""
    rng = np.random.default_rng(0)
    A = Pa[rng.choice(len(Pa), min(len(Pa), 3000), replace=False)]
    B = Pb[rng.choice(len(Pb), min(len(Pb), 3000), replace=False)]
    tree = cKDTree(B)
    best = (np.inf, 0.0, B.mean(0) - A.mean(0))
    for yaw0 in np.deg2rad(np.arange(-180, 180, 30)):
        yaw, t = yaw0, B.mean(0) - _rot(yaw0) @ A.mean(0)
        for _ in range(iters):
            X = A @ _rot(yaw).T + t
            d, nn = tree.query(X)
            keep = d < np.percentile(d, 80)
            src, dst = A[keep], B[nn[keep]]
            # closed-form 2D (x, z) rotation + 3D translation
            sc, dc = src.mean(0), dst.mean(0)
            s2, d2 = (src - sc)[:, [0, 2]], (dst - dc)[:, [0, 2]]
            yaw = np.arctan2((s2[:, 0] * d2[:, 1] - s2[:, 1] * d2[:, 0]).sum(), (s2 * d2).sum())
            yaw = -yaw  # convert to rotation about +y with our _rot convention
            t = dc - _rot(yaw) @ sc
        # symmetric objects fit equally well at several angles: prefer the smallest rotation
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
        err = np.median(tree.query(A @ _rot(yaw).T + t)[0]) * (1 + 0.05 * abs(yaw) / np.pi)
        if err < best[0]:
            best = (err, yaw, t)
    return float(np.degrees(best[1])), best[2]


def _rot(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# ------------------------------------------------------------------------------------ language ---

def _dir_words(d: np.ndarray, view0: np.ndarray, metric: bool) -> str:
    """Displacement in words, relative to the viewpoint where recording A started."""
    unit = "m" if metric else " units"
    lr, fb = view0 @ d[[0, 2]]
    norm = np.hypot(lr, fb)
    parts = []
    if abs(lr) > 0.25 * norm and abs(lr) > 0.05:
        parts.append(f"{abs(lr):.1f}{unit} to the {'right' if lr > 0 else 'left'}")
    if abs(fb) > 0.25 * norm and abs(fb) > 0.05:
        parts.append(f"{abs(fb):.1f}{unit} {'further away' if fb > 0 else 'closer'}")
    return " and ".join(parts)


def _where(c: Change, metric: bool, prep: str, place: str) -> str:
    if place:
        return place.replace("on the", f"{prep} the", 1) if prep != "on" else place
    if not metric:
        return ""
    if c.height < 0.12:
        return f"{prep} the floor"
    return f"{prep} a surface ~{c.height:.1f} m up"


def describe(c: Change, scene: dict) -> str:
    metric = bool(scene.get("metric"))
    name = c.phrase or f"a {c.label}"
    name = name[0].upper() + name[1:]
    unit = "m" if metric else " units"
    if c.kind == "moved":
        s = f"{name} was moved {c.displacement:.1f}{unit}"
        if c.place_from or c.place_to:
            strip = lambda t: t[3:] if t.startswith("on ") else t  # "from on the floor" -> "from the floor"
            s += f", from {strip(c.place_from) or 'its old spot'} to {strip(c.place_to) or 'a new spot'}"
        dw = _dir_words(np.array(c.centroid_to) - np.array(c.centroid), scene["view0"], metric)
        if dw:
            s += f" ({dw}, as seen from where recording A started)"
        if abs(c.yaw_deg) > 15:
            s += f", and turned ~{abs(c.yaw_deg):.0f}°"
        return s + "."
    size = f" (~{max(c.size[0], c.size[2]):.1f}{unit} wide, {c.size[1]:.1f}{unit} tall)" if c.size else ""
    if c.kind == "appearance":
        ext = sorted(c.size)[-2:] if c.size else None
        area = f" over ~{ext[1]:.1f} x {ext[0]:.1f}{unit}" if ext else ""
        where = _where(c, metric, "on", c.place_from)
        thing = c.phrase or f"the {c.label}"
        now = getattr(c, "label_after", "")
        now = f"; that spot now looks like a {now}" if now and now != c.label else ""
        return (f"{thing[0].upper() + thing[1:]}{' ' + where if where else ''} was removed or replaced: a flat change "
                f"(shape unchanged, colour changed{area}){now}.")
    if c.kind == "added":
        where = _where(c, metric, "on", c.place_from)
        return f"{name}{size} appeared{' ' + where if where else ''}."
    where = _where(c, metric, "from", c.place_from)
    return f"{name}{size} was removed{' ' + where if where else ''}."


# ------------------------------------------------------------------------------------ pipeline ---

def build_changes(scene: dict, a: Cloud, b: Cloud, clusters: list[ChangeCluster], use_vlm: bool = True,
                  max_changes: int = 12, marks=()) -> list[Change]:
    metric = bool(scene.get("metric"))
    clusters = sorted(clusters, key=lambda c: -c.score * len(c.core))[: 2 * max_changes]
    labeler = Labeler()
    items = []
    for cl in clusters:
        cloud, rec = (b, 1) if cl.kind == "added" else (a, 0)
        P = cloud.xyz[cl.idx]
        vis = best_views(P, scene, rec, "surface")
        if not vis:
            continue
        crops = [crop(scene, v) for v in vis]
        label, conf, emb = labeler(crops)
        ctr, size, bottom = extent(P)
        ch = Change(kind=cl.kind, label=label, label_conf=conf, score=cl.score, centroid=ctr.tolist(),
                    size=size.tolist(), height=bottom, points=P, colors=cloud.rgb[cl.idx], emb=emb)
        other = best_views(P, scene, 1 - rec, "surface" if cl.kind == "appearance" else "through", k=1)
        det = overlapping(P, marks, rec) if marks else None
        if det is not None:  # an independent detector covers the same 3D region: trust its name
            ch.label_source = "detector+siglip" if det.label == label else "detector"
            ch.label = det.label
            ch._det = det
        ch.views = {("before" if rec == 0 else "after"): vis[0]}
        if other:
            ch.views["after" if rec == 0 else "before"] = other[0]
        ch._hist = color_hist(ch.colors)
        if cl.kind == "appearance" and other:
            ch.label_after, _, _ = labeler([crop(scene, other[0])])
        items.append(ch)
    del labeler

    # --- pair removals with additions of the same-looking object -> moves
    rem = [c for c in items if c.kind == "removed"]
    add = [c for c in items if c.kind == "added"]
    moves, used_r, used_a = [], set(), set()
    if rem and add:
        cost = np.zeros((len(rem), len(add)))
        for i, r in enumerate(rem):
            for j, n in enumerate(add):
                app = 1 - float(r.emb @ n.emb)  # appearance (SigLIP)
                col = 1 - np.minimum(r._hist, n._hist).sum()  # colour histogram intersection
                sr, sn = np.sort(r.size), np.sort(n.size)
                shape = np.abs(np.log((sr + 1e-3) / (sn + 1e-3))).mean()  # size, rotation-invariant
                cost[i, j] = app + 0.6 * col + 0.5 * shape + (0 if r.label == n.label else 0.1)
                if col > 0.7 or shape > 0.5:  # clearly different colour or size: never the same object
                    cost[i, j] = np.inf
        ri, ai = linear_sum_assignment(np.minimum(cost, 1e3))
        for i, j in zip(ri, ai):
            if cost[i, j] < 0.5:
                r, n = rem[i], add[j]
                yaw, t = fit_move(r.points, n.points)
                m = Change(kind="moved", label=r.label if (r.label_source != "siglip" or r.label_conf >= n.label_conf) and n.label_source == "siglip" or r.label == n.label else n.label,
                           label_source=r.label_source if r.label_source != "siglip" else n.label_source,
                           label_conf=max(r.label_conf, n.label_conf), score=(r.score + n.score) / 2,
                           centroid=r.centroid, centroid_to=n.centroid, size=r.size, height=r.height,
                           yaw_deg=yaw, points=r.points, colors=r.colors, points_to=n.points, colors_to=n.colors,
                           views={"before": r.views["before"], "after": n.views["after"]})
                m.displacement = float(np.linalg.norm(np.subtract(n.centroid, r.centroid)[[0, 2]]))
                m._pair_cost = float(cost[i, j])
                moves.append(m)
                used_r.add(i)
                used_a.add(j)
    changes = moves + [c for i, c in enumerate(rem) if i not in used_r] + [c for j, c in enumerate(add) if j not in used_a] + \
        [c for c in items if c.kind == "appearance"]
    changes = sorted(changes, key=lambda c: -c.score)[:max_changes]

    # --- where things are, relative to the unchanged landmarks
    involved = set()
    for c in changes:
        for P, rec in ((c.points, 0 if c.kind != "added" else 1), (c.points_to, 1)):
            if P is not None:
                for m in marks:
                    if overlapping(P, [m], rec) is not None or overlapping(P, [m], 1 - rec) is not None:
                        involved.add(id(m))
    for c in changes:
        c.place_from = place_phrase(c.points, marks, involved) if marks else ""
        if c.kind == "moved":
            c.place_to = place_phrase(c.points_to, marks, involved) if marks else ""

    # --- language
    cap = None
    if use_vlm:
        try:
            cap = Captioner()
        except Exception as e:
            print(f"[semantics] VLM captioner unavailable ({e}); using SigLIP labels only")
    for c in changes:
        if cap is not None:
            v = c.views.get("before") if c.kind != "added" else c.views.get("after")
            c.phrase = cap(crop(scene, v), c.label)
        c.sentence = describe(c, scene)
    return changes
