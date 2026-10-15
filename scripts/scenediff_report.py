"""Write docs/SCENEDIFF.md from the SceneDiff runs (outputs/sd_val, outputs/sd_test).

    python scripts/fetch_scenediff.py --from data/scenediff/test_selection.txt
    python scripts/run_scenediff.py --split test $(cat data/scenediff/test_selection.txt)
    python scripts/scenediff_report.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import safe_pickle  # noqa: E402
from eval_scenediff import gt_objects, rle_decode  # noqa: E402

BINS = ((0, 0.005, "small (< 0.5% of the image: letters, pens, cups)"),
        (0.005, 0.03, "medium (0.5–3%: shoes, books, small boxes)"),
        (0.03, 1.0, "large (> 3%: chairs, bags, bins, boxes, stools)"))


def object_sizes(pair: str):
    seg = safe_pickle.load(ROOT / f"data/scenediff/pairs/{pair}/segments.pkl")
    out = []
    for g in gt_objects(seg):
        areas = [rle_decode(fr[k]).mean() for key in ("video1_objects", "video2_objects")
                 for fr in [seg[key].get(g["idx"], {})] for k in list(fr)[::15]]
        out.append(max(areas) if areas else 0.0)
    return out


def section(split: str) -> list[str]:
    s = json.loads((ROOT / f"outputs/sd_{split}/summary.json").read_text())
    md = [f"### {split} split: {s['pairs']} pairs, {s['gt']} annotated changes", "",
          f"**Recall {s['recall']:.0%}** ({s['found']}/{s['gt']}) · **precision {s['precision']:.0%}** "
          f"({s['false_positives']} false of {s['predictions']} reports) · change type correct for {s['type_correct']} · "
          f"by type: " + ", ".join(f"{k} {v}" for k, v in s["by_kind"].items()), ""]
    size_hits = {b[2]: [0, 0] for b in BINS}
    rows = ["| pair | found | false reports | missed objects |", "|---|---|---|---|"]
    for r in s["results"]:
        for size, o in zip(object_sizes(r["pair"]), r["objects"]):
            for lo, hi, name in BINS:
                if lo <= size < hi:
                    size_hits[name][0] += o["found"]
                    size_hits[name][1] += 1
        missed = ", ".join(f"{o['label']} ({o['kind']})" for o in r["objects"] if not o["found"]) or "–"
        rows.append(f"| {r['pair']} | {r['found']}/{r['gt']} | {r['false_positives']} | {missed} |")
    md += ["| object size (largest view) | recall |", "|---|---|"]
    md += [f"| {k} | **{v[0]}/{v[1]}** ({v[0] / max(v[1], 1):.0%}) |" for k, v in size_hits.items()]
    return md + [""] + rows + [""]


def main():
    md = ["# Real-world benchmark: SceneDiff", "",
          "[SceneDiff](https://huggingface.co/datasets/yuqun/SceneDiff) (Wu et al., 2025, "
          "[arXiv:2512.16908](https://arxiv.org/abs/2512.16908), MIT licence) is a set of real phone-video pairs of "
          "the same scene, before and after objects were added, removed or moved, with a mask for every changed "
          "object in every frame.", "",
          "**Protocol.**",
          "- **Pairs:** indoor, room-scale pairs only (bedrooms, living rooms, offices, lounges, …). Kitchen "
          "head-camera clips and outdoor scenes are out of scope.",
          "- **Validation split:** 13 pairs, used to develop and fix the system.",
          "- **Test split:** every third indoor pair of the official test split, taken alphabetically "
          "(`data/scenediff/test_selection.txt`). Run once, after the code was frozen.",
          "- **Run settings:** default; `--no-vlm`, because the noun phrases don't affect the scores.",
          "", "**Scoring** (`scripts/eval_scenediff.py`):",
          "- Each detected change is projected into the annotated frames (only where it is the visible surface).",
          "- An annotated object counts as found when a detection of a compatible type overlaps its mask with "
          "IoU ≥ 0.2 in some keyframe.",
          "- Moves count as found if either end is matched; \"type correct\" needs the detection itself to be a move.",
          "- A detection that overlaps no annotated object is a false report.",
          "- This is our own object-level protocol. The SceneDiff paper reports mask AP, which needs per-pixel "
          "confidence scores, so the numbers are not directly comparable.", ""]
    for split in ("test", "val"):
        if (ROOT / f"outputs/sd_{split}/summary.json").exists():
            md += section(split)
    md += ["**Reading the numbers.** Roomnesia reasons in 3D with ~2 cm voxels and requires several cameras "
           "to see through a spot before calling it a change. Furniture-scale objects (the obstacles a navigating "
           "robot cares about) are found reliably. Objects that cover under ~0.5% of the image (a letter, a "
           "marker) are below that resolution and are missed. Some \"false reports\" are real but unannotated "
           "changes (pillows and bedding moved, chairs nudged); they are counted as errors anyway.", ""]
    (ROOT / "docs/SCENEDIFF.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
