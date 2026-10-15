"""Run the full pipeline (VGGT geometry, not ground truth) on every synthetic scenario and score it.

    python scripts/benchmark.py            # renders missing scenarios, runs, writes docs/BENCHMARK.md
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = {
    "standard": "chair moved 1.7 m + turned 35°, bin removed, backpack added; after-video on another path/height/exposure",
    "hard": "chair moved, 1 cm rug removed (appearance only), removals the after-video cannot verify (never filmed / behind a cabinet / never visible before); after-video covers part of the room",
    "nochange": "same room filmed twice on different paths",
}


def sh(*cmd):
    print("$", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(list(map(str, cmd)), cwd=ROOT, check=True, capture_output=True, text=True).stdout


def main():
    py = sys.executable
    rows = []
    for name, desc in SCENARIOS.items():
        data = ROOT / ("data/synthetic" if name == "standard" else f"data/synthetic_{name}")
        out = ROOT / f"outputs/bench_{name}"
        if not (data / "B.mp4").exists():
            sh(py, "scripts/make_synthetic.py", "--scenario", name, "--out", data)
        sh(py, "-W", "ignore", "-m", "roomnesia", data / "A.mp4", data / "B.mp4", "--out", out, "--no-vlm")
        txt = sh(py, "-W", "ignore", "scripts/eval_synthetic.py", "--run", out, "--gt", data / "ground_truth.json")
        ev = json.loads(txt[txt.index("{"):])
        info = json.loads((out / "run_info.json").read_text())
        rows.append((name, desc, ev, info))
        print(name, json.dumps(ev), flush=True)

    md = ["# Synthetic benchmark", "",
          "Full pipeline on rendered videos (VGGT reconstruction, not ground-truth depth). "
          "Positions are scored after a similarity alignment of the camera centres to ground truth "
          "(the metric scale is estimated by a monocular network, so it is not exact). "
          "Reproduce: `uv run python scripts/benchmark.py`.", "",
          "| scenario | what happens | changes found | position error | must NOT be reported | false changes | null test (false clusters) |",
          "|---|---|---|---|---|---|---|"]
    for name, desc, ev, info in rows:
        found = sum(m["found"] for m in ev["matches"])
        errs = ", ".join(f"{m['gt'].split()[0]} {m['error_m'] * 100:.0f} cm" for m in ev["matches"] if m["found"]) or "–"
        abst = ", ".join(f"{a['object']}: {'✅ not reported' if a['correctly_not_reported'] else '❌ reported'}" for a in ev["abstentions"]) or "–"
        nt = info.get("null_test") or {}
        md.append(f"| {name} | {desc} | {found}/{len(ev['matches'])} | {errs} | {abst} | {ev['false_positives']} | "
                  f"{', '.join(str(v['false_changes']) for v in nt.values())} |")
    (ROOT / "docs/BENCHMARK.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
