"""Run Roomnesia on SceneDiff pairs and summarise the object-level evaluation.

    python scripts/run_scenediff.py --split val  pair1 pair2 ...
    python scripts/run_scenediff.py --split test pair1 pair2 ...   # writes docs/SCENEDIFF.md

Pairs are read from data/scenediff/pairs/<pair>/ (video1.mp4, video2.mp4, segments.pkl).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("pairs", nargs="+")
    args = ap.parse_args()
    py = sys.executable
    results = []
    for pair in args.pairs:
        src, out = ROOT / "data/scenediff/pairs" / pair, ROOT / f"outputs/sd_{args.split}/{pair}"
        if not (out / "eval_data.npz").exists():
            r = subprocess.run([py, "-W", "ignore", "-m", "roomnesia", src / "video1.mp4", src / "video2.mp4",
                                "--out", out, "--no-vlm"], cwd=ROOT, capture_output=True, text=True)
            if r.returncode:
                print(f"{pair}: FAILED\n{r.stderr[-1500:]}", flush=True)
                results.append(dict(pair=pair, failed=True))
                continue
        r = subprocess.run([py, "-W", "ignore", "scripts/eval_scenediff.py", src, out], cwd=ROOT, capture_output=True, text=True)
        res = json.loads(r.stdout[r.stdout.index("{"):])
        results.append(res)
        print(f"{pair}: found {res['found']}/{res['gt']} (type ok {res['type_correct']}), "
              f"{res['predictions']} predictions, {res['false_positives']} false", flush=True)

    ok = [r for r in results if not r.get("failed")]
    tot = {k: sum(r[k] for r in ok) for k in ("gt", "found", "type_correct", "predictions", "false_positives")}
    by_kind = {}
    for r in ok:
        for o in r["objects"]:
            d = by_kind.setdefault(o["kind"], [0, 0])
            d[0] += 1
            d[1] += o["found"]
    summary = dict(split=args.split, pairs=len(results), failed=len(results) - len(ok), **tot,
                   recall=tot["found"] / max(tot["gt"], 1),
                   precision=(tot["predictions"] - tot["false_positives"]) / max(tot["predictions"], 1),
                   by_kind={k: f"{v[1]}/{v[0]}" for k, v in by_kind.items()}, results=results)
    (ROOT / f"outputs/sd_{args.split}/summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
