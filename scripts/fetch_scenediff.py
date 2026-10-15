"""Fetch selected video pairs of the SceneDiff benchmark (Wu et al., 2025; MIT licence) WITHOUT downloading
the whole 12 GB archive: the zip's directory and the wanted members are read with HTTP range requests.

    python scripts/fetch_scenediff.py --list                      # list pairs in the archive
    python scripts/fetch_scenediff.py PAIR [PAIR ...]             # -> data/scenediff/pairs/PAIR/
    python scripts/fetch_scenediff.py --from data/scenediff/test_selection.txt

Annotations (segments.pkl) are only ever read with scripts/safe_pickle.py (plain data, no code).
"""

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://huggingface.co/datasets/yuqun/SceneDiff/resolve/main/scenediff_benchmark.zip"
OUT = Path(__file__).resolve().parents[1] / "data/scenediff/pairs"
FILES = ("video1.mp4", "video2.mp4", "segments.pkl", "review_meta.json")


class HttpFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests."""

    def __init__(self, url):
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.url, self.size = r.geturl(), int(r.headers["Content-Length"])
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.size, self.pos + len(b)) - 1
        with urllib.request.urlopen(urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})) as r:
            data = r.read()
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="*")
    ap.add_argument("--from", dest="src", type=Path)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    z = zipfile.ZipFile(io.BufferedReader(HttpFile(URL), buffer_size=1 << 20))
    if args.list:
        print("\n".join(sorted({n.split("/")[2] for n in z.namelist() if n.count("/") >= 3 and "/data/" in n})))
        return
    pairs = args.pairs + (args.src.read_text().split() if args.src else [])
    for pair in pairs:
        (OUT / pair).mkdir(parents=True, exist_ok=True)
        for f in FILES:
            (OUT / pair / f).write_bytes(z.read(f"scenediff_benchmark/data/{pair}/{f}"))
        print("fetched", pair, flush=True)


if __name__ == "__main__":
    main()
