"""Video -> a small set of sharp, well-spread keyframes."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

PATCH = 14


def extract_frames(video: Path, fps: float = 6.0) -> list[np.ndarray]:
    """Decode a video with ffmpeg (handles phone rotation metadata + HEVC) into RGB frames."""
    video = Path(video)
    if video.is_dir():  # a folder of images is also accepted
        paths = sorted(p for p in video.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        return [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths]
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", str(video),
             "-vf", f"fps={fps},scale='min(1280,iw)':-2", "-q:v", "2", f"{tmp}/%05d.jpg"],
            check=True,
        )
        paths = sorted(Path(tmp).glob("*.jpg"))
        return [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths]


def sharpness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(cv2.resize(img, (320, int(320 * img.shape[0] / img.shape[1]))), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_keyframes(frames: list[np.ndarray], n: int) -> list[int]:
    """Split the video into n equal time bins and keep the sharpest frame of each (motion blur kills depth)."""
    if len(frames) <= n:
        return list(range(len(frames)))
    bins = np.array_split(np.arange(len(frames)), n)
    return [int(b[np.argmax([sharpness(frames[i]) for i in b])]) for b in bins]


def model_size(h: int, w: int, long_side: int = 518) -> tuple[int, int]:
    """Resize so the long side is `long_side` and both sides are multiples of the ViT patch size."""
    s = long_side / max(h, w)
    return (max(PATCH, round(h * s / PATCH) * PATCH), max(PATCH, round(w * s / PATCH) * PATCH))


def load_recording(video: Path, n_frames: int, fps: float = 6.0) -> tuple[list[np.ndarray], list[int]]:
    frames = extract_frames(video, fps)
    if not frames:
        raise RuntimeError(f"no frames decoded from {video}")
    idx = select_keyframes(frames, n_frames)
    return [frames[i] for i in idx], idx
