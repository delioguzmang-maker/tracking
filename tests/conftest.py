import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from soccercal.camera import Camera  # noqa: E402


def look_at(C, target, f, W=1920, H=1080) -> Camera:
    fwd = np.asarray(target, float) - np.asarray(C, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    return Camera(f=f, R=np.stack([right, np.cross(fwd, right), fwd]), C=np.asarray(C, float), width=W, height=H)


@pytest.fixture
def broadcast_cam() -> Camera:
    """A typical main-camera view of the left half."""
    return look_at([5.0, -55.0, 18.0], [-30.0, 2.0, 0.0], 2400.0)


def render_lines(cam: Camera, thickness: int = 3) -> np.ndarray:
    """Synthetic broadcast frame: striped grass + white pitch markings seen by ``cam``."""
    import cv2

    from soccercal import pitch

    img = np.zeros((cam.height, cam.width, 3), np.uint8)
    img[:] = (40, 140, 50)
    img[:, ::80] = (35, 125, 45)
    for pl in pitch.polylines():
        if np.abs(pl[:, 2]).max() > 0:
            continue
        dense = []
        for a, b in zip(pl[:-1], pl[1:]):
            t = np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / 0.3)))[:, None]
            dense.append(a[None] * (1 - t) + b[None] * t)
        uv, z = cam.project(np.concatenate(dense))
        ok = z > 0
        if ok.sum() > 1:
            cv2.polylines(img, [uv[ok].astype(np.int32)], False, (235, 235, 235), thickness, cv2.LINE_AA)
    return img
