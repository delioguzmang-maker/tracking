"""Pitch-line pixels from the image, for dense (chamfer) camera refinement and QA.

The NBJW line network localises line *extremities* poorly on real broadcasts (its
"all lines" channel is a ~100 px wide band), so the precise line evidence comes from
classic image processing: white, thin, bright-on-grass structures (morphological
top-hat) inside the grass region. Refining the camera so that every projected
pitch marking falls on those pixels uses thousands of constraints instead of ~10
keypoints.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from . import pitch

MASK_W = 960  # line masks are stored at this width (half of 1080p)


def _model_points(step: float = 0.4) -> np.ndarray:
    pts = []
    for pl in pitch.polylines():
        if np.abs(pl[:, 2]).max() > 0:  # goal frames stand above the grass mask
            continue
        for a, b in zip(pl[:-1], pl[1:]):
            n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
            t = np.linspace(0, 1, n)[:, None]
            pts.append(a[None] * (1 - t) + b[None] * t)
    return np.unique(np.round(np.concatenate(pts), 3), axis=0)


MODEL_POINTS = _model_points()


def grass_mask(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    g = ((hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 40)).astype(np.uint8)
    k = max(3, int(round(frame_bgr.shape[1] / 77)) | 1)
    return cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))


@dataclass
class LineMask:
    mask: np.ndarray  # uint8 {0,1} at MASK_W width
    scale: float  # mask pixels per image pixel
    _dt: np.ndarray | None = None

    @staticmethod
    def from_frame(frame_bgr: np.ndarray, boxes: np.ndarray | None = None) -> "LineMask":
        s = MASK_W / frame_bgr.shape[1]
        small = cv2.resize(frame_bgr, (MASK_W, int(round(frame_bgr.shape[0] * s))), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
        m = ((th > 20) & (grass_mask(small) > 0)).astype(np.uint8)
        if boxes is not None:  # white kit / socks are not pitch lines
            for x1, y1, x2, y2 in np.asarray(boxes).reshape(-1, 4) * s:
                m[max(0, int(y1)):int(y2) + 1, max(0, int(x1)):int(x2) + 1] = 0
        return LineMask(m, s)

    # compact storage (a few KB per frame)
    def pack(self) -> tuple:
        return (self.mask.shape, self.scale, zlib.compress(np.packbits(self.mask).tobytes(), 6))

    @staticmethod
    def unpack(p: tuple) -> "LineMask":
        shape, scale, data = p
        bits = np.unpackbits(np.frombuffer(zlib.decompress(data), np.uint8))[: shape[0] * shape[1]]
        return LineMask(bits.reshape(shape).astype(np.uint8), scale)

    @property
    def dt(self) -> np.ndarray:
        """Distance (in *image* pixels) to the nearest line pixel."""
        if self._dt is None:
            self._dt = cv2.distanceTransform((1 - self.mask).astype(np.uint8), cv2.DIST_L2, 5) / self.scale
        return self._dt

    def distances(self, uv: np.ndarray, cap: float) -> np.ndarray:
        """Chamfer distance at image points (N,2); points outside the image get their
        outside distance added; everything capped at ``cap`` (robust truncation)."""
        h, w = self.mask.shape
        u = uv[:, 0] * self.scale
        v = uv[:, 1] * self.scale
        uc, vc = np.clip(u, 0, w - 1), np.clip(v, 0, h - 1)
        d = map_coordinates(self.dt, [vc, uc], order=1, mode="nearest")
        d = d + (np.abs(u - uc) + np.abs(v - vc)) / self.scale
        return np.minimum(np.nan_to_num(d, nan=cap), cap)


def visible_model_points(cam, margin: float = 2.0) -> np.ndarray:
    uv, z = cam.project(MODEL_POINTS)
    ok = (z > 0) & (uv[:, 0] >= margin) & (uv[:, 0] < cam.width - margin) & \
         (uv[:, 1] >= margin) & (uv[:, 1] < cam.height - margin)
    return MODEL_POINTS[ok]


def alignment(cam, lm: LineMask, tol_px: float = 3.0) -> float:
    """Fraction of visible pitch-marking points that land within ``tol_px`` (1080p pixels)
    of a detected line pixel. GT-free calibration quality: ~0.8 is excellent on real
    broadcast (players and worn paint hide part of the lines), < 0.4 is a bad calibration."""
    P = visible_model_points(cam, margin=5)
    if len(P) < 20:
        return 0.0
    uv, _ = cam.project(P)
    tol = tol_px * cam.width / 1920.0
    return float((lm.distances(uv, cap=10 * tol) <= tol).mean())
