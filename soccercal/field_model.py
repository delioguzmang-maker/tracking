"""Pitch keypoint detection with the HRNet heatmap network (57 SoccerNet points).

Improvements over the reference decoders:

* sub-pixel peak localisation (log-parabola fit on the 3x3 neighbourhood) instead of
  the integer argmax on a half-resolution map (one heatmap cell = 4 px on 1080p);
* the decoding offset (label = floor(x/2)) is applied; measured on real footage it
  raises the line alignment from 0.68 to 0.76;
* keypoints close to the image border are kept (the calibration down-weights them);
* batched inference, CUDA / Apple MPS / CPU.

The NBJW *line* network is optional (``use_line_net``): on real broadcasts its
extremity channels fire all over the crowd and its "all lines" channel is a ~100 px
wide band, so precise line evidence comes from ``lines.LineMask`` instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .device import pick_device
from .hrnet import build_keypoint_net, build_line_net
from .weights import get_weights

NET_W, NET_H = 960, 540
KP_THRESHOLD = 0.1486  # tuned by NBJW on SoccerNet-calibration
LINE_THRESHOLD = 0.3880
# Heatmap cell u covers input pixels [2u, 2u+2): labels were drawn at floor(x/2).
HM_OFFSET = 0.5


@dataclass
class FieldObservation:
    """Detected pitch features in full-resolution image pixels."""
    width: int
    height: int
    keypoints: dict = field(default_factory=dict)  # id(1..57) -> (u, v, score)
    lines: dict = field(default_factory=dict)  # line index(0..22) -> ((u1, v1, s1), (u2, v2, s2))

    def n_features(self) -> int:
        return len(self.keypoints) + len(self.lines)


def _subpixel(hm: np.ndarray, y: int, x: int) -> tuple[float, float]:
    h, w = hm.shape
    dx = dy = 0.0
    eps = 1e-10
    if 0 < x < w - 1:
        l, c, r = np.log(hm[y, x - 1] + eps), np.log(hm[y, x] + eps), np.log(hm[y, x + 1] + eps)
        den = l - 2 * c + r
        if den < -1e-9:
            dx = float(np.clip(0.5 * (l - r) / den, -0.5, 0.5))
    if 0 < y < h - 1:
        t, c, b = np.log(hm[y - 1, x] + eps), np.log(hm[y, x] + eps), np.log(hm[y + 1, x] + eps)
        den = t - 2 * c + b
        if den < -1e-9:
            dy = float(np.clip(0.5 * (t - b) / den, -0.5, 0.5))
    return x + dx, y + dy


def decode_keypoints(hm: np.ndarray, threshold: float = KP_THRESHOLD, subpixel: bool = True):
    """hm: (57, h, w) probabilities -> {id: (x_hm, y_hm, score)} (heatmap coordinates)."""
    out = {}
    c, h, w = hm.shape
    flat = hm.reshape(c, -1)
    idx = flat.argmax(1)
    for k in range(c):
        s = float(flat[k, idx[k]])
        if s <= threshold:
            continue
        y, x = divmod(int(idx[k]), w)
        xs, ys = _subpixel(hm[k], y, x) if subpixel else (float(x), float(y))
        out[k + 1] = (xs, ys, s)
    return out


def decode_lines(hm: np.ndarray, threshold: float = LINE_THRESHOLD, min_dist: int = 10, subpixel: bool = True):
    """hm: (23, h, w) -> {line_idx: ((x1, y1, s1), (x2, y2, s2))}: the two extremities of each line."""
    out = {}
    c, h, w = hm.shape
    k = 2 * min_dist + 1
    for li in range(c):
        m = hm[li]
        if m.max() <= threshold:
            continue
        mp = cv2.dilate(m, np.ones((k, k), np.uint8))
        peaks = (m == mp) & (m > threshold)
        ys, xs = np.nonzero(peaks)
        if len(xs) < 2:
            continue
        order = np.argsort(-m[ys, xs])[:2]
        pts = []
        for o in order:
            px, py = _subpixel(m, ys[o], xs[o]) if subpixel else (float(xs[o]), float(ys[o]))
            pts.append((px, py, float(m[ys[o], xs[o]])))
        out[li] = (pts[0], pts[1])
    return out


class FieldDetector:
    """Runs the network(s) on BGR frames and returns a ``FieldObservation`` per frame."""

    def __init__(self, device: str | None = None, weights_dir: str | None = None, half: bool | None = None,
                 kp_threshold: float = KP_THRESHOLD, line_threshold: float = LINE_THRESHOLD, use_line_net: bool = False):
        """use_line_net: also run the NBJW line network (2x slower). Off by default: on real
        broadcasts its extremity maps are noisy; line pixels come from ``lines.LineMask``."""
        import torch

        self.torch = torch
        self.device = pick_device(device)
        self.kp_threshold = kp_threshold
        self.line_threshold = line_threshold
        self.half = (self.device == "cuda") if half is None else half
        self.kp_net = self._load(build_keypoint_net(), get_weights("SV_kp", weights_dir))
        self.line_net = self._load(build_line_net(), get_weights("SV_lines", weights_dir)) if use_line_net else None

    def _load(self, net, path):
        torch = self.torch
        sd = torch.load(str(path), map_location="cpu", weights_only=True)
        net.load_state_dict(sd, strict=True)
        net.eval().to(self.device)
        if self.half:
            net.half()
        return net

    def heatmaps(self, frames_bgr: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray | None]:
        torch = self.torch
        batch = []
        for fr in frames_bgr:
            if fr.shape[1] != NET_W or fr.shape[0] != NET_H:
                fr = cv2.resize(fr, (NET_W, NET_H), interpolation=cv2.INTER_AREA)
            batch.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float().div_(255.0).to(self.device)
        if self.half:
            x = x.half()
        with torch.inference_mode():
            kp = self.kp_net(x)[:, :-1].float().cpu().numpy()
            ln = self.line_net(x)[:, :-1].float().cpu().numpy() if self.line_net is not None else None
        return kp, ln

    # The HRNet head runs at half resolution with 784 channels: ~1 GB of activations per
    # 960x540 image. Small chunks keep a 16 GB Mac / a Colab T4 far from out-of-memory.
    max_batch = 2

    def __call__(self, frames_bgr: list[np.ndarray]) -> list[FieldObservation]:
        out = []
        for s in range(0, len(frames_bgr), self.max_batch):
            chunk = frames_bgr[s:s + self.max_batch]
            kp, ln = self.heatmaps(chunk)
            out += [self.decode(kp[i], None if ln is None else ln[i], chunk[i].shape[1], chunk[i].shape[0])
                    for i in range(len(chunk))]
        return out

    def decode(self, kp_hm: np.ndarray, line_hm: np.ndarray | None, width: int, height: int) -> FieldObservation:
        # Continuous "edge" pixel coordinates everywhere (0..W), like the camera model and YOLO boxes.
        hh, hw = kp_hm.shape[1:]
        sx, sy = width / (2.0 * hw), height / (2.0 * hh)  # net pixel -> image pixel

        def to_img(x, y):
            return 2 * (x + HM_OFFSET) * sx, 2 * (y + HM_OFFSET) * sy

        obs = FieldObservation(width, height)
        for k, (x, y, s) in decode_keypoints(kp_hm, self.kp_threshold).items():
            u, v = to_img(x, y)
            obs.keypoints[k] = (u, v, s)
        if line_hm is not None:
            for li, (a, b) in decode_lines(line_hm, self.line_threshold).items():
                obs.lines[li] = (to_img(a[0], a[1]) + (a[2],), to_img(b[0], b[1]) + (b[2],))
        return obs
