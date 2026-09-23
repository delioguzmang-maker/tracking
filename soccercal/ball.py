"""Ball: pick one detection per frame with a gated nearest-neighbour tracker.

COCO "sports ball" from a general detector is the weakest link (small, blurred,
often in the air). The ground projection is exact only when the ball is on the
grass; aerial balls are flagged by their implausible jumps and dropped.
"""
from __future__ import annotations

import numpy as np

from .analysis import Analysis

BALL_R = 0.11


def track_ball(an: Analysis, cams: list, min_score: float = 0.15, max_speed: float = 35.0,
               max_gap_s: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pos (n,2) NaN when unknown, detected (n,) bool) on the analysed-frame grid."""
    n = len(an.frames)
    fps = an.proc_fps
    pos = np.full((n, 2), np.nan)
    det = np.zeros(n, bool)
    last, last_i = None, -10 ** 9
    for i, (fd, c) in enumerate(zip(an.frames, cams)):
        if c is None:
            continue
        b = fd.det.balls()
        if len(b) == 0:
            continue
        m = b.scores >= min_score
        if not m.any():
            continue
        boxes, sc = b.boxes[m], b.scores[m]
        ctr = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1)
        g, ok = c.image_to_plane(ctr, BALL_R)
        if not ok.any():
            continue
        g, sc = g[ok], sc[ok]
        dt = (i - last_i) / fps
        if last is not None and dt <= 1.0:
            d = np.linalg.norm(g - last, axis=1)
            gate = max_speed * dt + 1.0
            cand = np.nonzero(d <= gate)[0]
            if len(cand) == 0:
                if sc.max() < 0.5:
                    continue
                j = int(np.argmax(sc))
            else:
                j = int(cand[np.argmin(d[cand] - 2.0 * sc[cand])])
        else:
            j = int(np.argmax(sc))
        pos[i], det[i] = g[j], True
        last, last_i = g[j], i
    # fill short gaps linearly
    idx = np.nonzero(det)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap_s * fps:
            t = np.linspace(0, 1, b - a + 1)[1:-1, None]
            pos[a + 1:b] = pos[a] * (1 - t) + pos[b] * t
    return pos, det
