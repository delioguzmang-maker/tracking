"""Resample identities onto the SkillCorner 10 fps grid, filling what the camera missed.

* inside a tracklet: RTS-smoothed position, ``is_detected = True``;
* short gaps (occlusions, < 2 s): cubic Hermite with the end velocities;
* longer gaps (player left the camera view and came back): linear path between
  exit and re-entry;
* before the first / after the last sighting: held where last seen (velocity decays
  with 0.5 s time constant), up to ``max_extrapolate_s``;
* everything filled is ``is_detected = False`` (SkillCorner's "extrapolated").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .players import Identity


DETECTED, SHORT_GAP, LONG_GAP, EDGE = 1, 2, 3, 4


@dataclass
class Sampled:
    t: np.ndarray  # (K,) seconds
    pos: np.ndarray  # (K,2) NaN when absent
    detected: np.ndarray  # (K,) bool
    kind: np.ndarray  # (K,) 0 absent, DETECTED, SHORT_GAP, LONG_GAP, EDGE


def _hermite(p0, v0, p1, v1, T, s):
    s2, s3 = s * s, s * s * s
    h00, h10, h01, h11 = 2 * s3 - 3 * s2 + 1, s3 - 2 * s2 + s, -2 * s3 + 3 * s2, s3 - s2
    return h00[:, None] * p0 + h10[:, None] * T * v0 + h01[:, None] * p1 + h11[:, None] * T * v1


def sample_identity(ident: Identity, frame_times: np.ndarray, out_t: np.ndarray, extrapolate: bool = True,
                    max_extrapolate_s: float = 15.0, hermite_max_s: float = 2.0, max_fill_s: float = 60.0,
                    motion=None, team_follow: float = 0.8) -> Sampled:
    K = len(out_t)
    pos = np.full((K, 2), np.nan)
    det = np.zeros(K, bool)
    kind = np.zeros(K, np.int8)
    # spans of the tracklets on the analysed grid
    spans = []
    for tl in ident.tracklets:
        m = (ident.frames >= tl.start) & (ident.frames <= tl.end)
        ft = frame_times[ident.frames[m]]
        spans.append((ft, ident.pos[m], ident.vel[m]))
    spans.sort(key=lambda s: s[0][0])
    for ft, p, _ in spans:
        m = (out_t >= ft[0] - 1e-6) & (out_t <= ft[-1] + 1e-6)
        if m.any():
            pos[m, 0] = np.interp(out_t[m], ft, p[:, 0])
            pos[m, 1] = np.interp(out_t[m], ft, p[:, 1])
            det[m] = True
            kind[m] = DETECTED
    if not extrapolate or not spans:
        return Sampled(out_t, pos, det, kind)
    for (fa, pa, va), (fb, pb, vb) in zip(spans[:-1], spans[1:]):
        t0, t1 = fa[-1], fb[0]
        m = (out_t > t0) & (out_t < t1)
        if not m.any() or t1 - t0 > max_fill_s:
            continue
        T = t1 - t0
        s = (out_t[m] - t0) / T
        if T <= hermite_max_s:
            pos[m] = _hermite(pa[-1], va[-1], pb[0], vb[0], T, s)
            kind[m] = SHORT_GAP
        else:
            pos[m] = pa[-1][None] * (1 - s[:, None]) + pb[0][None] * s[:, None]
            kind[m] = LONG_GAP
    tau = 0.5
    fps = 1.0 / np.median(np.diff(frame_times)) if len(frame_times) > 1 else 25.0

    def team_shift(t_from, t_to):
        # displacement of the player's team block (visible teammates) between two times
        if motion is None:
            return np.zeros((len(t_to), 2))
        f0 = int(round((t_from - frame_times[0]) * fps))
        f1 = np.round((t_to - frame_times[0]) * fps).astype(int)
        return team_follow * np.array([motion.shift(ident.team, f0, f) for f in f1]).reshape(-1, 2)

    ft, p, v = spans[0]
    m = (out_t < ft[0]) & (out_t >= ft[0] - max_extrapolate_s)
    dt = ft[0] - out_t[m]
    pos[m] = p[0][None] - v[0][None] * (tau * (1 - np.exp(-dt / tau)))[:, None] + team_shift(ft[0], out_t[m])
    kind[m] = EDGE
    ft, p, v = spans[-1]
    m = (out_t > ft[-1]) & (out_t <= ft[-1] + max_extrapolate_s)
    dt = out_t[m] - ft[-1]
    pos[m] = p[-1][None] + v[-1][None] * (tau * (1 - np.exp(-dt / tau)))[:, None] + team_shift(ft[-1], out_t[m])
    kind[m] = EDGE
    return Sampled(out_t, pos, det, kind)
