"""Ball: weak detections + geometry + a globally optimal trajectory.

A general-purpose detector sees a 5-10 px ball with low confidence (0.05) while
white boots, heads and advertising letters often score higher. So:

1. keep every weak "sports ball" candidate (``Config.ball_conf``);
2. geometric filters that only a calibrated camera allows: the candidate must be
   on the pitch, its size must match a 22 cm ball at that distance, and it must not
   sit on a player's head or shirt (boots are allowed but down-weighted: dribbling);
3. candidates are chained into constant-velocity tracklets (a real ball moves
   smoothly; false positives jump around), and the non-overlapping set of tracklets
   with the most evidence is chosen, each paying a price for starting, so isolated
   false positives are ignored and weak true detections are kept when they continue
   a trajectory;
4. short gaps are interpolated (``is_detected`` stays False there).

The ground projection is exact only for a ball on the grass; aerial balls give
far-away positions that the speed limit mostly rejects (their height is not
estimated).
"""
from __future__ import annotations

import numpy as np

from . import pitch
from .analysis import Analysis

BALL_R = 0.11
BALL_D = 0.22
_SPOTS = np.array([[0.0, 0.0], [-41.5, 0.0], [41.5, 0.0]])


def candidates(fd, cam, ball_conf: float = 0.03):
    """Filtered ball candidates of one frame: (xy (k,2), weight-adjusted log score (k,), uv (k,2))."""
    b = fd.det.balls()
    if cam is None or len(b) == 0:
        return np.zeros((0, 2)), np.zeros(0), np.zeros((0, 2))
    boxes, sc = b.boxes.astype(float), b.scores.astype(float)
    ctr = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1)
    xy, ok = cam.image_to_plane(ctr, BALL_R)
    ok &= pitch.inside_pitch(np.nan_to_num(xy, nan=1e6), 2.0)
    size = np.sqrt(np.maximum(boxes[:, 2] - boxes[:, 0], 1) * np.maximum(boxes[:, 3] - boxes[:, 1], 1))
    exp = np.full(len(sc), np.nan)
    exp[ok] = cam.pixel_height(xy[ok], BALL_D)
    ratio = size / exp
    ok &= (ratio > 0.35) & (ratio < 4.0)  # an aerial ball looks bigger than a ball on the grass there
    w = np.ones(len(sc))
    # Anything on a player (head, shirt, and above all white / neon boots) is rejected. A ball
    # at a player's feet is lost here, but then that player has it: the export puts the ball
    # at his position (possession), which is where it is.
    p = fd.det.persons()
    for pb in p.boxes:
        bw, h = pb[2] - pb[0], pb[3] - pb[1]
        body = (ctr[:, 0] > pb[0]) & (ctr[:, 0] < pb[2]) & (ctr[:, 1] > pb[1]) & (ctr[:, 1] < pb[1] + 0.7 * h)
        # boots stick out of the box when running: widen the feet zone
        feet = (ctr[:, 0] > pb[0] - 0.15 * bw) & (ctr[:, 0] < pb[2] + 0.15 * bw) & \
               (ctr[:, 1] >= pb[1] + 0.7 * h) & (ctr[:, 1] < pb[3] + 0.06 * h)
        ok &= ~(body | feet)
    sv = getattr(fd, "ball_sv", None)
    if sv is not None and len(sv) == len(sc):
        ok &= sv[:, 0] <= 110  # a ball is (mostly) white: very saturated blobs are boots, logos, bibs
    if ok.any():
        d = np.min(np.linalg.norm(np.nan_to_num(xy, nan=1e6)[:, None] - _SPOTS[None], axis=2), 1)
        w[d < 0.5] *= 0.5  # painted spots look like a ball
    e = w * np.log(np.maximum(sc, 1e-4) / 0.06)
    return xy[ok], e[ok], ctr[ok]


def _ball_tracklets(C, t, fps, vmax, max_miss_s):
    """Constant-velocity tracklets through the candidates (greedy association to the prediction)."""
    tracks, active = [], []
    for i, (xy, e, _) in enumerate(C):
        used = np.zeros(len(e), bool)
        # extend active tracks, best-predicted first
        for tr in sorted(active, key=lambda tr: -tr["score"]):
            j_last = tr["frames"][-1]
            dt = t[i] - t[j_last]
            p = tr["pos"][-1] + tr["v"] * dt
            if tr["v_known"]:
                gate = 1.0 + 0.35 * np.linalg.norm(tr["v"]) * dt + 6.0 * dt
            else:
                gate = vmax * dt + 1.0
            if len(e):
                d = np.linalg.norm(xy - p, axis=1)
                d[used] = np.inf
                k = int(np.argmin(d))
                if d[k] <= gate:
                    used[k] = True
                    v_new = (xy[k] - tr["pos"][-1]) / max(dt, 1e-3)
                    tr["v"] = v_new if not tr["v_known"] else 0.5 * tr["v"] + 0.5 * v_new
                    tr["v_known"] = True
                    tr["frames"].append(i)
                    tr["pos"].append(xy[k])
                    tr["score"] += e[k]
        # retire tracks not seen for too long
        still = []
        for tr in active:
            if t[i] - t[tr["frames"][-1]] > max_miss_s:
                tracks.append(tr)
            else:
                still.append(tr)
        active = still
        for k in np.nonzero(~used)[0]:
            active.append({"frames": [i], "pos": [xy[k]], "v": np.zeros(2), "v_known": False, "score": float(e[k])})
    return tracks + active


def track_ball(an: Analysis, cams: list, ball_conf: float = 0.03, vmax: float = 40.0, max_miss_s: float = 0.4,
               restart_cost: float = 1.0, interp_max_s: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pos (n,2) NaN when unknown, detected (n,) bool) on the analysed-frame grid.

    Candidates are chained into constant-velocity tracklets (a real ball moves smoothly,
    false positives jump), then the set of non-overlapping tracklets with the highest total
    evidence is chosen (weighted interval scheduling); each tracklet pays ``restart_cost``,
    so isolated weak detections are not trusted.
    """
    n = len(an.frames)
    fps = an.proc_fps
    t = np.array([fd.t for fd in an.frames])
    C = [candidates(fd, c, ball_conf) for fd, c in zip(an.frames, cams)]
    trs = [tr for tr in _ball_tracklets(C, t, fps, vmax, max_miss_s)
           if tr["score"] - restart_cost > 0 and len(tr["frames"]) >= 2]
    trs.sort(key=lambda tr: tr["frames"][-1])
    ends = [tr["frames"][-1] for tr in trs]
    best = np.zeros(len(trs) + 1)
    take = np.zeros(len(trs), bool)
    prev = np.zeros(len(trs), int)
    for k, tr in enumerate(trs):
        p = int(np.searchsorted(ends, tr["frames"][0], side="left"))  # tracklets ending before this starts
        prev[k] = p
        w = tr["score"] - restart_cost
        if best[p] + w > best[k]:
            best[k + 1], take[k] = best[p] + w, True
        else:
            best[k + 1] = best[k]
    pos = np.full((n, 2), np.nan)
    det = np.zeros(n, bool)
    k = len(trs) - 1
    while k >= 0:
        if take[k]:
            for f, p in zip(trs[k]["frames"], trs[k]["pos"]):
                pos[f], det[f] = p, True
            k = prev[k] - 1
        else:
            k -= 1
    # interpolate short gaps between chosen detections
    idx = np.nonzero(det)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > 1 and t[b] - t[a] <= interp_max_s:
            s = ((t[a + 1:b] - t[a]) / (t[b] - t[a]))[:, None]
            pos[a + 1:b] = pos[a] * (1 - s) + pos[b] * s
    return pos, det
