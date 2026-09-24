"""Offline re-identification: link tracklets of the same player across gaps.

Appearance re-identification is weak in football (11 identical shirts, low
resolution, numbers rarely readable). What *is* strong, and only exists because
the camera is calibrated, is kinematics in metres: players cannot teleport, the
velocity at the end of a tracklet predicts where the player re-appears, and while
a player is off-screen his whole team keeps moving as a block, which we can
measure from the teammates that *are* on screen. So the link cost is kinematic
(team-motion compensated) first, jersey colour second, and a team mismatch forbids
the link.

All links are chosen jointly (optimal assignment: every tracklet has at most one
predecessor and one successor), and a link is only kept if it is unambiguous:
a wrong merge corrupts two players' data, a missed merge only splits one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import linear_sum_assignment

from .teams import OTHER


@dataclass
class Tracklet:
    tid: int
    frames: np.ndarray  # (N,) analysed-frame indices of the observations
    z: np.ndarray  # (N,2) pitch measurements
    R: np.ndarray  # (N,2,2)
    scores: np.ndarray
    det_idx: np.ndarray  # (N,) row in that frame's Measurement
    desc: np.ndarray | None = None  # mean jersey descriptor (unit)
    team: int = OTHER
    team_conf: float = 0.0
    outside: float = 0.0  # share of positions outside the lines (staff-like when high)
    jersey_reads: list = field(default_factory=list)  # [(number, confidence)] from OCR
    number: int | None = None  # voted shirt number
    # filled by ``endpoints``
    p0: np.ndarray = field(default_factory=lambda: np.zeros(2))
    v0: np.ndarray = field(default_factory=lambda: np.zeros(2))
    p1: np.ndarray = field(default_factory=lambda: np.zeros(2))
    v1: np.ndarray = field(default_factory=lambda: np.zeros(2))

    @property
    def start(self) -> int:
        return int(self.frames[0])

    @property
    def end(self) -> int:
        return int(self.frames[-1])


def endpoints(tl: Tracklet, fps: float, window_s: float = 0.6) -> None:
    """Robust position/velocity at both ends from a line fit on the end window."""
    f = tl.frames.astype(float) / fps
    for side in (0, 1):
        m = f <= f[0] + window_s if side == 0 else f >= f[-1] - window_s
        t, z = f[m], tl.z[m]
        t0 = f[0] if side == 0 else f[-1]
        if len(t) >= 3 and np.ptp(t) > 0.15:
            A = np.stack([np.ones_like(t), t - t0], 1)
            coef, *_ = np.linalg.lstsq(A, z, rcond=None)
            p, v = coef[0], coef[1]
            sp = np.linalg.norm(v)
            if sp > 10.0:  # physically impossible sustained speed -> distrust the slope
                v = v / sp * 10.0
        else:
            p, v = (z[0] if side == 0 else z[-1]), np.zeros(2)
        if side == 0:
            tl.p0, tl.v0 = p, v
        else:
            tl.p1, tl.v1 = p, v


class TeamMotion:
    """Average velocity of the on-screen players of each team over time.

    While a player is off-screen his team keeps shifting as a block (pressing up,
    dropping back); integrating the visible teammates' mean velocity over the gap
    predicts most of that displacement."""

    def __init__(self, tracklets: list[Tracklet], fps: float, n_frames: int):
        self.fps = fps
        self.v = {t: np.zeros((n_frames, 2)) for t in (0, 1, OTHER)}
        cnt = {t: np.zeros(n_frames) for t in (0, 1, OTHER)}
        for tl in tracklets:
            if len(tl.frames) < 5:
                continue
            f = tl.frames
            vel = np.gradient(tl.z, f / fps, axis=0)
            vel = np.clip(uniform_filter1d(vel, size=max(1, int(0.5 * fps)), axis=0, mode="nearest"), -9, 9)
            self.v[tl.team][f] += vel
            cnt[tl.team][f] += 1
        for t in self.v:
            ok = cnt[t] > 0
            self.v[t][ok] /= cnt[t][ok, None]
        self.cum = {t: np.vstack([np.zeros((1, 2)), np.cumsum(self.v[t], 0) / fps]) for t in self.v}

    def shift(self, team: int, f0: int, f1: int) -> np.ndarray:
        if team not in self.cum:
            return np.zeros(2)
        c = self.cum[team]
        f0, f1 = np.clip([f0, f1], 0, len(c) - 1)
        return c[f1] - c[f0]


@dataclass
class StitchConfig:
    vmax: float = 9.0  # m/s, hard gate on (team-compensated) displacement / time
    slack: float = 2.0  # m, position noise allowance
    sigma_v: float = 1.2  # m/s, growth of the prediction uncertainty with the gap
    horizon: float = 1.0  # s, own-velocity extrapolation horizon
    team_follow: float = 0.8  # fraction of the visible team's motion applied off-screen
    max_gap_s: float = 60.0
    app_weight: float = 0.6
    max_cost: float = 1.0
    ambiguity_ratio: float = 1.1  # a rival within ratio x best cost that prefers the link vetoes it
    min_team_conf: float = 0.5


def link_cost(a: Tracklet, b: Tracklet, fps: float, cfg: StitchConfig, motion: TeamMotion | None = None) -> float:
    """Cost of 'b is the continuation of a' (inf if impossible)."""
    dt = (b.start - a.end) / fps
    if dt <= 0 or dt > cfg.max_gap_s:
        return np.inf
    if a.team != OTHER and b.team != OTHER and a.team != b.team and \
            min(a.team_conf, b.team_conf) >= cfg.min_team_conf:
        return np.inf
    if (a.outside >= 0.6 and b.outside <= 0.2) or (b.outside >= 0.6 and a.outside <= 0.2):
        return np.inf  # a touchline official / coach never becomes a player
    same_number = a.number is not None and a.number == b.number
    if a.number is not None and b.number is not None and not same_number:
        return np.inf  # two different shirt numbers: two different players
    h = min(dt, cfg.horizon)
    drift = np.zeros(2)
    if motion is not None and dt > cfg.horizon:
        # after the own-velocity horizon the player moves with his team block
        drift = cfg.team_follow * motion.shift(a.team, a.end + int(h * fps), b.start)
    pred_a = a.p1 + a.v1 * h + drift
    pred_b = b.p0 - b.v0 * h - drift
    if np.linalg.norm(b.p0 - a.p1 - drift) > cfg.vmax * dt + cfg.slack:
        return np.inf
    err = 0.5 * (np.linalg.norm(pred_a - b.p0) + np.linalg.norm(pred_b - a.p1))
    scale = cfg.slack + cfg.sigma_v * dt
    c = err / scale
    if same_number:
        c *= 0.3  # the shirt number says it is the same player
    if a.desc is not None and b.desc is not None:
        c += cfg.app_weight * float(1.0 - np.clip(a.desc @ b.desc, 0, 1))
    return float(c)


def stitch(tracklets: list[Tracklet], fps: float, cfg: StitchConfig | None = None,
           n_frames: int | None = None, motion: TeamMotion | None = None) -> list[list[Tracklet]]:
    """Group tracklets into identities (lists of tracklets ordered in time)."""
    cfg = cfg or StitchConfig()
    if not tracklets:
        return []
    for t in tracklets:
        endpoints(t, fps)
    n_frames = n_frames or (max(t.end for t in tracklets) + 1)
    motion = motion or TeamMotion(tracklets, fps, n_frames)
    order = sorted(tracklets, key=lambda t: t.start)
    n = len(order)
    BIG = 1e6
    C = np.full((n, n), BIG)
    starts = np.array([t.start for t in order])
    for i, a in enumerate(order):
        lo = np.searchsorted(starts, a.end + 1)
        hi = np.searchsorted(starts, a.end + cfg.max_gap_s * fps, side="right")
        for j in range(lo, hi):
            c = link_cost(a, order[j], fps, cfg, motion)
            if c <= cfg.max_cost:
                C[i, j] = c
    # rows: predecessors; columns: successors + one private "no successor" option per row
    D = np.full((n, n), BIG)
    np.fill_diagonal(D, cfg.max_cost)
    r, c = linear_sum_assignment(np.hstack([C, D]))
    match_of_row = {i: j for i, j in zip(r, c) if j < n and C[i, j] < BIG}
    match_of_col = {j: i for i, j in match_of_row.items()}
    succ, pred = {}, {}
    for i, j in match_of_row.items():
        lim = cfg.ambiguity_ratio * C[i, j]
        # Ambiguity test against *real* rivals only: another predecessor that would rather
        # continue into j than into its own match, or another successor that would rather
        # follow i. (Earlier tracklets of the same player, already linked, are not rivals.)
        rivals = False
        for i2 in np.nonzero(C[:, j] < lim)[0]:
            if i2 != i and (i2 not in match_of_row or C[i2, match_of_row[i2]] > C[i2, j]):
                rivals = True
                break
        if not rivals:
            for j2 in np.nonzero(C[i] < lim)[0]:
                if j2 != j and (j2 not in match_of_col or C[match_of_col[j2], j2] > C[i, j2]):
                    rivals = True
                    break
        if rivals and C[i, j] > 0.15:
            continue
        succ[order[i].tid], pred[order[j].tid] = order[j].tid, order[i].tid
    by_id = {t.tid: t for t in tracklets}
    chains = []
    for t in order:
        if t.tid in pred:
            continue
        chain = [t]
        while chain[-1].tid in succ:
            chain.append(by_id[succ[chain[-1].tid]])
        chains.append(chain)
    return chains
