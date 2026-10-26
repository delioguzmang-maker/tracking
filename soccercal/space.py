"""Space creation (Fernández & Bornn, "Wide Open Spaces", MIT Sloan 2018), sections 4-5.

* ``pitch_value``: how valuable each location is given where the ball is. As in the paper, a
  feed-forward network with one hidden layer learns, from many defensive situations, the summed
  influence the defending team puts on every location (``min(sum_d I_d, 1)``, a 21 x 15 grid)
  given the ball position: defenders cover the valuable space. It is then multiplied by a
  distance-to-goal surface, so space near the opponent's goal is worth more. The paper's network
  was trained on 20 private Metrica matches; ours (``tools/train_pitch_value.py``) on open
  tracking data (SkillCorner, IDSSE, Metrica); the weights are in ``data/pitch_value.npz``.
* ``space_quality``: the quality of each player's owned space, ``Q_i = PC_i x V`` (eq. 6): team
  control (``tactics.space_control``) shared among teammates in proportion to their influence,
  times the pitch value, averaged over the pitch.
* ``space_occupation``: Space Occupation Gain / Loss (eqs. 7-9): the mean change of ``Q_i`` over a
  window ``w`` (3 s) after each controlled possession of the team, above a threshold; *active* when
  the player moves faster than a jog (1.5 m/s), *passive* otherwise.
* ``space_generation``: Space Generation Gain (eqs. 10-11): a player drags an opponent who was
  marking a teammate (within 5 m at the start, within 5 m of the dragger and > 5 m from the
  teammate at the end, after moving at least 3 m), and the teammate gains space.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .tactics import Match, Snapshot, grid, influence

WEIGHTS = Path(__file__).with_name("data") / "pitch_value.npz"
_W = None


def _weights():
    global _W
    if _W is None:
        _W = dict(np.load(WEIGHTS))
    return _W


def _mlp(Xin: np.ndarray, w: dict) -> np.ndarray:
    h = 1.0 / (1.0 + np.exp(-(Xin @ w["W1"] + w["b1"])))
    return 1.0 / (1.0 + np.exp(-(h @ w["W2"] + w["b2"])))[:, 0]


def value_inputs(ball: np.ndarray, P: np.ndarray, pitch=(105.0, 68.0)) -> np.ndarray:
    """Network inputs (attacking towards +x): ball and location, scaled to [-1, 1]."""
    L, W = pitch
    b = np.broadcast_to(np.asarray(ball, float), P.shape)
    return np.column_stack([b[:, 0] / (L / 2), b[:, 1] / (W / 2), P[:, 0] / (L / 2), P[:, 1] / (W / 2)])


def goal_surface(P: np.ndarray, pitch=(105.0, 68.0)) -> np.ndarray:
    """Distance to the opponent's goal (attacking +x), normalised to [0, 1]: 1 at the goal."""
    L, W = pitch
    d = np.hypot(P[:, 0] - L / 2, P[:, 1])
    return 1.0 - d / np.hypot(L, W / 2)


def pitch_value(ball, P: np.ndarray, attacks_right: bool = True, pitch=(105.0, 68.0), normalise: bool = True,
                weights: dict | None = None) -> np.ndarray:
    """Value in [0, 1] of locations ``P`` (m, 2) for the team attacking towards +x (``attacks_right``)
    or -x, with the ball at ``ball``. ``normalise`` multiplies by the distance-to-goal surface."""
    s = 1.0 if attacks_right else -1.0
    P2 = np.asarray(P, float) * s
    b = np.asarray(ball, float) * s
    v = _mlp(value_inputs(b, P2, pitch), weights or _weights())
    return v * goal_surface(P2, pitch) if normalise else v


def space_quality(s: Snapshot, team: int, n_grid_cells_x: int = 52, weights: dict | None = None) -> dict:
    """{player id: Q_i} for the players of ``team`` (eq. 6)."""
    xg, yg = grid(n_grid_cells_x, s.pitch)
    P = np.stack(np.meshgrid(xg, yg), -1).reshape(-1, 2)
    ok = np.isfinite(s.pos).all(1)
    ball = s.ball if np.isfinite(s.ball).all() else s.pos[ok].mean(0)
    I = influence(s.pos[ok], s.vel[ok], ball, P)
    mine = s.team[ok] == team
    pc = 1.0 / (1.0 + np.exp(-(I[mine].sum(0) - I[~mine].sum(0))))
    share = I[mine] / np.maximum(I[mine].sum(0), 1e-9)
    V = pitch_value(ball, P, s.attacks_right[team], s.pitch, weights=weights)
    Q = (share * pc * V).mean(1)
    return dict(zip(s.pid[ok][mine].tolist(), Q.tolist()))


def _possession_starts(match: Match, team: int) -> np.ndarray:
    """Frames where a player of ``team`` starts a controlled possession."""
    b = match.ball
    pid = b["possession_player_id"].values
    tm = b["possession_team"].values
    start = (pid >= 0) & (tm == team) & (np.r_[-1, pid[:-1]] != pid)
    return b.index.values[start]


def space_occupation(match: Match, team: int, window_s: float = 3.0, eps: float = 0.0, frames=None,
                     jog: float = 1.5) -> pd.DataFrame:
    """Space Occupation Gain / Loss of every player of ``team`` after each of its possessions (or at
    ``frames``). One row per (frame, player): ``gain`` = mean Q(t') - Q(t) over (t, t + w]; ``kind``
    is "SOG" (gain >= eps), "SOL" (gain <= -eps) or ""; ``active`` = moving faster than a jog;
    ``in_front`` = ahead of the ball; ``dist_ball``."""
    frames = _possession_starts(match, team) if frames is None else np.asarray(frames)
    w = int(round(window_s * match.fps))
    have = set(match.frames.tolist())
    rows = []
    cache: dict = {}

    def Q(k):
        if k not in cache:
            cache[k] = space_quality(match.snapshot(int(k)), team) if k in have else {}
        return cache[k]
    for t0 in frames:
        q0 = Q(t0)
        later = [Q(t0 + d) for d in range(1, w + 1)]
        s0 = match.snapshot(int(t0))
        right = s0.attacks_right[team]
        g = match.table[(match.table["frame"] > t0) & (match.table["frame"] <= t0 + w) & (match.table["team"] == team)]
        speed = g.assign(v=np.hypot(g["vx"], g["vy"])).groupby("player_id")["v"].mean()
        for pid, q in q0.items():
            diffs = [qq[pid] - q for qq in later if pid in qq]
            if len(diffs) < w // 2:
                continue
            G = float(np.mean(diffs))
            j = np.nonzero(s0.pid == pid)[0][0]
            ahead = (s0.pos[j, 0] - s0.ball[0]) * (1 if right else -1) > 0 if np.isfinite(s0.ball).all() else np.nan
            rows.append({"frame": int(t0), "t": t0 / match.fps, "player_id": int(pid), "gain": G,
                         "kind": "SOG" if G >= eps and G > 0 else "SOL" if G <= -eps and G < 0 else "",
                         "active": bool(speed.get(pid, 0.0) > jog), "in_front": ahead,
                         "dist_ball": float(np.hypot(*(s0.pos[j] - s0.ball))) if np.isfinite(s0.ball).all() else np.nan})
    return pd.DataFrame(rows)


def space_generation(match: Match, team: int, window_s: float = 3.0, close: float = 5.0, attract: float = 3.0,
                     eps: float = 0.0, frames=None) -> pd.DataFrame:
    """Space Generation Gain: ``generator`` drags ``opponent`` away from ``receiver``, who gains space
    (eqs. 10-11; ``close`` = lambda, ``attract`` = kappa as the opponent's displacement)."""
    occ = space_occupation(match, team, window_s, eps, frames)
    if not len(occ):
        return pd.DataFrame(columns=["frame", "generator", "receiver", "opponent", "gain"])
    gain = {(r.frame, r.player_id): r.gain for r in occ.itertuples()}
    w = int(round(window_s * match.fps))
    rows = []
    for t0 in sorted(occ["frame"].unique()):
        a, b = match.snapshot(int(t0)), match.snapshot(int(t0 + w))
        pa = {p: x for p, x in zip(a.pid, a.pos)}
        pb = {p: x for p, x in zip(b.pid, b.pos)}
        mates = [p for p, tm in zip(a.pid, a.team) if tm == team and p in pb]
        opps = [p for p, tm in zip(a.pid, a.team) if tm != team and p in pb]
        for j in opps:
            if np.linalg.norm(pb[j] - pa[j]) < attract:
                continue
            for r in mates:  # receiver: marked by j at the start, free of him at the end
                if np.linalg.norm(pa[r] - pa[j]) > close or np.linalg.norm(pb[r] - pb[j]) <= close:
                    continue
                for g in mates:  # generator: has j close at the end
                    if g == r or np.linalg.norm(pb[g] - pb[j]) > close:
                        continue
                    G = gain.get((t0, r), np.nan)
                    if np.isfinite(G) and G >= eps and G > 0:
                        rows.append({"frame": int(t0), "t": t0 / match.fps, "generator": int(g), "receiver": int(r),
                                     "opponent": int(j), "gain": G})
    return pd.DataFrame(rows)


def summary(match: Match, team: int, window_s: float = 3.0, eps: float = 0.0) -> pd.DataFrame:
    """Per player, as in the paper's Table 1-2: SOG total and count, % active, in front / behind the
    ball, mean distance to the ball; SOL; SGG generated."""
    occ = space_occupation(match, team, window_s, eps)
    gen = space_generation(match, team, window_s, eps=eps)
    rows = []
    for pid, g in occ.groupby("player_id"):
        sog = g[g["kind"] == "SOG"]
        sol = g[g["kind"] == "SOL"]
        sg = gen[gen["generator"] == pid] if len(gen) else gen
        rows.append({"player_id": pid, "number": match.numbers.get(pid), "SOG": sog["gain"].sum(), "n_SOG": len(sog),
                     "pct_active": 100 * sog["active"].mean() if len(sog) else np.nan,
                     "n_front": int(sog["in_front"].eq(True).sum()), "n_behind": int(sog["in_front"].eq(False).sum()),
                     "dist_ball": sog["dist_ball"].mean(), "SOL": -sol["gain"].sum(), "SGG": sg["gain"].sum() if len(sg) else 0.0,
                     "n_SGG": len(sg)})
    out = pd.DataFrame(rows)
    return out.sort_values("SOG", ascending=False).reset_index(drop=True) if len(out) else out
