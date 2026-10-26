"""Players the video never shows: an estimated position, learned from open tracking data.

A broadcast clip often never shows a goalkeeper or a winger on the far side, so that team has fewer
than 11 players and every space model hands the empty area to the opponent. ``complete`` adds
them, flagged (``imputed`` True, ``is_detected`` False, negative ids), for the tactical models only;
the tracking files keep only what was seen or followed.

* **Goalkeeper**: a small network (``data/impute.npz``) predicts where a team's goalkeeper stands
  from the ball, the team's shape (centroid and last defender) and who has the ball; trained on
  SkillCorner, IDSSE and Metrica tracking (``tools/train_impute.py``, error in the README).
* **Outfield players**: a small set network looks at where the seen teammates stand relative to
  their team (average over the clip, in units of the team's spread) and predicts the hole: where the
  missing player stands relative to them; it is then placed frame by frame with the team's centre and
  spread. Trained on real teams with one player hidden; a library of real formations (matching the
  observed shape with one place left free) is kept as the fallback and as a baseline.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .tactics import Match

MODEL = Path(__file__).with_name("data") / "impute.npz"
_M = None


def _model():
    global _M
    if _M is None:
        _M = dict(np.load(MODEL, allow_pickle=False))
    return _M


# ----------------------------------------------------------------------------------- goalkeeper
def gk_features(ball, own_poss, centroid_x, last_x, pitch=(105.0, 68.0)) -> np.ndarray:
    """Inputs, for a team defending the -x goal: ball x, y; possession (+1 own, -1 opponent, 0
    unknown); outfield centroid x; last defender x; all scaled by the pitch half-sizes."""
    L, W = pitch
    b = np.atleast_2d(ball).astype(float)
    return np.column_stack([b[:, 0] / (L / 2), b[:, 1] / (W / 2), np.asarray(own_poss, float),
                            np.asarray(centroid_x, float) / (L / 2), np.asarray(last_x, float) / (L / 2)])


def gk_predict(F: np.ndarray, pitch=(105.0, 68.0), m: dict | None = None) -> np.ndarray:
    """(n, 2) goalkeeper positions (team defending -x) from ``gk_features``."""
    m = m or _model()
    h = np.tanh(F @ m["gk_W1"] + m["gk_b1"])
    out = h @ m["gk_W2"] + m["gk_b2"]
    L, W = pitch
    return np.column_stack([out[:, 0] * L / 2, out[:, 1] * W / 2])


# ------------------------------------------------------------------------------------- outfield
def shape(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centre and spread (std along x and y, at least 3 m) of a team's outfield players (n, 2)."""
    return P.mean(0), np.maximum(P.std(0), 3.0)


def match_template(observed: np.ndarray, templates: np.ndarray, n_missing: int, top: int = 5) -> tuple[np.ndarray, float]:
    """``observed`` (k, 2): average normalised places of the k seen outfield players; ``templates``
    (T, 10, 2): library formations. Every template, with each choice of ``n_missing`` places left
    free, is compared with the observed shape (both re-normalised to zero mean and unit spread,
    places paired by the Hungarian algorithm). Returns the free places of the best matches (the
    ``top`` best averaged when one player is missing), in the observed players' units, and the cost."""
    co, so = observed.mean(0), np.maximum(observed.std(0), 1e-6)
    obs = (observed - co) / so
    found = []
    for tpl in templates:
        for gone in combinations(range(len(tpl)), n_missing):
            keep = np.setdiff1d(np.arange(len(tpl)), gone)
            c, s = tpl[keep].mean(0), np.maximum(tpl[keep].std(0), 1e-6)
            cost = ((obs[:, None] - ((tpl[keep] - c) / s)[None]) ** 2).sum(-1)
            r, q = linear_sum_assignment(cost)
            found.append((cost[r, q].sum() / len(obs), (tpl[list(gone)] - c) / s))
    found.sort(key=lambda f: f[0])
    best = found[:top] if n_missing == 1 else found[:1]
    w = np.exp(-np.array([f[0] for f in best]) / max(best[0][0], 1e-3))
    slots = sum(wi * f[1] for wi, f in zip(w, best)) / w.sum()
    return slots * so + co, best[0][0]


def set_predict(obs: np.ndarray, m: dict | None = None) -> np.ndarray:
    """Normalised place of the missing outfield player from the seen players' average normalised
    places ``obs`` (k, 2): a set network (shared per-player encoder, self-attention, mean and max
    pooling) trained on real teams with one player hidden (``tools/train_impute.py``)."""
    m = m or _model()
    relu = lambda x: np.maximum(x, 0.0)  # noqa: E731
    e = relu(relu(obs @ m["set_phi0_w"].T + m["set_phi0_b"]) @ m["set_phi2_w"].T + m["set_phi2_b"])
    h = e.shape[1]
    nh = int(m["set_heads"])
    q, k, v = np.split(e @ m["set_in_w"].T + m["set_in_b"], 3, axis=1)
    d = h // nh
    heads = []
    for i in range(nh):
        qi, ki, vi = (z[:, i * d:(i + 1) * d] for z in (q, k, v))
        a = qi @ ki.T / np.sqrt(d)
        a = np.exp(a - a.max(1, keepdims=True))
        heads.append((a / a.sum(1, keepdims=True)) @ vi)
    att = np.concatenate(heads, 1) @ m["set_out_w"].T + m["set_out_b"]
    z = np.concatenate([att.mean(0), att.max(0)])
    return relu(z @ m["set_rho0_w"].T + m["set_rho0_b"]) @ m["set_rho2_w"].T + m["set_rho2_b"]


def complete(match: Match, players_per_team: int = 11, min_seen_frac: float = 0.0) -> Match:
    """A copy of ``match`` with the players it lacks added at estimated positions (``imputed``)."""
    tab = match.table.copy()
    tab["imputed"] = False
    mdl = _model()
    templates = mdl["templates"]
    L, W = match.pitch_length, match.pitch_width
    new = []
    frames = match.frames
    for team in (0, 1):
        g = tab[tab["team"] == team]
        n_gk = int(g.loc[g["gk"], "player_id"].nunique())
        out_ids = g.loc[~g["gk"], "player_id"].unique()
        n_missing_out = max(0, players_per_team - 1 - len(out_ids))
        by_f = {k: v for k, v in g[~g["gk"]].groupby("frame")}
        # ------------------------------------------------ missing outfield players
        if n_missing_out and len(out_ids) >= 6:
            rel = {pid: [] for pid in out_ids}
            for k, v in by_f.items():
                if len(v) < len(out_ids):
                    continue
                sgn = 1.0 if match.attacks_right(team, int(v["period"].iloc[0])) else -1.0
                P = v[["x", "y"]].values * sgn
                c, s = shape(P)
                for pid, p in zip(v["player_id"].values, (P - c) / s):
                    rel[pid].append(p)
            obs = np.array([np.mean(r, 0) for r in rel.values() if len(r)])
            if len(obs) >= 6:
                slots = []
                for _ in range(min(n_missing_out, 2)):  # one at a time: the first fills a hole, then the next
                    slots.append(set_predict(obs, mdl) if "set_rho2_w" in mdl else match_template(obs, templates, 1)[0][0])
                    obs = np.vstack([obs, slots[-1]])
                for j, slot in enumerate(slots):
                    pid = -(10 * (team + 1) + j + 1)
                    for k, v in by_f.items():
                        sgn = 1.0 if match.attacks_right(team, int(v["period"].iloc[0])) else -1.0
                        c, s = shape(v[["x", "y"]].values * sgn)
                        x, y = (c + slot * s) * sgn
                        new.append((k, int(v["period"].iloc[0]), pid, team, False, float(np.clip(x, -L / 2, L / 2)),
                                    float(np.clip(y, -W / 2, W / 2))))
        # ------------------------------------------------------ missing goalkeeper
        if n_gk == 0:
            pid = -(10 * (team + 1))
            for k in frames:
                v = by_f.get(k)
                if v is None or len(v) < 5:
                    continue
                per = int(v["period"].iloc[0])
                sgn = 1.0 if match.attacks_right(team, per) else -1.0  # attacking +x = defending the -x goal
                P = v[["x", "y"]].values * sgn
                b = match.ball.loc[k] if k in match.ball.index else None
                ball = np.array([b["x"], b["y"]]) * sgn if b is not None and np.isfinite([b["x"], b["y"]]).all() else P.mean(0)
                pt = int(b["possession_team"]) if b is not None else -1
                own = 0 if pt < 0 else (1 if pt == team else -1)
                F = gk_features(ball, [own], [P[:, 0].mean()], [P[:, 0].min()], (L, W))
                x, y = gk_predict(F, (L, W), mdl)[0] * sgn
                new.append((k, per, pid, team, True, float(x), float(y)))
    if not new:
        return match
    add = pd.DataFrame(new, columns=["frame", "period", "player_id", "team", "gk", "x", "y"])
    add["is_detected"] = False
    add["imputed"] = True
    add["t"] = add["frame"] / match.fps
    from .physical import velocities

    add["vx"], add["vy"] = np.nan, np.nan
    for _, g in add.groupby(["player_id", "period"]):
        g = g.sort_values("frame")
        v = velocities(g["frame"].values / match.fps, g[["x", "y"]].values)
        add.loc[g.index, "vx"], add.loc[g.index, "vy"] = v[:, 0], v[:, 1]
    out = Match(pd.concat([tab, add], ignore_index=True), match.ball, match.views, match.pitch_length, match.pitch_width,
                match.home_side, match.names, match.colors, match.fps, dict(match.numbers))
    return out
