"""Space and value on top of the tracking data: who controls each part of the pitch, and what it is worth.

Works on the SkillCorner files (``*_match.json`` + ``*_tracking_extrapolated.jsonl``), so the same code
runs on this program's output and on SkillCorner's own open data::

    from soccercal.tactics import Match, pitch_control, plot_control
    m = Match.load("salida")              # or Match.from_result(res)
    s = m.snapshot(120)                   # frame 120 (10 fps)
    pc = pitch_control(s)                 # Spearman / Shaw
    plot_control(m, 120)

Models
------
* ``pitch_control``: William Spearman's (2018) physics model exactly as Laurie Shaw implemented it in
  LaurieOnTracking (``Metrica_PitchControl.py``, Friends of Tracking): same parameters, the same
  equations, the same early exits and convergence rule, evaluated for the whole grid at once
  (tens of milliseconds per frame instead of seconds). ``tests/test_tactics.py`` checks it against
  a line-by-line transcription of Shaw's loop.
* ``space_control``: Fernández & Bornn (2018, "Wide Open Spaces"): each player's influence is a
  bivariate Gaussian centred 0.5 s ahead of him, stretched along his velocity, wider the further he
  is from the ball (radius 4 m next to it, 10 m from ~18 m away); team control is the logistic of
  the difference of the summed influences.
* ``value_map``: pitch control times Shaw's EPV surface (``data/EPV_grid.csv``, the probability that
  a possession ends in a goal from each location; LaurieOnTracking, MIT licence): where the
  valuable space the attacking team can reach is.

Adapters: ``to_metrica`` (LaurieOnTracking's wide DataFrames, to run Shaw's own code on our data) and
``soccermap_tensor`` (the 13 input channels of SoccerMap, Fernández & Bornn 2020, for training it).

Broadcast caveat: players the camera has not seen for a while are estimated (``is_detected`` False)
and a player never seen in the clip is missing; both bias the surfaces where the camera is not
looking. ``plot_control`` greys out what is outside the camera view.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .physical import velocities

GROUPS = {"home team": 0, "away team": 1}


# ------------------------------------------------------------------------------------------- data
@dataclass
class Snapshot:
    """Everything the models need at one frame."""
    frame: int
    period: int
    pos: np.ndarray  # (n, 2) m, pitch-centred (x along the pitch, y towards the far touchline)
    vel: np.ndarray  # (n, 2) m/s, 0 where unknown
    team: np.ndarray  # (n,) 0 = home, 1 = away
    gk: np.ndarray  # (n,) bool
    detected: np.ndarray  # (n,) bool: seen by the camera now (else estimated)
    pid: np.ndarray  # (n,) player ids
    ball: np.ndarray  # (2,) NaN if unknown
    attacking: int  # team in possession (0/1), -1 if unknown
    carrier: int  # player id with the ball, -1 if unknown
    view: np.ndarray | None  # (k, 2) polygon of the pitch the camera sees
    attacks_right: tuple  # (team 0, team 1): attacks towards +x?
    pitch: tuple = (105.0, 68.0)


@dataclass
class Match:
    """Tracking data at 10 fps in the SkillCorner layout, ready for the models."""
    table: pd.DataFrame  # frame, period, t, player_id, team (0/1), gk, x, y, vx, vy, is_detected
    ball: pd.DataFrame  # index frame: x, y, possession_team (0/1/-1), possession_player_id (-1)
    views: dict  # frame -> (k, 2) polygon on the pitch seen by the camera
    pitch_length: float = 105.0
    pitch_width: float = 68.0
    home_side: list = field(default_factory=lambda: ["left_to_right", "right_to_left"])  # per period
    names: tuple = ("home", "away")
    colors: tuple = ("#d62728", "#1f4fb4")
    fps: float = 10.0
    numbers: dict = field(default_factory=dict)  # player id -> shirt number (when known)

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_skillcorner(cls, meta: dict, frames) -> "Match":
        home = meta["home_team"]["id"]
        team_of = {p["id"]: 0 if p["team_id"] == home else 1 for p in meta.get("players", [])}
        gk = {p["id"] for p in meta.get("players", []) if (p.get("player_role") or {}).get("acronym") == "GK"}
        rows, balls, views, ts = [], [], {}, []
        for fr in frames:
            k = int(fr["frame"])
            per = int(fr.get("period") or 1)
            if fr.get("timestamp"):
                h, m, s = fr["timestamp"].split(":")
                ts.append((k, 3600 * int(h) + 60 * int(m) + float(s)))
            for p in fr.get("player_data") or []:
                if p.get("player_id") in team_of and p.get("x") is not None:
                    rows.append((k, per, p["player_id"], team_of[p["player_id"]], p["player_id"] in gk,
                                 p["x"], p["y"], bool(p.get("is_detected"))))
            b = fr.get("ball_data") or {}
            poss = fr.get("possession") or {}
            balls.append((k, np.nan if b.get("x") is None else b["x"], np.nan if b.get("y") is None else b["y"],
                          GROUPS.get(poss.get("group"), -1), poss.get("player_id") or -1))
            c = fr.get("image_corners_projection") or {}
            pts = [(c.get(f"x_{n}"), c.get(f"y_{n}")) for n in ("top_left", "bottom_left", "bottom_right", "top_right")]
            if all(x is not None and y is not None for x, y in pts):
                views[k] = np.array(pts, float)
        # frame rate from the timestamps (SkillCorner: 10 fps)
        fps = 10.0
        if len(ts) > 2:
            d = np.diff(np.array(ts, float), axis=0)
            d = d[(d[:, 0] > 0) & (d[:, 1] > 0)]
            if len(d):
                fps = float(np.round(np.median(d[:, 0] / d[:, 1]), 2))
        tab = pd.DataFrame(rows, columns=["frame", "period", "player_id", "team", "gk", "x", "y", "is_detected"])
        tab["t"] = tab["frame"] / fps
        tab["vx"], tab["vy"] = np.nan, np.nan
        for _, g in tab.groupby(["player_id", "period"]):
            g = g.sort_values("frame")
            k = g["frame"].values
            full = np.arange(k[0], k[-1] + 1)
            xy = np.full((len(full), 2), np.nan)
            xy[k - k[0]] = g[["x", "y"]].values
            v = velocities(full / fps, xy)[k - k[0]]
            tab.loc[g.index, "vx"], tab.loc[g.index, "vy"] = v[:, 0], v[:, 1]
        ball = pd.DataFrame(balls, columns=["frame", "x", "y", "possession_team", "possession_player_id"]).set_index("frame")
        hex_ = lambda kit, d: (kit or {}).get("jersey_color") or d  # noqa: E731
        return cls(tab, ball, views, float(meta.get("pitch_length") or 105.0), float(meta.get("pitch_width") or 68.0),
                   list(meta.get("home_team_side") or ["left_to_right", "right_to_left"]),
                   (meta["home_team"].get("short_name") or "home", meta["away_team"].get("short_name") or "away"),
                   (hex_(meta.get("home_team_kit"), "#d62728"), hex_(meta.get("away_team_kit"), "#1f4fb4")), fps,
                   {p["id"]: p["number"] for p in meta.get("players", []) if p.get("number") is not None})

    @classmethod
    def load(cls, folder: str | Path) -> "Match":
        """From a folder with ``*_match.json`` and ``*_tracking_extrapolated.jsonl`` (ours or SkillCorner's)."""
        folder = Path(folder)
        mj = sorted(folder.glob("*_match.json")) or sorted(folder.glob("*match*.json"))
        tj = sorted(folder.glob("*_tracking_extrapolated.jsonl")) or sorted(folder.glob("*tracking*.jsonl"))
        if not mj or not tj:
            raise FileNotFoundError(f"No encuentro *_match.json y *_tracking_extrapolated.jsonl en {folder}")
        meta = json.loads(mj[0].read_text())
        with open(tj[0]) as f:
            return cls.from_skillcorner(meta, (json.loads(line) for line in f if line.strip()))

    @classmethod
    def from_result(cls, res) -> "Match":
        """From the ``Result`` returned by ``soccercal.run`` / ``build``."""
        from .skillcorner import match_json, tracking_frames

        return cls.from_skillcorner(match_json(res), tracking_frames(res))

    # ---------------------------------------------------------------- access
    @property
    def frames(self) -> np.ndarray:
        return np.array(sorted(self.table["frame"].unique()))

    def attacks_right(self, team: int, period: int = 1) -> bool:
        side = self.home_side[min(max(period, 1), len(self.home_side)) - 1] if self.home_side else "left_to_right"
        home_right = side != "right_to_left"
        return home_right if team == 0 else not home_right

    def snapshot(self, frame: int, max_possession_age_s: float = 5.0) -> Snapshot:
        g = self.table[self.table["frame"] == frame]
        period = int(g["period"].iloc[0]) if len(g) else 1
        b = self.ball.loc[frame] if frame in self.ball.index else None
        ball = np.array([b["x"], b["y"]], float) if b is not None else np.full(2, np.nan)
        # the team in possession: now, or the last one known in the previous few seconds
        poss = self.ball.loc[(self.ball.index <= frame) & (self.ball.index > frame - max_possession_age_s * self.fps)]
        poss = poss[poss["possession_team"] >= 0]
        att = int(poss["possession_team"].iloc[-1]) if len(poss) else -1
        carrier = int(poss["possession_player_id"].iloc[-1]) if len(poss) else -1
        vel = np.nan_to_num(g[["vx", "vy"]].values.astype(float))
        return Snapshot(int(frame), period, g[["x", "y"]].values.astype(float), vel, g["team"].values.astype(int),
                        g["gk"].values.astype(bool), g["is_detected"].values.astype(bool), g["player_id"].values,
                        ball, att, carrier, self.views.get(int(frame)),
                        (self.attacks_right(0, period), self.attacks_right(1, period)),
                        (self.pitch_length, self.pitch_width))


@dataclass
class Surface:
    values: np.ndarray  # (ny, nx): probability that ``team`` controls the ball there (or value)
    xgrid: np.ndarray  # (nx,) cell centres (m)
    ygrid: np.ndarray  # (ny,)
    team: int
    model: str
    notes: list = field(default_factory=list)

    def at(self, xy) -> float:
        ix = int(np.argmin(np.abs(self.xgrid - xy[0])))
        iy = int(np.argmin(np.abs(self.ygrid - xy[1])))
        return float(self.values[iy, ix])


def grid(n_grid_cells_x: int = 50, pitch: tuple = (105.0, 68.0)) -> tuple[np.ndarray, np.ndarray]:
    """Cell centres, as in Shaw's ``generate_pitch_control_for_event``."""
    L, W = pitch
    ny = int(n_grid_cells_x * W / L)
    dx, dy = L / n_grid_cells_x, W / ny
    return np.arange(n_grid_cells_x) * dx - L / 2 + dx / 2, np.arange(ny) * dy - W / 2 + dy / 2


def _team_or_attacking(s: Snapshot, team: int | None, notes: list) -> int:
    if team is not None:
        return team
    if s.attacking >= 0:
        return s.attacking
    notes.append("posesión desconocida: se calcula para el equipo local (0)")
    return 0


# ------------------------------------------------------------------- Spearman / Shaw pitch control
def default_params(time_to_control_veto: float = 3) -> dict:
    """Shaw's ``default_model_params`` (LaurieOnTracking), unchanged."""
    p = {"max_player_accel": 7.0, "max_player_speed": 5.0, "reaction_time": 0.7, "tti_sigma": 0.45, "kappa_def": 1.0,
         "lambda_att": 4.3, "average_ball_speed": 15.0, "int_dt": 0.04, "max_int_time": 10, "model_converge_tol": 0.01}
    p["lambda_def"] = 4.3 * p["kappa_def"]
    p["lambda_gk"] = p["lambda_def"] * 3.0
    k = time_to_control_veto * np.log(10)
    p["time_to_control_att"] = k * (np.sqrt(3) * p["tti_sigma"] / np.pi + 1 / p["lambda_att"])
    p["time_to_control_def"] = k * (np.sqrt(3) * p["tti_sigma"] / np.pi + 1 / p["lambda_def"])
    return p


def offside_players(s: Snapshot, attacking: int, tol: float = 0.2) -> np.ndarray:
    """Shaw's ``check_offsides``: attackers beyond the second-last defender (and the ball, and the
    halfway line) by more than ``tol``. Needs the defending goalkeeper; returns all False without him."""
    att, dfd = s.team == attacking, s.team != attacking
    gk = dfd & s.gk
    if not gk.any() or dfd.sum() < 2:
        return np.zeros(len(s.team), bool)
    half = np.sign(s.pos[gk][0, 0])
    second = np.sort(half * s.pos[dfd, 0])[::-1][1]
    line = max(second, half * s.ball[0] if np.isfinite(s.ball[0]) else -np.inf, 0.0) + tol
    return att & (s.pos[:, 0] * half > line)


def pitch_control(s: Snapshot, attacking: int | None = None, params: dict | None = None, n_grid_cells_x: int = 50,
                  offsides: bool = True, targets: np.ndarray | None = None) -> Surface:
    """Probability that the attacking team controls the ball if it is played to each cell.

    Shaw's ``calculate_pitch_control_at_target`` for all cells at once. ``targets`` (m, 2) evaluates
    arbitrary points instead of the grid (``values`` is then (m,)). The ball is assumed to be at the
    target when its position is unknown (as in Shaw's code)."""
    p = params or default_params()
    notes: list = []
    attacking = _team_or_attacking(s, attacking, notes)
    keep = np.isfinite(s.pos).all(1)
    if offsides:
        off = offside_players(s, attacking)
        if off.any():
            notes.append(f"{int(off.sum())} atacante(s) en fuera de juego no cuentan")
        keep &= ~off
    att = keep & (s.team == attacking)
    dfd = keep & (s.team != attacking)
    if targets is None:
        xg, yg = grid(n_grid_cells_x, s.pitch)
        P = np.stack(np.meshgrid(xg, yg), -1).reshape(-1, 2)
    else:
        xg = yg = None
        P = np.asarray(targets, float).reshape(-1, 2)
    if not att.any() or not dfd.any():
        notes.append("falta un equipo entero en este fotograma")
        v = np.full(len(P), 1.0 if att.any() else 0.0)
        return Surface(v.reshape(len(yg), len(xg)) if targets is None else v, xg, yg, attacking, "spearman", notes)
    pos, vel = s.pos[att | dfd], s.vel[att | dfd]
    is_att = att[att | dfd]
    lam = np.where(is_att, p["lambda_att"], np.where(s.gk[att | dfd], p["lambda_gk"], p["lambda_def"]))
    dt = p["int_dt"]
    tau_ball = np.zeros(len(P)) if not np.isfinite(s.ball).all() else np.linalg.norm(P - s.ball, axis=1) / p["average_ball_speed"]
    r_react = pos + vel * p["reaction_time"]
    tti = p["reaction_time"] + np.linalg.norm(P[:, None] - r_react[None], axis=2) / p["max_player_speed"]  # (m, n)
    big = np.inf
    tmin_att = np.where(is_att[None], tti, big).min(1)
    tmin_def = np.where(~is_att[None], tti, big).min(1)
    PPCFa, PPCFd = np.zeros(len(P)), np.zeros(len(P))
    short_def = tmin_att - np.maximum(tau_ball, tmin_def) >= p["time_to_control_def"]
    short_att = ~short_def & (tmin_def - np.maximum(tau_ball, tmin_att) >= p["time_to_control_att"])
    PPCFd[short_def] = 1.0
    PPCFa[short_att] = 1.0
    rest = np.nonzero(~short_def & ~short_att)[0]
    if len(rest):
        tt = tti[rest]
        use = np.where(is_att[None], tt - tmin_att[rest, None] < p["time_to_control_att"],
                       tt - tmin_def[rest, None] < p["time_to_control_def"])
        rate = use * lam[None]  # 0 for players vetoed at that target
        k = np.pi / np.sqrt(3.0) / p["tti_sigma"]
        start = tau_ball[rest] - dt
        n_steps = np.ceil(((tau_ball[rest] + p["max_int_time"]) - start) / dt).astype(int)  # len(np.arange(...))
        ppcf = np.zeros_like(tt)  # each player's cumulative probability
        a = np.zeros(len(rest))
        d = np.zeros(len(rest))
        active = np.ones(len(rest), bool)
        i = 1
        while active.any():
            ix = np.nonzero(active)[0]
            T = start[ix] + i * dt
            f = 1.0 / (1.0 + np.exp(-k * (T[:, None] - tt[ix])))
            ppcf[ix] += ((1 - a[ix] - d[ix])[:, None] * f * rate[ix]) * dt
            a[ix] = (ppcf[ix] * is_att[None]).sum(1)
            d[ix] = (ppcf[ix] * ~is_att[None]).sum(1)
            i += 1
            active[ix] = (1 - (a[ix] + d[ix]) > p["model_converge_tol"]) & (i < n_steps[ix])
        PPCFa[rest], PPCFd[rest] = a, d
    if targets is not None:
        return Surface(PPCFa, xg, yg, attacking, "spearman", notes)
    return Surface(PPCFa.reshape(len(yg), len(xg)), xg, yg, attacking, "spearman", notes)


# ------------------------------------------------------------------ Fernández & Bornn space control
def influence(pos: np.ndarray, vel: np.ndarray, ball: np.ndarray, P: np.ndarray, max_speed: float = 13.0) -> np.ndarray:
    """(n, m) influence of each player at points ``P`` (Fernández & Bornn 2018, appendix): a Gaussian
    with mean ``pos + 0.5 vel``, axes ``R (1 ± Srat) / 2`` along / across the velocity, where
    ``Srat = (speed / 13)^2`` and the radius ``R = min(4 + d_ball^3 / 972, 10)`` m; normalised by its
    value at the player's own position."""
    d_ball = np.linalg.norm(pos - ball, axis=1)
    R = np.minimum(4.0 + d_ball ** 3 / 972.0, 10.0)
    speed = np.linalg.norm(vel, axis=1)
    srat = np.minimum(speed, max_speed) ** 2 / max_speed ** 2
    s1 = R * (1 + srat) / 2
    s2 = np.maximum(R * (1 - srat) / 2, 1e-3)
    th = np.arctan2(vel[:, 1], vel[:, 0])
    c, s = np.cos(th), np.sin(th)
    mu = pos + 0.5 * vel

    def maha(q):  # squared Mahalanobis distance of points q (n, m, 2) to each player's Gaussian
        dq = q - mu[:, None]
        u = (dq[..., 0] * c[:, None] + dq[..., 1] * s[:, None]) / s1[:, None]  # along the velocity
        w = (-dq[..., 0] * s[:, None] + dq[..., 1] * c[:, None]) / s2[:, None]  # across
        return u * u + w * w
    return np.exp(-0.5 * (maha(np.broadcast_to(P, (len(pos),) + P.shape)) - maha(pos[:, None])))


def space_control(s: Snapshot, team: int | None = None, n_grid_cells_x: int = 105) -> Surface:
    """Fernández & Bornn (2018) control of ``team`` (default: the team in possession) at each cell:
    logistic(sum of its players' influence - sum of the opponents')."""
    notes: list = []
    team = _team_or_attacking(s, team, notes)
    xg, yg = grid(n_grid_cells_x, s.pitch)
    P = np.stack(np.meshgrid(xg, yg), -1).reshape(-1, 2)
    ok = np.isfinite(s.pos).all(1)
    ball = s.ball
    if not np.isfinite(ball).all():
        ball = s.pos[ok & s.detected].mean(0) if (ok & s.detected).any() else s.pos[ok].mean(0)
        notes.append("balón desconocido: se usa el centro de los jugadores vistos")
    I = influence(s.pos[ok], s.vel[ok], ball, P)
    mine = s.team[ok] == team
    z = I[mine].sum(0) - I[~mine].sum(0)
    return Surface((1.0 / (1.0 + np.exp(-z))).reshape(len(yg), len(xg)), xg, yg, team, "fernandez_bornn", notes)


# ------------------------------------------------------------------------------------------- value
def epv_grid() -> np.ndarray:
    """Shaw's EPV surface (32 x 50, attacking left to right; LaurieOnTracking, MIT licence)."""
    return np.loadtxt(Path(__file__).with_name("data") / "EPV_grid.csv", delimiter=",")


def epv_at(xy: np.ndarray, attacks_right: bool, pitch: tuple = (105.0, 68.0), epv: np.ndarray | None = None) -> np.ndarray:
    """EPV at points ``xy`` (m, 2) for a team attacking towards +x (``attacks_right``) or -x; 0 off the pitch."""
    E = epv_grid() if epv is None else epv
    if not attacks_right:
        E = np.fliplr(E)
    ny, nx = E.shape
    L, W = pitch
    xy = np.atleast_2d(xy)
    ix = np.clip(((xy[:, 0] + L / 2) / L * nx).astype(int), 0, nx - 1)
    iy = np.clip(((xy[:, 1] + W / 2) / W * ny).astype(int), 0, ny - 1)
    out = E[iy, ix]
    out[(np.abs(xy[:, 0]) > L / 2) | (np.abs(xy[:, 1]) > W / 2)] = 0.0
    return out


def value_map(s: Snapshot, surface: Surface | None = None) -> Surface:
    """Pitch control x EPV for the team of ``surface`` (default: Spearman for the team in possession):
    how likely a ball played there is kept, times how likely a goal is from there (Shaw, Tutorial 4)."""
    surface = surface or pitch_control(s)
    P = np.stack(np.meshgrid(surface.xgrid, surface.ygrid), -1).reshape(-1, 2)
    e = epv_at(P, s.attacks_right[surface.team], s.pitch).reshape(surface.values.shape)
    return Surface(surface.values * e, surface.xgrid, surface.ygrid, surface.team, surface.model + "_x_epv",
                   list(surface.notes))


# ---------------------------------------------------------------------------------------- adapters
def to_metrica(match: Match) -> tuple[pd.DataFrame, pd.DataFrame, tuple]:
    """LaurieOnTracking's ``tracking_home``, ``tracking_away`` (index = frame; ``Period``, ``Time [s]``,
    ``Home_<id>_x/_y/_vx/_vy/_speed``, ``ball_x``, ``ball_y``; metres, pitch-centred, the same axes as
    Shaw's ``to_metric_coordinates``) and ``GK_numbers`` (home, away), ready for ``Metrica_PitchControl``
    and ``Metrica_Viz``. Unknown positions are NaN, which Shaw's code skips."""
    tab = match.table
    frames = match.frames
    out = []
    for team, name in ((0, "Home"), (1, "Away")):
        g = tab[tab["team"] == team]
        cols = {}
        for c in ("x", "y", "vx", "vy"):
            w = g.pivot_table(index="frame", columns="player_id", values=c, aggfunc="first").reindex(frames)
            for pid in w.columns:
                cols[f"{name}_{pid}_{c}"] = w[pid].values
        for pid in sorted({k.split("_")[1] for k in cols}, key=int):
            cols[f"{name}_{pid}_speed"] = np.hypot(cols[f"{name}_{pid}_vx"], cols[f"{name}_{pid}_vy"])
        df = pd.DataFrame(cols, index=pd.Index(frames, name="Frame"))
        df = df[sorted(df.columns, key=lambda k: (int(k.split("_")[1]), ["x", "y", "vx", "vy", "speed"].index(k.split("_")[2])))]
        per = tab.groupby("frame")["period"].first().reindex(frames).ffill().bfill()
        df.insert(0, "Time [s]", frames / match.fps)
        df.insert(0, "Period", per.values.astype(int))
        df["ball_x"] = match.ball["x"].reindex(frames).values
        df["ball_y"] = match.ball["y"].reindex(frames).values
        out.append(df)
    gks = []
    for team in (0, 1):
        g = tab[(tab["team"] == team) & tab["gk"]]
        gks.append(str(int(g["player_id"].mode().iloc[0])) if len(g) else None)
    return out[0], out[1], tuple(gks)


SOCCERMAP_CHANNELS = ("attack_location", "attack_vx", "attack_vy", "defence_location", "defence_vx", "defence_vy",
                      "dist_ball", "dist_goal", "sin_goal_ball", "cos_goal_ball", "angle_to_goal",
                      "sin_vel_carrier", "cos_vel_carrier")


def soccermap_tensor(s: Snapshot, team: int | None = None, dims: tuple = (68, 104)) -> np.ndarray:
    """The 13 input channels of SoccerMap (Fernández & Bornn 2020) at this frame, (13, 68, 104),
    oriented so that ``team`` (default: in possession) attacks towards +x:

    0-5  sparse: location (1) and vx, vy of the attacking, then the defending players;
    6-7  distance (m) from every cell to the ball and to the goal centre;
    8-9  sine and cosine of the angle goal-cell-ball; 10 angle (rad) from the cell to the goal;
    11-12 sparse, attacking players: sine and cosine of the angle between their velocity and the
         ball carrier's.

    Dense channels follow the un-xPass implementation (KU Leuven) of the same model."""
    notes: list = []
    team = _team_or_attacking(s, team, notes)
    L, W = s.pitch
    ny, nx = dims
    flip = -1.0 if not s.attacks_right[team] else 1.0  # rotate the pitch 180 degrees if needed
    pos, vel, ball = s.pos * flip, s.vel * flip, s.ball * flip
    X = np.zeros((13, ny, nx), np.float32)
    ok = np.isfinite(pos).all(1)
    ix = np.clip(((pos[:, 0] + L / 2) / L * nx).astype(int), 0, nx - 1)
    iy = np.clip(((pos[:, 1] + W / 2) / W * ny).astype(int), 0, ny - 1)
    for base, m in ((0, ok & (s.team == team)), (3, ok & (s.team != team))):
        X[base, iy[m], ix[m]] = 1.0
        X[base + 1, iy[m], ix[m]] = vel[m, 0]
        X[base + 2, iy[m], ix[m]] = vel[m, 1]
    cx = (np.arange(nx) + 0.5) * L / nx - L / 2
    cy = (np.arange(ny) + 0.5) * W / ny - W / 2
    CX, CY = np.meshgrid(cx, cy)
    goal = np.array([L / 2, 0.0])
    if not np.isfinite(ball).all():
        c = ok & (s.pid == s.carrier)
        ball = pos[c][0] if c.any() else np.array([0.0, 0.0])
    X[6] = np.hypot(CX - ball[0], CY - ball[1])
    X[7] = np.hypot(CX - goal[0], CY - goal[1])
    ax, ay, bx, by = goal[0] - CX, goal[1] - CY, ball[0] - CX, ball[1] - CY
    cos = np.clip((ax * bx + ay * by) / np.maximum(np.hypot(ax, ay) * np.hypot(bx, by), 1e-9), -1, 1)
    X[8], X[9] = np.sqrt(1 - cos ** 2), cos
    X[10] = np.abs(np.arctan2(ay, ax))
    c = ok & (s.pid == s.carrier)
    if c.any():
        vc = vel[c][0]
        m = ok & (s.team == team)
        n1, n2 = np.hypot(vel[m, 0], vel[m, 1]), np.hypot(*vc)
        if n2 > 0.1:
            good = n1 > 0.1
            mm = np.nonzero(m)[0][good]
            v = vel[mm]
            X[11, iy[mm], ix[mm]] = (v[:, 0] * vc[1] - v[:, 1] * vc[0]) / (n1[good] * n2)
            X[12, iy[mm], ix[mm]] = (v[:, 0] * vc[0] + v[:, 1] * vc[1]) / (n1[good] * n2)
    return X


# ----------------------------------------------------------------------------------------- series
def control_series(match: Match, model: str = "spearman", every: int = 5, n_grid_cells_x: int = 50) -> pd.DataFrame:
    """Share of the pitch (and of the part the camera sees) controlled by each team, every ``every`` frames."""
    rows = []
    for k in match.frames[::every]:
        s = match.snapshot(int(k))
        if s.team.size == 0:
            continue
        sf = space_control(s, team=0, n_grid_cells_x=n_grid_cells_x) if model != "spearman" else None
        if sf is None:
            pc = pitch_control(s, n_grid_cells_x=n_grid_cells_x)
            v0 = pc.values if pc.team == 0 else 1 - pc.values
            xg, yg = pc.xgrid, pc.ygrid
        else:
            v0, xg, yg = sf.values, sf.xgrid, sf.ygrid
        r = {"frame": int(k), "t": k / match.fps, "possession": s.attacking, "home_share": float((v0 > 0.5).mean()),
             "n_home": int((s.team == 0).sum()), "n_away": int((s.team == 1).sum())}
        if s.view is not None:
            from matplotlib.path import Path as MPath

            P = np.stack(np.meshgrid(xg, yg), -1).reshape(-1, 2)
            inside = MPath(s.view).contains_points(P).reshape(v0.shape)
            r["home_share_in_view"] = float((v0[inside] > 0.5).mean()) if inside.any() else np.nan
        rows.append(r)
    out = pd.DataFrame(rows)
    if len(out):
        out["away_share"] = 1 - out["home_share"]
    return out


# ------------------------------------------------------------------------------------------- plots
def _cmap(c_neg: str, c_pos: str):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("ctl", [c_neg, "#ffffff", c_pos])


def _distinct(c0: str, c1: str) -> tuple[str, str]:
    """The kit colours if they make a readable map; a dark, grey or white kit (or two kits of similar
    hue) is replaced by red or blue, whichever differs more from the other team."""
    from matplotlib.colors import rgb_to_hsv, to_rgb

    red, blue = "#d62728", "#1f4fb4"
    hsv = [rgb_to_hsv(to_rgb(c)) for c in (c0, c1)]
    ok = [h[1] > 0.35 and h[2] > 0.25 for h in hsv]
    dh = abs(hsv[0][0] - hsv[1][0])
    if all(ok) and min(dh, 1 - dh) > 0.12:
        return c0, c1
    if ok[0] != ok[1]:
        good = 0 if ok[0] else 1
        bluish = 0.45 < hsv[good][0] < 0.8
        out = [c0, c1]
        out[1 - good] = red if bluish else blue
        return out[0], out[1]
    return red, blue


def draw_pitch(ax, length: float = 105.0, width: float = 68.0, color: str = "#888888"):
    from . import pitch

    for pl in pitch.polylines(length, width):
        if np.abs(pl[:, 2]).max() == 0:
            ax.plot(pl[:, 0], pl[:, 1], color=color, lw=0.8, zorder=2)
    ax.set_xlim(-length / 2 - 3, length / 2 + 3)
    ax.set_ylim(-width / 2 - 3, width / 2 + 3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


BALL_COLOR = "#ffd400"  # yellow, as on the verification video's minimap


def plot_control(match: Match, frame: int, model: str = "spearman", ax=None, n_grid_cells_x: int | None = None,
                 show_view: bool = True, arrows: bool = True, labels: bool = True):
    """Map of who controls the space at ``frame``. ``model``: ``"spearman"`` (Spearman / Shaw),
    ``"fernandez"`` (Fernández & Bornn) or ``"value"`` (Spearman x EPV of the team in possession).

    Colour = team that would win a ball played there (white = 50/50). Solid dots = seen by the
    camera, see-through with a dashed outline = estimated; yellow = the ball; arrows = where each
    player will be in 1 s; grey = outside the camera view (positions there are estimated). Labels:
    shirt number when read, else the track id."""
    import matplotlib.pyplot as plt

    s = match.snapshot(int(frame))
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 7.4))
    L, W = s.pitch
    c0, c1 = _distinct(*match.colors)
    cols = (c0, c1)
    if model == "fernandez":
        sf = space_control(s, n_grid_cells_x=n_grid_cells_x or 105)
    else:
        sf = pitch_control(s, n_grid_cells_x=n_grid_cells_x or 50)
    ext = (-L / 2, L / 2, -W / 2, W / 2)
    if model == "value":
        sf = value_map(s, sf)
        from matplotlib.colors import LinearSegmentedColormap

        cm = LinearSegmentedColormap.from_list("v", ["#ffffff", cols[sf.team]])
        im = ax.imshow(sf.values, extent=ext, origin="lower", cmap=cm, vmin=0, vmax=max(sf.values.max(), 1e-3),
                       interpolation="bilinear", zorder=1)
        label = f"valor (control x EPV) de {match.names[sf.team]}"
    else:
        other = 1 - sf.team
        im = ax.imshow(sf.values, extent=ext, origin="lower", cmap=_cmap(cols[other], cols[sf.team]), vmin=0, vmax=1,
                       interpolation="bilinear", zorder=1)
        label = (f"probabilidad de que {match.names[sf.team]} gane un balón jugado ahí" if model == "spearman"
                 else f"control de {match.names[sf.team]} (0-1)")
    draw_pitch(ax, L, W)
    if show_view and s.view is not None:
        from matplotlib.path import Path as MPath

        gx, gy = np.linspace(-L / 2, L / 2, 211), np.linspace(-W / 2, W / 2, 137)
        Q = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)
        out = ~MPath(s.view).contains_points(Q).reshape(len(gy), len(gx))
        rgba = np.zeros(out.shape + (4,))
        rgba[out] = (0.55, 0.55, 0.55, 0.55)
        ax.imshow(rgba, extent=ext, origin="lower", zorder=3, interpolation="nearest")
        ax.plot(*np.vstack([s.view, s.view[:1]]).T, ls="--", c="#333333", lw=0.8, zorder=4)
    import matplotlib.patheffects as pe
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D

    for t in (0, 1):
        m = s.team == t
        for det in (True, False):  # seen: solid; estimated (off camera): see-through, dashed outline
            mm = m & (s.detected == det)
            ax.scatter(s.pos[mm, 0], s.pos[mm, 1], s=80, zorder=6, linewidths=1.2,
                       facecolors=cols[t] if det else to_rgba(cols[t], 0.4), edgecolors="k",
                       linestyles="-" if det else (0, (2, 1.5)))
        if arrows and m.any():
            ax.quiver(s.pos[m, 0], s.pos[m, 1], s.vel[m, 0], s.vel[m, 1], angles="xy", scale_units="xy", scale=1,
                      color=cols[t], width=0.003, zorder=5)
    if labels:
        for pid, xy in zip(s.pid, s.pos):
            num = match.numbers.get(int(pid))
            ax.text(xy[0] + 1.2, xy[1] + 1.2, f"{num}" if num is not None else f"id{pid}", fontsize=7, color="white",
                    zorder=7, path_effects=[pe.withStroke(linewidth=2, foreground="k")])
    car = s.pid == s.carrier
    if car.any():
        ax.scatter(*s.pos[car][0], s=280, facecolors="none", edgecolors="k", linewidths=1.2, zorder=6)
    if np.isfinite(s.ball).all():
        ax.scatter(*s.ball, s=32, c=BALL_COLOR, edgecolors="k", linewidths=1.0, zorder=8)
    handles = [Line2D([], [], ls="", marker="o", ms=8, mfc=cols[0], mec="k", label="visto por la cámara"),
               Line2D([], [], ls="", marker="o", ms=8, mfc=to_rgba(cols[0], 0.4), mec="k", label="estimado (fuera de cámara)"),
               Line2D([], [], ls="", marker="o", ms=6, mfc=BALL_COLOR, mec="k", label="balón"),
               Line2D([], [], ls="", marker="o", ms=13, mfc="none", mec="k", label="tiene el balón")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=8, frameon=False)
    n0, n1 = int((s.team == 0).sum()), int((s.team == 1).sum())
    name = {"spearman": "Spearman / Shaw", "fernandez": "Fernández & Bornn", "value": "Spearman x EPV"}.get(model, model)
    ttl = (f"{name}  ·  t = {frame / match.fps:.1f} s  ·  posesión: "
           f"{match.names[s.attacking] if s.attacking >= 0 else 'posesión desconocida'}  ·  "
           f"jugadores {match.names[0]} {n0}, {match.names[1]} {n1}")
    ax.set_title(ttl, fontsize=10)
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label(label, fontsize=9)
    if sf.notes:
        ax.text(-L / 2, W / 2 + 1.0, "; ".join(sf.notes), fontsize=8, va="bottom")
    return ax


def render_control_video(match: Match, path: str | Path, model: str = "spearman", frames=None, dpi: int = 80,
                         progress: bool = True) -> str:
    """The ``plot_control`` map for every frame (10 fps) as an .mp4 (H.264 if ffmpeg is installed)."""
    import shutil
    import subprocess

    import cv2
    import matplotlib.pyplot as plt

    frames = match.frames if frames is None else frames
    path = str(path)
    raw = path if not shutil.which("ffmpeg") else path + ".mp4v.mp4"
    vw = None
    it = frames
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(frames, unit="frame", desc="Control del espacio")
        except ImportError:
            pass
    fig = plt.figure(figsize=(11, 7.4), dpi=dpi)
    for k in it:
        fig.clf()
        plot_control(match, int(k), model=model, ax=fig.add_subplot(111))
        fig.tight_layout()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        if vw is None:
            vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), match.fps, (w - w % 2, h - h % 2))
        vw.write(img[: h - h % 2, : w - w % 2])
    plt.close(fig)
    if vw is not None:
        vw.release()
    if raw != path:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-vcodec", "libx264", "-pix_fmt", "yuv420p", path],
                       check=True)
        Path(raw).unlink()
    return path
