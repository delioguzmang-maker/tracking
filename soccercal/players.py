"""Pass 2b: detections -> pitch measurements -> tracks -> identities -> trajectories."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import pitch
from .analysis import Analysis
from .camera import Camera
from .config import Config
from .detect import DESC_DIM
from .stitch import StitchConfig, TeamMotion, Tracklet, stitch
from .teams import OTHER, TeamModel, decide, fit_team_model
from .tracker import PitchTracker, rts_smooth

HEAD_Z = 1.75  # metres: height used to place people whose feet are out of the image


@dataclass
class Measurement:
    xy: np.ndarray  # (N,2)
    R: np.ndarray  # (N,2,2)
    score: np.ndarray
    det_idx: np.ndarray  # index into the frame's person detections
    desc: np.ndarray  # (N,D) float, NaN rows when unknown


def measure(fd, cam: Camera, cfg: Config) -> Measurement:
    """Person detections of one frame on the pitch, with calibrated uncertainty."""
    p = fd.det.persons()
    empty = Measurement(np.zeros((0, 2)), np.zeros((0, 2, 2)), np.zeros(0), np.zeros(0, int), np.zeros((0, DESC_DIM)))
    if len(p) == 0:
        return empty
    b = p.boxes.astype(float)
    H = cam.height
    foot = np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]], 1)
    xy, ok = cam.image_to_ground(foot)
    cut = b[:, 3] >= H - 2  # feet below the image border -> place by the head instead
    if cut.any():
        head = np.stack([(b[cut, 0] + b[cut, 2]) / 2, b[cut, 1]], 1)
        xy_h, ok_h = cam.image_to_plane(head, HEAD_Z)
        xy[cut], ok[cut] = xy_h, ok_h
    keep = ok & pitch.inside_pitch(np.nan_to_num(xy, nan=1e6), cfg.pitch_margin, cfg.pitch_length, cfg.pitch_width)
    exp_h = np.full(len(b), np.nan)
    exp_h[keep] = cam.pixel_height(xy[keep], 1.8)
    ratio = (b[:, 3] - b[:, 1]) / exp_h
    lo, hi = cfg.height_ratio
    keep &= ~cut & (ratio > lo) & (ratio < hi) | cut & keep & (ratio < hi)
    idx = np.nonzero(keep)[0]
    if len(idx) == 0:
        return empty
    xy = xy[idx]
    bw, bh = b[idx, 2] - b[idx, 0], b[idx, 3] - b[idx, 1]
    su = np.maximum(1.0, 0.12 * bw)
    sv = np.where(cut[idx], np.maximum(3.0, 0.15 * bh), np.maximum(1.0, 0.05 * bh))
    J = cam.ground_jacobian(xy)
    Ji = np.linalg.inv(J)
    Rimg = np.zeros((len(idx), 2, 2))
    Rimg[:, 0, 0], Rimg[:, 1, 1] = su ** 2, sv ** 2
    R = Ji @ Rimg @ np.transpose(Ji, (0, 2, 1)) + np.eye(2) * 0.15 ** 2
    d = fd.desc[idx].astype(np.float32) / 255.0
    nrm = np.linalg.norm(d, axis=1)
    d[nrm < 0.5] = np.nan
    d[nrm >= 0.5] /= nrm[nrm >= 0.5, None]
    return Measurement(xy, R, p.scores[idx], idx, d)


@dataclass
class Identity:
    pid: int
    team: int  # 0, 1 or OTHER
    role: str  # "player", "goalkeeper", "referee"
    tracklets: list[Tracklet]
    color_bgr: tuple = (128, 128, 128)
    # trajectory on the analysed-frame grid
    frames: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    vel: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    observed: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))


@dataclass
class TrackingOutput:
    identities: list[Identity]
    team_model: TeamModel | None
    team_colors: list  # BGR per team
    # per analysed frame: {person detection index: identity pid}
    assignments: list[dict]
    measurements: list[Measurement | None]
    motion: TeamMotion | None = None  # on-screen mean velocity per team (for extrapolation)


def track_players(an: Analysis, cams: list, cfg: Config | None = None, progress: bool = False) -> TrackingOutput:
    cfg = cfg or Config()
    fps = an.proc_fps
    n = len(an.frames)
    meas: list[Measurement | None] = [None] * n
    for i, (fd, c) in enumerate(zip(an.frames, cams)):
        if c is not None:
            meas[i] = measure(fd, c, cfg)

    # ---- team colours model over all accepted detections
    D = [m.desc[np.isfinite(m.desc[:, 0])] for m in meas if m is not None and len(m.xy)]
    X = np.concatenate(D) if D else np.zeros((0, DESC_DIM))
    if len(X) > 20000:
        X = X[np.random.default_rng(0).choice(len(X), 20000, replace=False)]
    tm = fit_team_model(X) if len(X) >= 10 else None

    # ---- online tracking on the pitch, restarted at every discontinuity
    trackers_out = []
    tracker, cur_seg = None, None
    for i in range(n):
        c, m = cams[i], meas[i]
        seg = None if c is None else c.info.get("segment")
        if tracker is not None and (c is None or seg != cur_seg):
            trackers_out += tracker.close()
            tracker = None
        if c is None or m is None:
            continue
        if tracker is None:
            tracker = PitchTracker(fps, high=cfg.track_high, low=cfg.track_low, max_lost_s=cfg.max_lost_s)
            cur_seg = seg

        def visible(xy, c=c):
            uv, z = c.ground_to_image(xy)
            return (z > 0) & (uv[:, 0] > -5) & (uv[:, 0] < c.width + 5) & (uv[:, 1] > -5) & (uv[:, 1] < c.height + 5)

        desc = m.desc if tm is not None else None
        tracker.step(i, m.xy, m.R, m.score, desc, visible)
    if tracker is not None:
        trackers_out += tracker.close()

    # ---- tracklets with team votes
    tls = []
    for t in trackers_out:
        if len(t.obs) < 3:
            continue
        fr = np.array([o[0] for o in t.obs])
        z = np.array([o[1] for o in t.obs])
        R = np.array([o[2] for o in t.obs])
        di = np.array([o[3] for o in t.obs])
        sc = np.array([o[4] for o in t.obs])
        descs = np.array([meas[f].desc[d] for f, d in zip(fr, di)])  # d = measurement row
        good = np.isfinite(descs[:, 0])
        tl = Tracklet(t.id, fr, z, R, sc, di)
        if good.any():
            md = descs[good].mean(0)
            tl.desc = md / (np.linalg.norm(md) + 1e-9)
            if tm is not None:
                tl.team, tl.team_conf = decide(tm.distances(descs[good]))
        tls.append(tl)

    # ---- stitching into identities
    scfg = StitchConfig(max_gap_s=cfg.stitch_max_gap_s, **cfg.extra.get("stitch", {}))
    motion = TeamMotion(tls, fps, n)
    chains = stitch(tls, fps, scfg, n_frames=n, motion=motion)
    idents = []
    for k, ch in enumerate(chains, start=1):
        team, conf = OTHER, 0.0
        if tm is not None:
            D = [tm.distances(meas[f].desc[d][None]) for t in ch for f, d in zip(t.frames, t.det_idx)
                 if np.isfinite(meas[f].desc[d][0])]
            if D:
                team, conf = decide(np.concatenate(D))
        idents.append(Identity(k, team, "player" if team != OTHER else "referee", ch))
    _assign_goalkeepers(idents, cfg)
    for ident in idents:
        _trajectory(ident, fps)

    # ---- team colours (for drawing / match.json)
    colors = []
    for tm_ in (0, 1):
        cols = [an.frames[f].color[meas[f].det_idx[d]] for ident in idents if ident.team == tm_
                for t in ident.tracklets for f, d in zip(t.frames, t.det_idx)]
        colors.append(tuple(int(v) for v in np.median(np.array(cols), 0)) if cols else ((0, 0, 255) if tm_ == 0 else (255, 0, 0)))

    assignments = [dict() for _ in range(n)]
    for ident in idents:
        for t in ident.tracklets:
            for f, d in zip(t.frames, t.det_idx):
                assignments[f][int(meas[f].det_idx[d])] = ident.pid
    # identities may have changed team (goalkeepers): recompute the team motion on final teams
    for ident in idents:
        for t in ident.tracklets:
            t.team = ident.team
    motion = TeamMotion([t for i in idents for t in i.tracklets], fps, n)
    return TrackingOutput(idents, tm, colors, assignments, meas, motion)


def defending_sides(idents: list[Identity]) -> dict:
    """Which goal each team defends: {team: -1 (left goal) or +1 (right goal)}.

    Per frame, the two deepest outfield players of each team towards a goal; the team
    whose deepest players are nearer to the left goal defends it (defenders sit between
    attackers and their goal much more consistently than team centroids do)."""
    by_frame: dict[int, dict[int, list]] = {}
    for ident in idents:
        if ident.team not in (0, 1) or ident.role != "player":
            continue
        for t in ident.tracklets:
            for f, z in zip(t.frames, t.z):
                by_frame.setdefault(int(f), {0: [], 1: []})[ident.team].append(z[0])
    score = []
    for d in by_frame.values():
        if len(d[0]) >= 3 and len(d[1]) >= 3:
            score.append(np.mean(np.sort(d[0])[:2]) - np.mean(np.sort(d[1])[:2]))
    if not score:
        return {}
    left = 0 if np.mean(score) < 0 else 1  # team whose deepest players are further left
    return {left: -1, 1 - left: 1}


def _assign_goalkeepers(idents: list[Identity], cfg: Config) -> None:
    """At most one goalkeeper per goal: among people whose kit matches neither team, the one
    who lives in a penalty area and stays closest to that goal. Anyone else with an odd kit
    is a referee (or unknown) and is left out of the player data."""
    L = cfg.pitch_length / 2
    sides = cfg.extra.get("defending_sides") or defending_sides(idents)
    for side in (-1, 1):
        best, best_d = None, np.inf
        for ident in idents:
            if ident.team != OTHER:
                continue
            z = np.concatenate([t.z for t in ident.tracklets])
            if len(z) < 5:
                continue
            in_box = (side * z[:, 0] > L - 16.5) & (np.abs(z[:, 1]) < 20.16)
            if in_box.mean() < 0.6:
                continue
            d = float(np.median(np.hypot(z[:, 0] - side * L, z[:, 1])))
            if d < best_d:
                best, best_d = ident, d
        if best is None:
            continue
        team = next((t for t, s in sides.items() if s == side), None)
        if team is None:
            continue
        best.team, best.role = team, "goalkeeper"


def _trajectory(ident: Identity, fps: float) -> None:
    """RTS-smoothed positions on the analysed-frame grid for each tracklet; gaps between
    tracklets are filled later (``trajectories.py``)."""
    F, P, V, O = [], [], [], []
    for t in ident.tracklets:
        fr, pos, vel = rts_smooth(t.frames, t.z, t.R, fps)
        obs = np.isin(fr, t.frames)
        F.append(fr)
        P.append(pos)
        V.append(vel)
        O.append(obs)
    ident.frames = np.concatenate(F)
    ident.pos = np.concatenate(P)
    ident.vel = np.concatenate(V)
    ident.observed = np.concatenate(O)
