"""Pass 2b: detections -> pitch measurements -> tracks -> identities -> trajectories."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import pitch
from .analysis import Analysis
from .camera import Camera
from .config import Config
from .detect import DESC_DIM
from .jersey import vote_number
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
    role: str  # "player", "goalkeeper", "referee", "assistant_referee", "staff" (coach, bench, ball boy)
    tracklets: list[Tracklet]
    color_bgr: tuple = (128, 128, 128)
    number: int | None = None  # shirt number read by OCR (None = unknown)
    number_votes: int = 0
    kit_group: int | None = None  # colour group most of the detections are nearest to
    absorbed: list = field(default_factory=list)  # duplicate fragments: drawn with this id, not in the data
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
        tl.outside = outside_share(z, cfg)
        for f, d in zip(fr, di):  # shirt numbers read on this person
            jr = an.frames[f].jersey
            p = int(meas[f].det_idx[d])
            if jr and p in jr:
                tl.jersey_reads.append(jr[p])
        tl.number, _ = vote_number(tl.jersey_reads)
        if good.any() and tm is not None:
            tl.team, tl.team_conf = decide(tm.distances(descs[good]), has_number=tl.number is not None)
        if good.any():
            md = descs[good].mean(0)
            tl.desc = md / (np.linalg.norm(md) + 1e-9)
        tls.append(tl)

    # ---- stitching into identities
    scfg = StitchConfig(max_gap_s=cfg.stitch_max_gap_s, **cfg.extra.get("stitch", {}))
    motion = TeamMotion(tls, fps, n)
    chains = stitch(tls, fps, scfg, n_frames=n, motion=motion)
    idents = []
    for k, ch in enumerate(chains, start=1):
        number, votes = vote_number([r for t in ch for r in t.jersey_reads])
        team, conf, group = OTHER, 0.0, None
        if tm is not None:
            D = [meas[f].desc[d] for t in ch for f, d in zip(t.frames, t.det_idx) if np.isfinite(meas[f].desc[d][0])]
            if D:
                dist = tm.distances(np.array(D))
                team, conf = decide(dist, has_number=number is not None)
                group = int(np.bincount(np.argmin(dist, 1)).argmax())
        ident = Identity(k, team, "player" if team != OTHER else "referee", ch, number=number, number_votes=votes,
                         kit_group=group)
        z = np.concatenate([t.z for t in ch])
        if off_pitch(z, cfg):
            # lives outside the lines: coach, assistant referee, substitute, ball boy -> not a player
            ident.team, ident.role = OTHER, "staff"
        idents.append(ident)
    idents = merge_by_number(idents)
    _assign_goalkeepers(idents, cfg)
    mark_assistant_referees(idents, cfg)
    idents = merge_unique_roles(idents)
    for ident in idents:
        _trajectory(ident, fps)

    # ---- team colours (for drawing / match.json)
    raw = []
    for tm_ in (0, 1):
        cols = [an.frames[f].color[meas[f].det_idx[d]] for ident in idents if ident.team == tm_ and ident.role == "player"
                for t in ident.tracklets for f, d in zip(t.frames, t.det_idx)]
        raw.append(np.median(np.array(cols), 0) if cols else None)
    colors = kit_display_colors(raw)

    assignments = [dict() for _ in range(n)]
    for ident in idents:
        for t in ident.tracklets:
            for f, d in zip(t.frames, t.det_idx):
                assignments[f][int(meas[f].det_idx[d])] = ident.pid
    for ident in idents:
        for t in ident.absorbed:
            for f, d in zip(t.frames, t.det_idx):
                assignments[f].setdefault(int(meas[f].det_idx[d]), ident.pid)
    # identities may have changed team (goalkeepers): recompute the team motion on final teams
    for ident in idents:
        for t in ident.tracklets:
            t.team = ident.team
    motion = TeamMotion([t for i in idents for t in i.tracklets], fps, n)
    return TrackingOutput(idents, tm, colors, assignments, meas, motion)


def outside_share(z: np.ndarray, cfg: Config, tol: float = 0.75) -> float:
    """Fraction of positions clearly outside the touch / goal lines."""
    if len(z) == 0:
        return 0.0
    out = (np.abs(z[:, 0]) > cfg.pitch_length / 2 + tol) | (np.abs(z[:, 1]) > cfg.pitch_width / 2 + tol)
    return float(out.mean())


def off_pitch(z: np.ndarray, cfg: Config) -> bool:
    """Someone who lives outside the lines: often outside, or typically more than 0.5 m out (a
    coach at the edge of the technical area is ~1 m out; calibration noise puts him on the line
    half of the time)."""
    if len(z) == 0:
        return False
    beyond = np.maximum(np.abs(z[:, 0]) - cfg.pitch_length / 2, np.abs(z[:, 1]) - cfg.pitch_width / 2)
    return outside_share(z, cfg) >= cfg.staff_outside_share or float(np.median(beyond)) > 0.5


def mark_assistant_referees(idents: list[Identity], cfg: Config) -> None:
    """Assistant referees live on a touchline and wear the referee's kit. Anyone else on the
    touchline with a kit of neither team (coach in a black jacket, fourth official,
    substitutes) is staff. Without a referee to compare with, an odd kit on the line is an
    assistant referee."""
    W2 = cfg.pitch_width / 2

    def med_y(i):
        return float(np.median(np.abs(np.concatenate([t.z for t in i.tracklets])[:, 1])))

    def size(i):
        return sum(len(t.frames) for t in i.tracklets)

    refs = [i for i in idents if i.role == "referee" and med_y(i) <= W2 - 1.5]
    ref_group = max(refs, key=size).kit_group if refs else None
    for ident in idents:
        if ident.team != OTHER or ident.role not in ("referee", "staff") or not W2 - 1.5 < med_y(ident) < W2 + 3:
            continue
        same_kit = ident.kit_group == ref_group
        if ident.role == "referee":
            ident.role = "assistant_referee" if ref_group is None or same_kit else "staff"
        elif ref_group is not None and same_kit:
            ident.role = "assistant_referee"  # runs just outside the line: staff by position, not by kit


def merge_unique_roles(idents: list[Identity], dup_dist: float = 2.0) -> list[Identity]:
    """There is one referee on the pitch and one goalkeeper per team: their fragments that
    never appear at the same time are the same person. A fragment seen *at the same time*
    as the main one and at the same place is a duplicate (a second, partial box of an
    occluded referee): it is absorbed (drawn with the same label, no data of its own)."""
    drop = set()
    for key in [("referee", OTHER), ("goalkeeper", 0), ("goalkeeper", 1)]:
        grp = sorted([i for i in idents if i.role == key[0] and (key[0] == "referee" or i.team == key[1])],
                     key=lambda i: -sum(len(t.frames) for t in i.tracklets))
        if len(grp) < 2:
            continue
        host = grp[0]
        used = set(int(f) for t in host.tracklets for f in t.frames)
        for other in grp[1:]:
            fr = set(int(f) for t in other.tracklets for f in t.frames)
            if fr & used:
                if _median_gap(host, other) < dup_dist:
                    host.absorbed += other.tracklets + other.absorbed
                    drop.add(other.pid)
                continue
            host.tracklets = sorted(host.tracklets + other.tracklets, key=lambda t: t.start)
            used |= fr
            drop.add(other.pid)
    return [i for i in idents if i.pid not in drop]


def _median_gap(host: Identity, other: Identity) -> float:
    """Median distance between ``other``'s observations and ``host``'s (interpolated) path."""
    hf = np.concatenate([t.frames for t in host.tracklets]).astype(float)
    hz = np.concatenate([t.z for t in host.tracklets])
    o = np.argsort(hf)
    hf, hz = hf[o], hz[o]
    of = np.concatenate([t.frames for t in other.tracklets]).astype(float)
    oz = np.concatenate([t.z for t in other.tracklets])
    inside = (of >= hf[0]) & (of <= hf[-1])
    if not inside.any():
        return np.inf
    p = np.stack([np.interp(of[inside], hf, hz[:, 0]), np.interp(of[inside], hf, hz[:, 1])], 1)
    return float(np.median(np.linalg.norm(oz[inside] - p, axis=1)))


def merge_by_number(idents: list[Identity]) -> list[Identity]:
    """Fragments of one player that carry the same team + shirt number and never appear at
    the same time are the same person: merge them (the kinematic stitcher could not, e.g.
    after a long time off-screen). Two identities with the same number *at the same time*
    are a conflict: the one with fewer votes loses its number."""
    groups: dict[tuple, list[Identity]] = {}
    for ident in idents:
        if ident.number is not None and ident.team in (0, 1) and ident.role != "staff":
            groups.setdefault((ident.team, ident.number), []).append(ident)
    drop = set()
    for (_, _), g in groups.items():
        g.sort(key=lambda i: -i.number_votes)
        keep = []
        for ident in g:
            fr = set(int(f) for t in ident.tracklets for f in t.frames)
            host = next((k for k in keep if not fr & k[1]), None)
            if host is None:
                if keep:  # overlaps every kept identity with that number: its reading is wrong
                    ident.number = None
                else:
                    keep.append((ident, fr))
                continue
            host[0].tracklets = sorted(host[0].tracklets + ident.tracklets, key=lambda t: t.start)
            host[0].number_votes += ident.number_votes
            host[1].update(fr)
            drop.add(ident.pid)
    return [i for i in idents if i.pid not in drop]


def kit_display_colors(raw: list) -> list:
    """Readable team colours from the median shirt colour (BGR): saturated and bright for
    coloured kits, near-black / near-white for dark / white kits; if the two teams would
    look alike, fall back to red vs blue."""
    import cv2

    out = []
    for c in raw:
        if c is None:
            out.append(None)
            continue
        h, s, v = cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_BGR2HSV)[0, 0].astype(int)
        if v < 95:  # dark kit (navy, black): shade makes its hue meaningless
            out.append((45, 45, 45))
        elif s < 70:  # white / grey kit
            v2 = 45 if v < 110 else 235
            out.append((v2, v2, v2))
        else:
            b, g, r = cv2.cvtColor(np.uint8([[[_kit_hue(h), 220, 235]]]), cv2.COLOR_HSV2BGR)[0, 0]
            out.append((int(b), int(g), int(r)))
    fallback = [(40, 40, 230), (230, 120, 20)]
    for k in (0, 1):
        if out[k] is None:
            out[k] = fallback[k]
    a, b = np.array(out[0], float), np.array(out[1], float)
    if np.linalg.norm(a - b) < 90:
        out = fallback
    return [tuple(int(x) for x in c) for c in out]


def _kit_hue(h: int) -> int:
    """Snap an OpenCV hue (0-179) to a canonical kit colour. Skin, stadium light and shade pull
    a red shirt towards orange (measured hue ~14 for Bayern's red at night)."""
    for top, canon in ((17, 0), (25, 15), (40, 28), (85, 60), (105, 100), (135, 118), (165, 150)):
        if h < top:
            return canon
    return 0


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
            if ident.team != OTHER or ident.role == "staff":
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
