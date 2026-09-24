"""SkillCorner broadcast-tracking format (the open-data "v3" layout).

Writes ``match.json`` + ``tracking_extrapolated.jsonl`` exactly as SkillCorner ships
them, so the files open in SkillCorner's own ``SkillCorner_Tracking_Viewer.html`` and
load with ``kloppy.skillcorner.load(meta_data=..., raw_data=...)``.

Differences you should know about: player identities are anonymous (``player_id`` is
our track identity, no names / jersey numbers), the team called "home" is simply the
first colour cluster, and ball ``z`` is not estimated (null).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BALL_ID = 55
ROLE_GK = {"id": 0, "position_group": "Goalkeeper", "name": "Goalkeeper", "acronym": "GK"}
ROLE_UNKNOWN = {"id": 1, "position_group": "Unknown", "name": "Unknown", "acronym": "UNK"}


def _ts(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - 3600 * h - 60 * m
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def _hex(bgr) -> str:
    b, g, r = [int(v) for v in bgr]
    return f"#{r:02x}{g:02x}{b:02x}"


def team_ids() -> dict:
    return {0: 1, 1: 2}


def match_json(res) -> dict:
    cfg = res.config
    tids = team_ids()
    ids = [i for i in res.tracking.identities if i.team in (0, 1) and i.role in ("player", "goalkeeper")]
    n = len(res.out_t)
    dur = (res.out_t[-1] - res.out_t[0]) if n else 0.0
    players = []
    for ident in ids:
        fr = res.table.loc[res.table["player_id"] == ident.pid, "timestamp"]
        start, end = (float(fr.min()), float(fr.max())) if len(fr) else (0.0, 0.0)
        team_name = cfg.home_name if ident.team == 0 else cfg.away_name
        players.append({
            "id": int(ident.pid),
            "trackable_object": int(ident.pid),
            "team_id": tids[ident.team],
            "team_player_id": int(ident.pid),
            "number": ident.number,  # read from the shirt by OCR, None if never readable
            "first_name": team_name,
            "last_name": f"#{ident.number}" if ident.number is not None else f"id{ident.pid}",
            "short_name": f"#{ident.number}" if ident.number is not None else f"id{ident.pid}",
            "player_role": ROLE_GK if ident.role == "goalkeeper" else ROLE_UNKNOWN,
            "start_time": _ts(start) if start > 0.05 else "00:00:00",
            "end_time": _ts(end).split(".")[0],
            "yellow_card": 0, "red_card": 0, "injured": False, "goal": 0, "own_goal": 0,
            "birthday": None, "gender": "male",
        })
    colors = res.tracking.team_colors
    # attacking direction: the team whose players sit further left defends the left goal
    side = ["left_to_right", "right_to_left"]
    tx = [res.table.loc[res.table["team"] == t, "x"].mean() for t in ("A", "B")]
    if np.isfinite(tx).all() and tx[0] > tx[1]:
        side = side[::-1]
    return {
        "id": int(cfg.extra.get("match_id", 1)),
        "home_team_score": cfg.extra.get("home_score", 0),
        "away_team_score": cfg.extra.get("away_score", 0),
        "date_time": cfg.extra.get("date_time"),
        "stadium": None,
        "home_team": {"id": tids[0], "name": cfg.home_name, "short_name": cfg.home_name, "acronym": cfg.home_name[:3].upper()},
        "home_team_kit": {"id": 1, "team_id": tids[0], "name": "estimated", "jersey_color": _hex(colors[0]), "number_color": "#ffffff"},
        "away_team": {"id": tids[1], "name": cfg.away_name, "short_name": cfg.away_name, "acronym": cfg.away_name[:3].upper()},
        "away_team_kit": {"id": 2, "team_id": tids[1], "name": "estimated", "jersey_color": _hex(colors[1]), "number_color": "#ffffff"},
        "home_team_coach": None,
        "away_team_coach": None,
        "competition_edition": None,
        "match_periods": [{"period": cfg.period, "name": f"period_{cfg.period}", "start_frame": 0, "end_frame": max(0, n - 1),
                           "duration_frames": n, "duration_minutes": round(dur / 60.0, 2)}],
        "competition_round": None,
        "referees": [],
        "players": players,
        "status": "closed",
        "home_team_side": side,
        "ball": {"trackable_object": BALL_ID},
        "pitch_length": cfg.pitch_length,
        "pitch_width": cfg.pitch_width,
        "generated_by": "soccercal (broadcast video -> tracking); anonymous identities",
    }


def tracking_frames(res):
    """Yields one SkillCorner frame dict per 10 fps output frame."""
    cfg = res.config
    an = res.analysis
    frame_t = np.array([fd.t for fd in an.frames])
    t0 = res.out_t[0] if len(res.out_t) else 0.0
    tab = res.table[res.table["role"].isin(["player", "goalkeeper"])] if len(res.table) else res.table
    by_frame = {k: g for k, g in tab.groupby("frame")} if len(tab) else {}
    team_of = {i.pid: i.team for i in res.tracking.identities}
    # NaN (unknown ball) propagates through the interpolation, as it should
    ball_x = np.interp(res.out_t, frame_t, res.ball_pos[:, 0])
    ball_y = np.interp(res.out_t, frame_t, res.ball_pos[:, 1])
    fps_an = an.proc_fps
    poss_team, poss_pid, last_seen = None, None, -1e9
    cand, cand_n = None, 0  # hysteresis: a new possessor must hold the ball 3 frames (0.3 s)
    for k, t in enumerate(res.out_t):
        i = int(np.clip(np.searchsorted(frame_t, t), 0, len(frame_t) - 1))
        if i > 0 and abs(frame_t[i - 1] - t) < abs(frame_t[i] - t):
            i -= 1
        cam = res.cameras[i]
        corners = cam.corners_projection(cfg.pitch_length, cfg.pitch_width) if cam is not None else \
            {f"{c}_{n}": None for n in ("top_left", "bottom_left", "bottom_right", "top_right") for c in ("x", "y")}
        players = []
        g = by_frame.get(k)
        if g is not None:
            for r in g.itertuples(index=False):
                players.append({"x": round(float(r.x), 2), "y": round(float(r.y), 2),
                                "player_id": int(r.player_id), "is_detected": bool(r.is_detected)})
        # ball: detected only if a detection exists within half a frame of the analysed grid
        near_det = bool(res.ball_det[i]) and bool(abs(frame_t[i] - t) <= 0.5 / fps_an + 1e-6)
        bx, by = ball_x[k], ball_y[k]
        if cam is not None and not (np.isfinite(bx) and np.isfinite(by)) and poss_pid is not None and t - last_seen <= 2.0:
            # ball not seen but a player had it a moment ago: it is at his feet (dribbling, shielding)
            p = next((q for q in players if q["player_id"] == poss_pid), None)
            if p is not None:
                bx, by, near_det = p["x"], p["y"], False
        ball_ok = cam is not None and np.isfinite(bx) and np.isfinite(by)
        ball = {"x": round(float(bx), 2) if ball_ok else None, "y": round(float(by), 2) if ball_ok else None,
                "z": None, "is_detected": near_det if ball_ok else None}
        # possession: team of the closest player within 1.5 m of a seen ball (with memory)
        if ball_ok and near_det and players:
            P = np.array([[p["x"], p["y"]] for p in players])
            d = np.hypot(P[:, 0] - bx, P[:, 1] - by)
            j = int(np.argmin(d))
            if d[j] < 1.5:
                pid = players[j]["player_id"]
                cand_n = cand_n + 1 if pid == cand else 1
                cand = pid
                if pid == poss_pid or cand_n >= 3 or poss_pid is None:
                    poss_pid, poss_team = pid, team_of.get(pid)
                if pid == poss_pid:
                    last_seen = t
        if t - last_seen > 3.0:
            poss_team, poss_pid = None, None
        group = {0: "home team", 1: "away team"}.get(poss_team)
        yield {
            "frame": k,
            "timestamp": _ts(t - t0 + cfg.time_offset_s),
            "period": cfg.period,
            "ball_data": ball,
            "possession": {"player_id": poss_pid if group else None, "group": group},
            "image_corners_projection": corners,
            "player_data": players,
        }


def write_skillcorner(res, out_dir: str | Path, match_id: int | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mid = match_id or int(res.config.extra.get("match_id", 1))
    mpath = out / f"{mid}_match.json"
    tpath = out / f"{mid}_tracking_extrapolated.jsonl"
    with open(mpath, "w") as f:
        json.dump(match_json(res), f, indent=1)
    with open(tpath, "w") as f:
        for fr in tracking_frames(res):
            f.write(json.dumps(fr) + "\n")
    return {"match_json": str(mpath), "tracking_jsonl": str(tpath)}


def load_kloppy(out_dir: str | Path, match_id: int = 1, **kw):
    """Open the export with kloppy (pip install kloppy)."""
    from kloppy import skillcorner

    out = Path(out_dir)
    return skillcorner.load(meta_data=str(out / f"{match_id}_match.json"),
                            raw_data=str(out / f"{match_id}_tracking_extrapolated.jsonl"), **kw)
