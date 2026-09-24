"""End-to-end: broadcast video -> SkillCorner-style tracking data.

    from soccercal import run
    result = run("partido.mp4", "salida/")

``run`` = ``analyze`` (neural networks, slow, cached to ``analysis.pkl.gz``) +
``build`` (calibration over time, tracking, re-identification, 10 fps resampling;
seconds) + ``export`` (+ optional ``render`` of a verification video).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import Analysis, analyze
from .ball import track_ball
from .cameras import Segment, solve_cameras
from .config import Config
from .physical import physical_summary, speeds
from .players import TrackingOutput, track_players
from .teams import OTHER
from .timeline import EDGE, LONG_GAP, sample_identity


@dataclass
class Result:
    analysis: Analysis
    config: Config
    cameras: list
    segments: list[Segment]
    tracking: TrackingOutput
    ball_pos: np.ndarray
    ball_det: np.ndarray
    out_t: np.ndarray  # output timestamps (s, video time)
    table: pd.DataFrame  # long format, one row per player per output frame
    physical: pd.DataFrame
    timings: dict

    # ------------------------------------------------------------ summaries
    def summary(self) -> dict:
        n = len(self.cameras)
        valid = sum(c is not None for c in self.cameras)
        ids = self.tracking.identities
        per_team = {t: sum(1 for i in ids if i.team == t and i.role in ("player", "goalkeeper")) for t in (0, 1)}
        shots = [getattr(fd, "shot", "unknown") for fd in self.analysis.frames]
        aligns = [c.info["align"] for c in self.cameras if c is not None and c.info.get("align") is not None]
        return {
            "frames_analysed": n,
            "frames_with_camera": valid,
            "pct_frames_with_camera": round(100.0 * valid / max(n, 1), 1),
            "camera_shots_used": sum(any(self.cameras[i] is not None for i in range(s.start, s.end)) for s in self.segments),
            "median_line_alignment": round(float(np.median(aligns)), 3) if aligns else None,
            "identities_team_A": per_team[0],
            "identities_team_B": per_team[1],
            "referees": sum(1 for i in ids if i.role == "referee"),
            "assistant_referees": sum(1 for i in ids if i.role == "assistant_referee"),
            "staff_off_pitch": sum(1 for i in ids if i.role == "staff"),
            "goalkeepers": sum(1 for i in ids if i.role == "goalkeeper"),
            "shirt_numbers_read": sum(1 for i in ids if i.number is not None and i.role in ("player", "goalkeeper")),
            "pct_frames_closeup_or_crowd": round(100.0 * sum(s in ("closeup", "no_pitch") for s in shots) / max(n, 1), 1),
            "output_frames": int(len(self.out_t)),
            "pct_player_rows_detected": round(100.0 * float(self.table["is_detected"].mean()), 1) if len(self.table) else 0.0,
            "ball_detected_pct": round(100.0 * float(self.ball_det[[c is not None for c in self.cameras]].mean()), 1) if valid else 0.0,
            **{k: round(v, 1) for k, v in self.timings.items()},
        }

    def save(self, out_dir: str | Path) -> dict:
        from .skillcorner import write_skillcorner

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = write_skillcorner(self, out)
        self.table.to_csv(out / "tracking.csv", index=False)
        self.physical.to_csv(out / "physical.csv", index=False)
        cam_rows = []
        for fd, c in zip(self.analysis.frames, self.cameras):
            if c is None:
                cam_rows.append({"frame": fd.index, "t": fd.t, "valid": False})
            else:
                d = c.to_dict()
                cam_rows.append({"frame": fd.index, "t": fd.t, "valid": True, "pan": d["pan"], "tilt": d["tilt"],
                                 "roll": d["roll"], "focal_px": d["f"], "cam_x": d["C"][0], "cam_y": d["C"][1],
                                 "cam_z": d["C"][2], "line_alignment": c.info.get("align")})
        pd.DataFrame(cam_rows).to_csv(out / "cameras.csv", index=False)
        with open(out / "summary.json", "w") as f:
            json.dump(self.summary(), f, indent=2)
        paths.update({k: str(out / v) for k, v in (("csv", "tracking.csv"), ("physical", "physical.csv"),
                                                   ("cameras", "cameras.csv"), ("summary", "summary.json"))})
        return paths


def build(an: Analysis, cfg: Config | None = None) -> Result:
    cfg = cfg or Config()
    if not an.frames:
        raise ValueError(f"No se leyó ningún fotograma de {an.video} (¿start_s más allá del final? ¿vídeo dañado?)")
    tim = {}
    t0 = time.time()
    cams, segs = solve_cameras(an, cfg)
    tim["calibration_s"] = time.time() - t0
    t0 = time.time()
    trk = track_players(an, cams, cfg)
    tim["tracking_s"] = time.time() - t0
    ball_pos, ball_det = track_ball(an, cams)

    frame_t = np.array([fd.t for fd in an.frames])
    # lookup analysed-frame index -> time (identities store analysed-frame indices)
    t_first, t_last = frame_t[0], frame_t[-1]
    out_t = np.arange(t_first, t_last + 1e-9, 1.0 / cfg.out_fps)
    near = np.clip(np.searchsorted(frame_t, out_t), 0, len(frame_t) - 1)
    prev = np.clip(near - 1, 0, len(frame_t) - 1)
    near = np.where(np.abs(frame_t[prev] - out_t) < np.abs(frame_t[near] - out_t), prev, near)
    valid = np.array([cams[i] is not None for i in near])

    rows = []
    for ident in trk.identities:
        s = sample_identity(ident, frame_t, out_t, cfg.extrapolate, cfg.max_extrapolate_s, motion=trk.motion)
        ok = np.isfinite(s.pos).all(1)
        # no data while the broadcast shows something we cannot calibrate, unless filling is on
        if not cfg.extrapolate:
            ok &= valid
        # A filled-in position that the camera was looking at would have been detected:
        # such "ghosts" (usually a fragment of another identity) are not output.
        far = ok & np.isin(s.kind, (LONG_GAP, EDGE))
        for k in np.nonzero(far)[0]:
            c = cams[near[k]]
            if c is not None and _in_view(c, s.pos[k]):
                ok[k] = False
        sp = speeds(out_t, np.where(ok[:, None], s.pos, np.nan))
        team = "A" if ident.team == 0 else "B" if ident.team == 1 else "other"
        num = ident.number if ident.number is not None else -1
        for k in np.nonzero(ok)[0]:
            rows.append((k, out_t[k], ident.pid, num, team, ident.role, s.pos[k, 0], s.pos[k, 1], bool(s.detected[k]), sp[k]))
    table = pd.DataFrame(rows, columns=["frame", "t", "player_id", "shirt_number", "team", "role", "x", "y", "is_detected",
                                       "speed_ms"])
    if len(table):
        table["timestamp"] = table["t"] - t_first + cfg.time_offset_s
        table["speed_kmh"] = table["speed_ms"] * 3.6
    phys = physical_summary(table[table["role"].isin(["player", "goalkeeper"])]) if len(table) else pd.DataFrame()
    return Result(an, cfg, cams, segs, trk, ball_pos, ball_det, out_t, table, phys, tim)


# settings that change what the slow pass computes (everything else is re-done in seconds)
PASS1_KEYS = ("start_s", "max_seconds", "stride", "keyframe_every", "det_model", "det_imgsz", "det_conf", "ball_conf",
              "jersey_ocr", "jersey_crops", "cut_threshold")


def _in_view(cam, xy, margin: float = 0.02) -> bool:
    uv, z = cam.ground_to_image(np.asarray(xy, float)[None])
    mx, my = margin * cam.width, margin * cam.height
    return bool(z[0] > 0 and mx < uv[0, 0] < cam.width - mx and my < uv[0, 1] < cam.height - my)


def run(video: str | Path, out_dir: str | Path = "salida", cfg: Config | None = None, render: bool = True,
        reuse_analysis: bool = True, progress: bool = True) -> Result:
    """Full pipeline. Re-uses ``out_dir/analysis.pkl.gz`` if it exists (delete it to redo)."""
    cfg = cfg or Config()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "analysis.pkl.gz"
    an = None
    if reuse_analysis and cache.exists():
        an = Analysis.load(cache)
        new = cfg.to_dict()
        changed = [k for k in PASS1_KEYS if an.config.get(k) != new.get(k)]
        if Path(an.video).name != Path(video).name:
            changed.append("video")
        if changed:
            if progress:
                print(f"El análisis guardado se hizo con otros ajustes ({', '.join(changed)}): se repite.")
            an = None
        else:
            an.video = str(video)  # the file may have been moved since
            if progress:
                print(f"Reutilizando análisis guardado: {cache}")
    if an is None:
        an = analyze(video, cfg, progress=progress)
        an.save(cache)
    res = build(an, cfg)
    res.save(out)
    if render:
        from .viz import render_video

        render_video(res, out / "verificacion.mp4", progress=progress)
    return res


__all__ = ["Config", "Result", "analyze", "build", "run", "OTHER"]
