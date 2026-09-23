"""Physical metrics per player, SkillCorner-style speed bands."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

BANDS_KMH = [("walking", 0, 7), ("jogging", 7, 15), ("running", 15, 20), ("hsr", 20, 25), ("sprint", 25, 1e9)]


def speeds(t: np.ndarray, xy: np.ndarray, window_s: float = 0.7, vmax: float = 12.0) -> np.ndarray:
    """Speed (m/s) from positions sampled at a constant rate; NaN where positions are NaN."""
    out = np.full(len(t), np.nan)
    ok = np.isfinite(xy).all(1)
    if ok.sum() < 3:
        return out
    dt = float(np.median(np.diff(t)))
    # contiguous runs
    idx = np.nonzero(ok)[0]
    breaks = np.nonzero(np.diff(idx) > 1)[0]
    for run in np.split(idx, breaks + 1):
        if len(run) < 3:
            continue
        w = max(3, int(round(window_s / dt)) | 1)
        w = min(w, len(run) if len(run) % 2 else len(run) - 1)
        if w < 3:
            continue
        vx = savgol_filter(xy[run, 0], w, 2, deriv=1, delta=dt)
        vy = savgol_filter(xy[run, 1], w, 2, deriv=1, delta=dt)
        out[run] = np.minimum(np.hypot(vx, vy), vmax)
    return out


def _efforts(v_kmh: np.ndarray, thr: float, dt: float, min_s: float = 1.0) -> int:
    above = np.nan_to_num(v_kmh, nan=0) >= thr
    n, run = 0, 0
    for a in above:
        run = run + 1 if a else 0
        if run == int(round(min_s / dt)):
            n += 1
    return n


def physical_summary(table: pd.DataFrame) -> pd.DataFrame:
    """table: long format with columns player_id, team, t, x, y, is_detected, speed_ms."""
    rows = []
    for pid, g in table.groupby("player_id"):
        g = g.sort_values("t")
        dt = float(np.median(np.diff(g["t"].values))) if len(g) > 1 else 0.1
        v = g["speed_ms"].values
        kmh = v * 3.6
        step = np.nan_to_num(v) * dt
        r = {"player_id": pid, "team": g["team"].iloc[0], "role": g["role"].iloc[0],
             "minutes": len(g) * dt / 60.0, "pct_detected": 100.0 * g["is_detected"].mean(),
             "distance_m": float(step.sum())}
        for name, lo, hi in BANDS_KMH:
            m = (kmh >= lo) & (kmh < hi)
            r[f"dist_{name}_m"] = float(step[m].sum())
        vv = v[np.isfinite(v)]
        r["top_speed_kmh"] = float(pd.Series(v).rolling(max(1, int(round(1.0 / dt))), min_periods=1).mean().max() * 3.6) if len(vv) else np.nan
        r["psv99_kmh"] = float(np.percentile(vv, 99) * 3.6) if len(vv) else np.nan
        r["n_hsr_efforts"] = _efforts(kmh, 20, dt)
        r["n_sprints"] = _efforts(kmh, 25, dt)
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["team", "player_id"]).reset_index(drop=True)
