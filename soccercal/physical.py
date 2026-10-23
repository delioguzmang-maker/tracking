"""Physical metrics per player, SkillCorner-style speed bands."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

BANDS_KMH = [("walking", 0, 7), ("jogging", 7, 15), ("running", 15, 20), ("hsr", 20, 25), ("sprint", 25, 1e9)]


def velocities(t: np.ndarray, xy: np.ndarray, window_s: float = 0.7, vmax: float = 12.0) -> np.ndarray:
    """(K, 2) velocity (m/s) from positions sampled at a constant rate; NaN where positions are NaN.

    Savitzky-Golay derivative over each run of known positions. As in Laurie Shaw's
    ``calc_player_velocities`` (LaurieOnTracking), a step faster than ``vmax`` is a tracking glitch,
    not a sprint: the run is cut there, and a speed above ``vmax`` is NaN (never clipped to it)."""
    out = np.full((len(t), 2), np.nan)
    ok = np.isfinite(xy).all(1)
    if ok.sum() < 3:
        return out
    dt = float(np.median(np.diff(t)))
    idx = np.nonzero(ok)[0]
    step = np.hypot(*np.diff(xy[idx], axis=0).T) / (np.diff(idx) * dt)
    breaks = np.nonzero((np.diff(idx) > 1) | (step > vmax))[0]
    for run in np.split(idx, breaks + 1):
        w = min(max(3, int(round(window_s / dt)) | 1), len(run) if len(run) % 2 else len(run) - 1)
        if w < 3:
            continue
        out[run, 0] = savgol_filter(xy[run, 0], w, 2, deriv=1, delta=dt)
        out[run, 1] = savgol_filter(xy[run, 1], w, 2, deriv=1, delta=dt)
    out[np.hypot(out[:, 0], out[:, 1]) > vmax] = np.nan
    return out


def speeds(t: np.ndarray, xy: np.ndarray, window_s: float = 0.7, vmax: float = 12.0) -> np.ndarray:
    """Speed (m/s), see ``velocities``."""
    v = velocities(t, xy, window_s, vmax)
    return np.hypot(v[:, 0], v[:, 1])


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
