"""Open football tracking + event data in one format, to train soccercal's tactical models.

Sources (downloaded on demand into ``--cache``):

* SkillCorner open data: broadcast tracking at 10 fps (the same kind of data soccercal produces,
  off-camera players extrapolated) + dynamic events, 20 matches, MIT licence.
  https://github.com/SkillCorner/opendata
* Metrica Sports sample data: optical tracking (25 fps) + events, games 1 and 2.
  https://github.com/metrica-sports/sample-data (anonymised; "acknowledge the source")
* IDSSE: 7 Bundesliga matches, official DFL optical tracking (25 fps) + events, CC BY 4.0
  (Bassek et al., Scientific Data 2025), loaded with kloppy.

Every match becomes a ``soccercal.tactics.Match`` (10 fps, metres, pitch-centred, positions and
velocities of the 22 players, ball, possession) plus a ``passes`` table: frame, team, passer, x0, y0
(the ball when it is played), x1, y1 (where it is next controlled: the receiver or the opponent who
wins it), success. ``python tools/open_data.py`` builds ``matches/*.pkl`` in the cache.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soccercal.physical import velocities  # noqa: E402
from soccercal.tactics import Match  # noqa: E402

FPS = 10.0
IDSSE = ["J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9"]


# ----------------------------------------------------------------------------------------- helpers
def _side_from_gk(tab: pd.DataFrame) -> list:
    """home_team_side per period from where the home goalkeeper stands."""
    out = []
    for per in sorted(tab["period"].unique()):
        g = tab[(tab["period"] == per) & tab["gk"]]
        hx = g.loc[g["team"] == 0, "x"].mean()
        ax = g.loc[g["team"] == 1, "x"].mean()
        home_left = hx < 0 if np.isfinite(hx) else (ax > 0 if np.isfinite(ax) else True)
        out.append("left_to_right" if home_left else "right_to_left")
    return out


def _table(frames: np.ndarray, period: np.ndarray, X: np.ndarray, Y: np.ndarray, pids: list, team: list,
           gk: list) -> pd.DataFrame:
    """Long player table (10 fps) from (T, P) position matrices; velocities per player and period."""
    rows = []
    for j, pid in enumerate(pids):
        ok = np.isfinite(X[:, j]) & np.isfinite(Y[:, j])
        if not ok.any():
            continue
        vx = np.full(len(frames), np.nan)
        vy = np.full(len(frames), np.nan)
        for per in np.unique(period):
            m = period == per
            v = velocities(frames[m] / FPS, np.stack([X[m, j], Y[m, j]], 1))
            vx[m], vy[m] = v[:, 0], v[:, 1]
        rows.append(pd.DataFrame({"frame": frames[ok], "period": period[ok], "player_id": pid, "team": team[j],
                                  "gk": gk[j], "x": X[ok, j], "y": Y[ok, j], "is_detected": True,
                                  "vx": vx[ok], "vy": vy[ok]}))
    tab = pd.concat(rows, ignore_index=True)
    tab["t"] = tab["frame"] / FPS
    return tab


def _resample(t: np.ndarray, period: np.ndarray, values: np.ndarray, period_start_frame: dict) -> tuple:
    """Linear resampling of (T, K) values to 10 fps within each period (gaps stay NaN)."""
    out_f, out_p, out_v = [], [], []
    for per in np.unique(period):
        m = period == per
        tt = t[m]
        grid = np.arange(np.ceil(tt[0] * FPS) / FPS, tt[-1] + 1e-9, 1 / FPS)
        V = np.full((len(grid), values.shape[1]), np.nan)
        idx = np.clip(np.searchsorted(tt, grid), 0, len(tt) - 1)
        for k in range(values.shape[1]):
            v = values[m, k]
            ok = np.isfinite(v)
            if ok.sum() < 2:
                continue
            V[:, k] = np.interp(grid, tt[ok], v[ok], left=np.nan, right=np.nan)
            V[~np.isfinite(v[idx]), k] = np.nan  # do not bridge gaps (player off the pitch)
        out_f.append(period_start_frame[per] + np.round(grid * FPS).astype(int))
        out_p.append(np.full(len(grid), per))
        out_v.append(V)
    return np.concatenate(out_f), np.concatenate(out_p), np.vstack(out_v)


def _next_control(passes: pd.DataFrame, controls: pd.DataFrame, max_s: float = 6.0) -> pd.DataFrame:
    """Destination = where the ball is next controlled (``controls``: frame, x, y) within ``max_s``."""
    c = controls.sort_values("frame")
    cf = c["frame"].values
    x1, y1 = [], []
    for f in passes["frame"].values:
        k = np.searchsorted(cf, f, side="right")
        if k < len(cf) and cf[k] - f <= max_s * FPS:
            x1.append(c["x"].values[k])
            y1.append(c["y"].values[k])
        else:
            x1.append(np.nan)
            y1.append(np.nan)
    passes = passes.copy()
    passes["x1"], passes["y1"] = x1, y1
    return passes


# -------------------------------------------------------------------------------------- SkillCorner
def skillcorner(root: Path, skip=()):
    repo = root / "skillcorner"
    if not repo.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/SkillCorner/opendata.git", str(repo)],
                       check=True, env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"})
    for d in sorted((repo / "data" / "matches").iterdir()):
        mid = d.name
        if mid in skip:
            continue
        trk = d / f"{mid}_tracking_extrapolated.jsonl"
        if trk.read_bytes()[:20].startswith(b"version https"):  # Git LFS pointer: fetch the file
            url = f"https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/{mid}/{trk.name}"
            subprocess.run(["curl", "-sSL", "-o", str(trk), url], check=True)
        m = Match.load(d)
        ev = pd.read_csv(d / f"{mid}_dynamic_events.csv", low_memory=False)
        meta = json.loads((d / f"{mid}_match.json").read_text())
        home = meta["home_team"]["id"]
        pp = ev[ev["event_type"] == "player_possession"].sort_values("frame_start")
        p = pp[(pp["end_type"] == "pass") & pp["pass_outcome"].isin(["successful", "unsuccessful"])]
        passes = pd.DataFrame({"frame": p["frame_end"].values, "team": np.where(p["team_id"].values == home, 0, 1),
                               "passer": p["player_id"].values,
                               "success": (p["pass_outcome"] == "successful").values.astype(int)})
        b = m.ball.reindex(passes["frame"].values)
        passes["x0"] = np.where(np.isfinite(b["x"].values), b["x"].values, p["x_end"].values)
        passes["y0"] = np.where(np.isfinite(b["y"].values), b["y"].values, p["y_end"].values)
        controls = pd.DataFrame({"frame": pp["frame_start"].values, "x": pp["x_start"].values, "y": pp["y_start"].values})
        passes = _next_control(passes, controls)
        yield "skillcorner", mid, m, passes
        print(f"skillcorner {mid}: {len(m.frames)} frames, {len(passes)} passes", flush=True)


# ------------------------------------------------------------------------------------------ Metrica
def _metrica_csv(path: Path):
    with open(path) as f:
        f.readline()
        f.readline()
        names = f.readline().strip().split(",")
    df = pd.read_csv(path, skiprows=3, header=None)
    return df, [(n, k) for k, n in enumerate(names) if n.startswith("Player") or n == "Ball"]


def metrica(root: Path, skip=()):
    repo = root / "metrica"
    if not repo.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/metrica-sports/sample-data.git", str(repo)],
                       check=True)
    for g in (1, 2):
        if f"game{g}" in skip:
            continue
        d = repo / "data" / f"Sample_Game_{g}"
        H, hc = _metrica_csv(d / f"Sample_Game_{g}_RawTrackingData_Home_Team.csv")
        A, ac = _metrica_csv(d / f"Sample_Game_{g}_RawTrackingData_Away_Team.csv")
        t = H[2].values.astype(float)
        period = H[0].values.astype(int)
        pids, team, Xs, Ys = [], [], [], []
        for df, cols, tm, off in ((H, hc, 0, 0), (A, ac, 1, 100)):
            for n, k in cols:
                if n == "Ball":
                    bx, by = (df[k].values - 0.5) * 105, -(df[k + 1].values - 0.5) * 68
                    continue
                pids.append(off + int(n[6:]))
                team.append(tm)
                Xs.append((df[k].values - 0.5) * 105)
                Ys.append(-(df[k + 1].values - 0.5) * 68)
        X, Y = np.stack(Xs, 1), np.stack(Ys, 1)
        # goalkeeper: the player of each team furthest from the halfway line at kick-off
        first = np.isfinite(X[0])
        gk = [False] * len(pids)
        for tm in (0, 1):
            js = [j for j in range(len(pids)) if team[j] == tm and first[j]]
            gk[max(js, key=lambda j: abs(X[0, j]))] = True
        start = {1: 0, 2: 100000}
        frames, per, V = _resample(t, period, np.hstack([X, Y, bx[:, None], by[:, None]]), start)
        n = len(pids)
        tab = _table(frames, per, V[:, :n], V[:, n:2 * n], pids, team, gk)
        ev = pd.read_csv(d / f"Sample_Game_{g}_RawEventsData.csv")
        ev["frame"] = ev["Period"].map(start) + np.round(ev["Start Time [s]"] * FPS).astype(int)
        ev["tm"] = (ev["Team"] == "Away").astype(int)
        # possession: the team of the last on-ball event (a lost ball goes to the other team)
        onball = ev[ev["Type"].isin(["PASS", "SHOT", "BALL LOST", "RECOVERY", "SET PIECE"])].sort_values("frame")
        poss_team = np.full(len(frames), -1)
        k = np.searchsorted(onball["frame"].values, frames, side="right") - 1
        ok = k >= 0
        poss_team[ok] = np.where(onball["Type"].values[k[ok]] == "BALL LOST", 1 - onball["tm"].values[k[ok]],
                                 onball["tm"].values[k[ok]])
        ball = pd.DataFrame({"frame": frames, "x": V[:, 2 * n], "y": V[:, 2 * n + 1], "possession_team": poss_team,
                             "possession_player_id": -1}).set_index("frame")
        m = Match(tab, ball, {}, 105.0, 68.0, _side_from_gk(tab), ("Home", "Away"), ("#d62728", "#1f4fb4"), FPS)
        lost = ev["Type"].eq("BALL LOST") & ev["Subtype"].fillna("").str.contains("INTERCEPTION")
        p = ev[ev["Type"].eq("PASS") | lost]
        passes = pd.DataFrame({"frame": p["frame"].values, "team": p["tm"].values,
                               "passer": [100 * t_ + int(s[6:]) for t_, s in zip(p["tm"].values, p["From"].values)],
                               "x0": (p["Start X"].values - 0.5) * 105, "y0": -(p["Start Y"].values - 0.5) * 68,
                               "x1": (p["End X"].values - 0.5) * 105, "y1": -(p["End Y"].values - 0.5) * 68,
                               "success": p["Type"].eq("PASS").values.astype(int)})
        yield "metrica", f"game{g}", m, passes
        print(f"metrica game {g}: {len(m.frames)} frames, {len(passes)} passes", flush=True)


# -------------------------------------------------------------------------------------------- IDSSE
def idsse(root: Path, skip=(), match_ids=IDSSE):
    from kloppy import sportec

    for mid in match_ids:
        if mid in skip:
            continue
        tr = sportec.load_open_tracking_data(match_id=mid, coordinates="secondspectrum")
        ev = sportec.load_open_event_data(match_id=mid, coordinates="secondspectrum")
        home_team = tr.metadata.teams[0]
        players = [p for tm in tr.metadata.teams for p in tm.players]
        team_of = {p.player_id: (0 if p.team == home_team else 1) for p in players}
        gk_of = {p.player_id: bool(p.starting_position and p.starting_position.code == "GK") for p in players}
        recs = tr.records
        t = np.array([r.timestamp.total_seconds() for r in recs])
        period = np.array([r.period.id for r in recs])
        pids = [p.player_id for p in players]
        col = {p: j for j, p in enumerate(pids)}
        X = np.full((len(recs), len(pids)), np.nan)
        Y = np.full((len(recs), len(pids)), np.nan)
        B = np.full((len(recs), 2), np.nan)
        own = np.full(len(recs), -1.0)
        for i, r in enumerate(recs):
            for pl, pd_ in r.players_data.items():
                if pd_.coordinates is not None:
                    X[i, col[pl.player_id]] = pd_.coordinates.x
                    Y[i, col[pl.player_id]] = pd_.coordinates.y
            if r.ball_coordinates is not None:
                B[i] = r.ball_coordinates.x, r.ball_coordinates.y
            if r.ball_owning_team is not None:
                own[i] = 0 if r.ball_owning_team == home_team else 1
            if r.ball_state is not None and r.ball_state.value == "dead":
                own[i] = -1
        start = {1: 0, 2: 100000, 3: 200000, 4: 300000}
        frames, per, V = _resample(t, period, np.hstack([X, Y, B]), start)
        _, _, O = _resample(t, period, own[:, None], start)
        n = len(pids)
        ints = {p: k for k, p in enumerate(pids)}
        tab = _table(frames, per, V[:, :n], V[:, n:2 * n], [ints[p] for p in pids], [team_of[p] for p in pids],
                     [gk_of[p] for p in pids])
        poss = np.where(np.isfinite(O[:, 0]), np.round(O[:, 0]), -1).astype(int)
        ball = pd.DataFrame({"frame": frames, "x": V[:, 2 * n], "y": V[:, 2 * n + 1], "possession_team": poss,
                             "possession_player_id": -1}).set_index("frame")
        m = Match(tab, ball, {}, tr.metadata.pitch_dimensions.pitch_length or 105.0,
                  tr.metadata.pitch_dimensions.pitch_width or 68.0, _side_from_gk(tab), (mid + "_home", mid + "_away"),
                  ("#d62728", "#1f4fb4"), FPS)
        rows, controls = [], []
        for e in ev.records:
            if e.coordinates is None:
                continue
            f = start[e.period.id] + int(round(e.timestamp.total_seconds() * FPS))
            controls.append((f, e.coordinates.x, e.coordinates.y))
            if e.event_type.name == "PASS" and e.result is not None and e.result.name in ("COMPLETE", "INCOMPLETE"):
                rc = e.receiver_coordinates
                rows.append((f, 0 if e.team == home_team else 1, ints.get(e.player.player_id, -1) if e.player else -1,
                             e.coordinates.x, e.coordinates.y, rc.x if rc else np.nan, rc.y if rc else np.nan,
                             int(e.result.name == "COMPLETE")))
        passes = pd.DataFrame(rows, columns=["frame", "team", "passer", "x0", "y0", "x1", "y1", "success"])
        nxt = _next_control(passes, pd.DataFrame(controls, columns=["frame", "x", "y"]).drop_duplicates("frame"))
        miss = ~np.isfinite(passes["x1"].values)
        passes.loc[miss, "x1"] = nxt.loc[miss, "x1"].values
        passes.loc[miss, "y1"] = nxt.loc[miss, "y1"].values
        yield "idsse", mid, m, passes
        print(f"idsse {mid}: {len(m.frames)} frames, {len(passes)} passes", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="open_data")
    ap.add_argument("--sources", default="skillcorner,metrica,idsse")
    a = ap.parse_args()
    root = Path(a.cache)
    (root / "matches").mkdir(parents=True, exist_ok=True)
    for src in a.sources.split(","):
        done = {p.stem.split("_", 1)[1] for p in (root / "matches").glob(f"{src}_*.pkl")}  # resume
        for name, mid, m, passes in {"skillcorner": skillcorner, "metrica": metrica, "idsse": idsse}[src](root, done):
            with open(root / "matches" / f"{name}_{mid}.pkl", "wb") as f:
                pickle.dump({"source": name, "match": mid, "m": m, "passes": passes}, f)


if __name__ == "__main__":
    main()
