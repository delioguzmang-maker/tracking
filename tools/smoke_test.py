"""Smoke test: the whole chain on the first few seconds of a video, before any long run.

    python tools/smoke_test.py video.mp4 [--seconds 5] [--out smoke]

Runs ``soccercal.run`` (detection, calibration, tracking, second look, files, verification video)
on ``--seconds`` of the video, then the tactical layer on the result (space models, SoccerMap,
unseen players, space occupation), and checks that every output exists and makes sense. Prints
one line per check and exits with status 1 if any fails: a few minutes instead of finding the
problem hours into an evaluation or a training.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FILES = ("tracking.csv", "ball.csv", "physical.csv", "cameras.csv", "summary.json", "analysis.pkl.gz")


def check_outputs(out: Path, render: bool = True) -> list[tuple[str, bool, str]]:
    """[(check, passed, detail)] for a ``soccercal.run`` output folder."""
    import pandas as pd

    res = []

    def step(name, fn):
        t0 = time.time()
        try:
            detail = fn() or ""
            res.append((name, True, f"{detail} ({time.time() - t0:.1f} s)".strip()))
        except Exception as e:  # noqa: BLE001 - every failure is reported, not raised
            res.append((name, False, f"{type(e).__name__}: {e}"))
            traceback.print_exc()

    def files():
        need = FILES + (("verificacion.mp4",) if render else ())
        missing = [f for f in need if not (out / f).exists()]
        missing += [p for p in ("*_match.json", "*_tracking_extrapolated.jsonl") if not list(out.glob(p))]
        assert not missing, f"missing {missing}"
        return f"{len(need) + 2} files"

    def tracking():
        t = pd.read_csv(out / "tracking.csv")
        assert len(t), "no rows"
        for c in ("frame", "player_id", "team", "x", "y", "is_detected"):
            assert c in t, f"no column {c}"
        seen = t[t["is_detected"].astype(bool)]
        assert len(seen), "nobody seen"
        assert seen[["x", "y"]].abs().max().max() < 80, "positions far outside the pitch"
        per = seen.groupby("frame").size()
        assert per.median() >= 4, f"only {per.median():.0f} players seen per frame"
        teams = set(t.loc[t["role"].isin(["player", "goalkeeper"]), "team"].dropna()) - {"other"}
        assert len(teams) == 2, f"teams found: {sorted(teams)}"
        return f"{t['player_id'].nunique()} ids, median {per.median():.0f} seen per frame"

    def ball():
        b = pd.read_csv(out / "ball.csv")
        for c in ("frame", "x", "y", "kind", "possession_team"):
            assert c in b, f"no column {c}"
        assert set(b["kind"].dropna()) <= {"detected", "interpolated", "carried"}, "unknown kind"
        known = b["x"].notna().mean()
        return f"ball known {known:.0%}"

    m = {}

    def tactical():
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from soccercal.tactics import Match, plot_control

        m["m"] = Match.load(out)
        k = int(m["m"].frames[len(m["m"].frames) // 2])
        for model in ("spearman", "fernandez", "value", "soccermap"):
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_control(m["m"], k, model=model, ax=ax)
            plt.close(fig)
        return "4 space maps"

    def soccermap():
        from soccercal.soccermap import pass_success

        mm = m["m"]
        v = pass_success(mm.snapshot(int(mm.frames[len(mm.frames) // 2]))).values
        assert v.shape == (68, 104) and np.isfinite(v).all() and 0 <= v.min() and v.max() <= 1
        return f"pass success {v.min():.2f}-{v.max():.2f}"

    def impute():
        from soccercal import impute

        c = impute.complete(m["m"])
        n = int(c.table["imputed"].sum()) if "imputed" in c.table else 0
        return f"{n} rows of unseen players"

    def space():
        from soccercal import space

        s = space.summary(m["m"], 0)
        return f"{len(s)} players with space gain"

    step("output files", files)
    step("tracking.csv", tracking)
    step("ball.csv", ball)
    step("space maps", tactical)
    if m:
        step("SoccerMap", soccermap)
        step("unseen players", impute)
        step("space occupation", space)
    return res


def smoke(video: str | Path, out: str | Path, seconds: float = 5.0, render: bool = True, cfg=None) -> bool:
    """Run the chain on ``seconds`` of ``video`` into ``out`` and print the checks. True if all pass."""
    from dataclasses import replace

    import soccercal
    from soccercal.config import Config

    out = Path(out)
    t0 = time.time()
    ok_run = True
    try:
        soccercal.run(video, out, replace(cfg or Config(), max_seconds=seconds), render=render, progress=False)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        ok_run = False
    rows = [("soccercal.run", ok_run, f"{seconds:g} s of video ({time.time() - t0:.0f} s)")]
    if ok_run:
        rows += check_outputs(out, render)
    for name, ok, detail in rows:
        print(f"{'OK ' if ok else 'FAIL'}  {name:18s} {detail}", flush=True)
    return all(ok for _, ok, _ in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--out", default="smoke")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    sys.exit(0 if smoke(a.video, a.out, a.seconds, not a.no_render) else 1)


if __name__ == "__main__":
    main()
