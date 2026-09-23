"""End-to-end after detection: simulated broadcast -> tracking -> SkillCorner files -> kloppy.

Uses the synthetic benchmark (real Metrica trajectories) when its data can be
downloaded; otherwise the test is skipped.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


@pytest.fixture(scope="module")
def result():
    sb = pytest.importorskip("synthetic_benchmark")
    try:
        an, cams, gt = sb.simulate(start_s=900.0, minutes=0.4, seed=0)
    except Exception as e:  # no network
        pytest.skip(f"Metrica sample data unavailable: {e}")
    from soccercal.config import Config

    # build() calls solve_cameras on real keyframes; here the cameras are given
    import soccercal.pipeline as pl

    cfg = Config()
    orig = pl.solve_cameras
    pl.solve_cameras = lambda an_, cfg_: (cams, [])
    try:
        res = pl.build(an, cfg)
    finally:
        pl.solve_cameras = orig
    return res, gt


def test_identities_are_pure(result):
    res, gt = result
    counts = {}
    for i, asg in enumerate(res.tracking.assignments):
        for j, pid in asg.items():
            g = gt["det_gt"][i][j]
            if g is not None:
                counts.setdefault(pid, []).append(g)
    purity = sum(max(v.count(x) for x in set(v)) for v in counts.values()) / sum(len(v) for v in counts.values())
    assert purity > 0.97


def test_skillcorner_files_load_in_kloppy(result, tmp_path):
    res, _ = result
    paths = res.save(tmp_path)
    m = json.load(open(paths["match_json"]))
    for k in ("home_team", "away_team", "players", "pitch_length", "pitch_width", "ball", "referees"):
        assert k in m
    ids = {p["id"] for p in m["players"]}
    with open(paths["tracking_jsonl"]) as f:
        frames = [json.loads(line) for line in f]
    assert len(frames) == len(res.out_t)
    fr = frames[len(frames) // 2]
    assert set(fr) == {"frame", "timestamp", "period", "ball_data", "possession", "image_corners_projection", "player_data"}
    for p in fr["player_data"]:
        assert set(p) == {"x", "y", "player_id", "is_detected"} and p["player_id"] in ids
    pytest.importorskip("kloppy")
    from soccercal.skillcorner import load_kloppy

    ds = load_kloppy(tmp_path)
    assert len(ds.records) > 0 and ds.metadata.frame_rate == 10
    assert len(ds.metadata.teams[0].players) > 0


def test_physical_metrics_are_sane(result):
    res, _ = result
    ph = res.physical
    assert len(ph) > 0
    assert (ph["top_speed_kmh"].dropna() < 40).all()
    band_sum = ph[[c for c in ph.columns if c.startswith("dist_")]].sum(axis=1)
    assert np.allclose(band_sum, ph["distance_m"], rtol=1e-6)
