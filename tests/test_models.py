"""SoccerMap network, space occupation, unseen-player imputation and the OSNet embedder."""
import numpy as np
import pytest

from soccercal.tactics import Match, Snapshot


def _snap(attacks_right=(True, False)):
    rng = np.random.default_rng(0)
    pos = rng.uniform([-40, -25], [40, 25], (22, 2))
    team = np.repeat([0, 1], 11)
    gk = np.zeros(22, bool)
    gk[[0, 11]] = True
    return Snapshot(0, 1, pos, rng.normal(0, 2, (22, 2)), team, gk, np.ones(22, bool), np.arange(22),
                    np.array([0.0, 0.0]), 0, 5, None, attacks_right)


def test_soccermap_network_and_orientation():
    torch = pytest.importorskip("torch")
    from soccercal.soccermap import SoccerMapNet, pass_success

    net = SoccerMapNet(13).eval()
    assert net(torch.zeros(2, 13, 68, 104)).shape == (2, 1, 68, 104)
    right = pass_success(_snap((True, False)), model=net)
    assert right.values.shape == (68, 104) and 0 <= right.values.min() and right.values.max() <= 1
    # the same situation seen from the other end of the pitch gives the rotated surface
    s = _snap((False, True))
    s.pos, s.vel = -s.pos, -s.vel
    left = pass_success(s, model=net)
    assert np.allclose(left.values, right.values[::-1, ::-1], atol=1e-5)


def test_match_template_finds_the_missing_place():
    from soccercal.impute import match_template

    rng = np.random.default_rng(1)
    base = np.array([[-3, -2], [-3, -0.7], [-3, 0.7], [-3, 2], [0, -2], [0, -0.7], [0, 0.7], [0, 2], [3, -1], [3, 1]], float)
    lib = np.stack([(base + rng.normal(0, 0.05, base.shape) - base.mean(0)) / base.std(0) for _ in range(8)])
    true = (base - base.mean(0)) / base.std(0)
    for gone in (0, 7, 9):
        keep = np.setdiff1d(np.arange(10), [gone])
        seen = base[keep]
        obs = (seen - seen.mean(0)) / seen.std(0) * 0.9  # observed shapes come out a bit shrunk
        slot, cost = match_template(obs, lib, 1)
        est = slot[0] / 0.9 * seen.std(0) + seen.mean(0)
        assert np.linalg.norm(est - base[gone]) < 0.3, (gone, est, base[gone], true[gone])


def _match_without(gk_team: int, n_out: int = 9):
    import pandas as pd

    rows = []
    for k in range(30):
        for team, sgn in ((0, 1), (1, -1)):
            xs = np.linspace(-30, 10, 10)[: (n_out if team == gk_team else 10)] * sgn
            for j, x in enumerate(xs):
                rows.append((k, 1, 100 * team + j + 1, team, False, x + 0.1 * k, (j % 4 - 1.5) * 12, True, 1.0, 0.0))
            if team != gk_team:
                rows.append((k, 1, 100 * team, team, True, -48.0 * sgn, 0.0, True, 0.0, 0.0))
    tab = pd.DataFrame(rows, columns=["frame", "period", "player_id", "team", "gk", "x", "y", "is_detected", "vx", "vy"])
    tab["t"] = tab["frame"] / 10
    ball = pd.DataFrame({"frame": range(30), "x": 0.0, "y": 0.0, "possession_team": 0,
                         "possession_player_id": 5}).set_index("frame")
    return Match(tab, ball, {}, 105.0, 68.0, ["left_to_right"])


def test_complete_adds_flagged_players(monkeypatch):
    from soccercal import impute

    rng = np.random.default_rng(0)
    fake = {"gk_W1": rng.normal(0, 0.1, (5, 32)), "gk_b1": np.zeros(32), "gk_W2": np.zeros((32, 2)),
            "gk_b2": np.array([-0.85, 0.0]), "templates": np.stack([np.c_[np.linspace(-1.5, 1.5, 10), np.zeros(10)]] * 3)}
    monkeypatch.setattr(impute, "_M", fake)
    m = _match_without(gk_team=1)
    out = impute.complete(m)
    add = out.table[out.table["imputed"]]
    assert set(add["team"]) == {1} and add["player_id"].lt(0).all() and (~add["is_detected"]).all()
    gk = add[add["gk"]]
    assert len(gk) == 30 and np.allclose(gk["x"], 0.85 * 52.5)  # away attacks -x: defends the +x goal
    assert out.snapshot(5).imputed.sum() == 2  # the goalkeeper and the tenth outfield player


def test_space_occupation_runs_on_a_match(monkeypatch):
    from soccercal import space

    rng = np.random.default_rng(0)
    monkeypatch.setattr(space, "_W", {"W1": rng.normal(0, 1, (4, 8)), "b1": np.zeros(8), "W2": rng.normal(0, 1, (8, 1)),
                                      "b2": np.zeros(1)})
    m = _match_without(gk_team=-1, n_out=10)
    occ = space.space_occupation(m, 0, window_s=1.0, frames=[0, 10])
    assert set(occ["player_id"]) <= set(m.table.loc[m.table["team"] == 0, "player_id"])
    assert {"gain", "kind", "active", "in_front", "dist_ball"} <= set(occ.columns)
    # every player of team 0 runs at 1 m/s: active only above a jog (1.5 m/s)
    assert not occ["active"].any()
    q = space.space_quality(m.snapshot(3), 0)
    assert len(q) == 11 and all(v >= 0 for v in q.values())


def test_osnet_architecture_matches_the_published_weights():
    torch = pytest.importorskip("torch")
    from soccercal import reid
    from soccercal.weights import cache_dir

    path = cache_dir() / "osnet_x0_25_msmt17.pth"
    if not path.exists():
        pytest.skip("OSNet weights not downloaded")
    e = reid.Embedder("x0_25")  # load_state_dict(strict=True) inside: every layer name and shape must match
    f = e([np.full((120, 50, 3), 128, np.uint8), np.zeros((80, 30, 3), np.uint8)])
    assert f.shape == (2, 512) and np.allclose(np.linalg.norm(f, axis=1), 1, atol=1e-5)
    assert isinstance(e.net, torch.nn.Module)
