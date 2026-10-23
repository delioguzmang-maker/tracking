"""Tactical models (``soccercal.tactics``) against their reference implementations."""
import numpy as np
import pytest

from soccercal.tactics import (Match, Snapshot, default_params, epv_at, grid, influence, offside_players,
                               pitch_control, soccermap_tensor, space_control, to_metrica, value_map)


def _snap(seed, n=11, ball=(0.0, 0.0), attacking=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform([-50, -32], [50, 32], (2 * n, 2))
    vel = rng.normal(0, 3, (2 * n, 2))
    team = np.repeat([0, 1], n)
    gk = np.zeros(2 * n, bool)
    gk[[0, n]] = True
    pos[0], pos[n] = [-48, 0], [48, 0]
    return Snapshot(0, 1, pos, vel, team, gk, np.ones(2 * n, bool), np.arange(2 * n), np.array(ball, float),
                    attacking, -1, None, (True, False))


def _shaw_at_target(target, att, dfd, ball, p):
    """Laurie Shaw's ``calculate_pitch_control_at_target`` (LaurieOnTracking, Friends of Tracking,
    MIT licence) transcribed line by line; players are (position, velocity, lambda) tuples."""
    def tti(pl):
        r = pl[0] + pl[1] * p["reaction_time"]
        return p["reaction_time"] + np.linalg.norm(target - r) / p["max_player_speed"]

    def prob(T, t):
        return 1 / (1. + np.exp(-np.pi / np.sqrt(3.0) / p["tti_sigma"] * (T - t)))
    tb = 0.0 if np.isnan(ball).any() else np.linalg.norm(target - ball) / p["average_ball_speed"]
    ta, td = [tti(pl) for pl in att], [tti(pl) for pl in dfd]
    tmin_a, tmin_d = np.nanmin(ta), np.nanmin(td)
    if tmin_a - max(tb, tmin_d) >= p["time_to_control_def"]:
        return 0., 1.
    elif tmin_d - max(tb, tmin_a) >= p["time_to_control_att"]:
        return 1., 0.
    A = [(t, pl[2]) for t, pl in zip(ta, att) if t - tmin_a < p["time_to_control_att"]]
    D = [(t, pl[2]) for t, pl in zip(td, dfd) if t - tmin_d < p["time_to_control_def"]]
    dT = np.arange(tb - p["int_dt"], tb + p["max_int_time"], p["int_dt"])
    Pa, Pd = np.zeros_like(dT), np.zeros_like(dT)
    ca, cd = [0.0] * len(A), [0.0] * len(D)
    ptot, i = 0.0, 1
    while 1 - ptot > p["model_converge_tol"] and i < dT.size:
        T = dT[i]
        for j, (t, lam) in enumerate(A):
            ca[j] += (1 - Pa[i - 1] - Pd[i - 1]) * prob(T, t) * lam * p["int_dt"]
            Pa[i] += ca[j]
        for j, (t, lam) in enumerate(D):
            cd[j] += (1 - Pa[i - 1] - Pd[i - 1]) * prob(T, t) * lam * p["int_dt"]
            Pd[i] += cd[j]
        ptot = Pa[i] + Pd[i]
        i += 1
    return Pa[i - 1], Pd[i - 1]


@pytest.mark.parametrize("seed,ball", [(0, (0.0, 0.0)), (1, (20.0, -10.0)), (2, (np.nan, np.nan))])
def test_pitch_control_matches_shaw(seed, ball):
    s = _snap(seed, ball=ball)
    p = default_params()
    lam = lambda i, attacking: p["lambda_att"] if attacking else (p["lambda_gk"] if s.gk[i] else p["lambda_def"])  # noqa: E731
    att = [(s.pos[i], s.vel[i], lam(i, True)) for i in np.nonzero(s.team == 0)[0]]
    dfd = [(s.pos[i], s.vel[i], lam(i, False)) for i in np.nonzero(s.team == 1)[0]]
    pc = pitch_control(s, offsides=False, n_grid_cells_x=21)
    xg, yg = grid(21)
    ref = np.array([[_shaw_at_target(np.array([x, y]), att, dfd, s.ball, p)[0] for x in xg] for y in yg])
    assert np.abs(pc.values - ref).max() < 1e-12
    # either a team is sure to win the ball, or the probabilities sum to (almost) 1
    pd_ = np.array([[_shaw_at_target(np.array([x, y]), att, dfd, s.ball, p)[1] for x in xg] for y in yg])
    assert np.all(pc.values + pd_ > 1 - p["model_converge_tol"] - 1e-9)


def test_offside_attacker_is_ignored():
    s = _snap(3, ball=(0.0, 0.0))
    # team 1 defends +x (its goalkeeper is at x = 48): an attacker alone at x = 45 is offside
    s.pos[s.team == 1, 0] = np.r_[48, np.full(10, 10.0)]
    s.pos[1:11, 0] = np.linspace(-40, 0, 10)  # the other attackers in their own half
    s.pos[5] = [45.0, 5.0]
    s.vel[:] = 0.0
    off = offside_players(s, 0)
    assert off[5] and off.sum() == 1
    on = pitch_control(s, offsides=False, targets=np.array([[40.0, 10.0]])).values[0]
    off_v = pitch_control(s, offsides=True, targets=np.array([[40.0, 10.0]])).values[0]
    assert on > 0.5 > off_v


def test_fernandez_bornn_influence_shape():
    ball = np.array([0.0, 0.0])
    # standing next to the ball: radius 4 m, axes 2 m, influence 1 at the player, exp(-1/2) 2 m away
    I = influence(np.array([[0.0, 0.0]]), np.array([[0.0, 0.0]]), ball, np.array([[0.0, 0.0], [2.0, 0.0], [0.0, -2.0]]))[0]
    assert np.allclose(I, [1.0, np.exp(-0.5), np.exp(-0.5)])
    # far from the ball the radius grows to 10 m
    far = influence(np.array([[30.0, 0.0]]), np.array([[0.0, 0.0]]), ball, np.array([[35.0, 0.0]]))[0, 0]
    assert np.isclose(far, np.exp(-0.5))
    # running: the influence peaks 0.5 s ahead and is stretched along the run
    v = np.array([[6.0, 0.0]])
    I = influence(np.array([[0.0, 0.0]]), v, ball, np.array([[3.0, 0.0], [0.0, 0.0], [3.0, 2.0], [5.5, 0.0]]))[0]
    assert I[0] > I[1] and I[3] > I[2]


def test_space_control_symmetric():
    pos = np.array([[-10.0, 0.0], [10.0, 0.0]])
    s = Snapshot(0, 1, pos, np.zeros((2, 2)), np.array([0, 1]), np.zeros(2, bool), np.ones(2, bool), np.arange(2),
                 np.array([0.0, 20.0]), 0, -1, None, (True, False))
    sf = space_control(s, n_grid_cells_x=105)
    assert 0 < sf.values.min() and sf.values.max() < 1
    assert abs(sf.at((0.0, 5.0)) - 0.5) < 0.02 and sf.at((-10.0, 0.0)) > 0.7 and sf.at((10.0, 0.0)) < 0.3


def test_epv_orientation_and_value_map():
    goal_r, goal_l = np.array([[50.0, 0.0]]), np.array([[-50.0, 0.0]])
    assert epv_at(goal_r, True)[0] > 10 * epv_at(goal_l, True)[0]
    assert np.isclose(epv_at(goal_r, True)[0], epv_at(goal_l, False)[0])
    s = _snap(4)
    v = value_map(s)
    assert v.values.shape == pitch_control(s).values.shape and v.values.max() <= 0.6


def _frames(n=30):
    meta = {"home_team": {"id": 1, "short_name": "H"}, "away_team": {"id": 2, "short_name": "A"},
            "players": [{"id": 10, "team_id": 1, "player_role": {"acronym": "GK"}},
                        {"id": 11, "team_id": 1, "player_role": {"acronym": "UNK"}},
                        {"id": 20, "team_id": 2, "player_role": {"acronym": "GK"}},
                        {"id": 21, "team_id": 2, "player_role": {"acronym": "UNK"}}],
            "home_team_side": ["right_to_left", "left_to_right"], "pitch_length": 105, "pitch_width": 68}
    frames = []
    for k in range(n):
        t = k / 10
        frames.append({"frame": k, "period": 1, "timestamp": f"00:00:{t:05.2f}",
                       "ball_data": {"x": 5.0 * t, "y": 0.0, "z": None, "is_detected": True},
                       "possession": {"player_id": 11 if k < 20 else None, "group": "home team" if k < 20 else None},
                       "image_corners_projection": {"x_top_left": -20, "y_top_left": 39, "x_bottom_left": -10,
                                                    "y_bottom_left": -34, "x_bottom_right": 10, "y_bottom_right": -34,
                                                    "x_top_right": 20, "y_top_right": 39},
                       "player_data": [{"x": 45.0, "y": 0.0, "player_id": 10, "is_detected": False},
                                       {"x": 5.0 * t, "y": -1.0, "player_id": 11, "is_detected": True},
                                       {"x": -45.0, "y": 0.0, "player_id": 20, "is_detected": False},
                                       {"x": 3.0, "y": 2.0 * t, "player_id": 21, "is_detected": True}]})
    return meta, frames


def test_match_from_skillcorner_and_adapters():
    meta, frames = _frames()
    m = Match.from_skillcorner(meta, frames)
    assert m.fps == 10.0 and len(m.frames) == 30 and m.names == ("H", "A")
    s = m.snapshot(15)
    v = dict(zip(s.pid, s.vel))
    assert np.allclose(v[11], [5.0, 0.0], atol=1e-6) and np.allclose(v[21], [0.0, 2.0], atol=1e-6)
    assert s.attacking == 0 and s.carrier == 11 and s.view.shape == (4, 2)
    assert m.snapshot(23).attacking == 0  # last known possession, < 5 s ago
    assert s.attacks_right == (False, True)  # home plays right to left in period 1
    home, away, gks = to_metrica(m)
    assert gks == ("10", "20")
    assert list(home.columns[:2]) == ["Period", "Time [s]"]
    assert {"Home_11_x", "Home_11_vx", "Home_11_speed", "ball_x", "ball_y"} <= set(home.columns)
    assert np.isclose(home.loc[15, "Home_11_speed"], 5.0)
    X = soccermap_tensor(s)
    assert X.shape == (13, 68, 104)
    assert X[0].sum() == 2 and X[3].sum() == 2  # two attackers, two defenders
    # home attacks right to left: the tensor is rotated so that it attacks +x (the ball carrier ran -x... now +x)
    assert X[1].sum() < 0 and X[7][34, -1] < X[7][34, 0]


def test_velocity_glitch_is_nan_not_a_sprint():
    from soccercal.physical import velocities

    t = np.arange(20) / 10
    xy = np.stack([np.arange(20) * 0.3, np.zeros(20)], 1)
    xy[10:, 0] += 5.0  # a 50 m/s jump: an identity switch, not a run
    v = velocities(t, xy)
    sp = np.hypot(v[:, 0], v[:, 1])
    assert np.nanmax(sp) < 3.5 and np.all(np.isfinite(sp[[2, 15]]))
