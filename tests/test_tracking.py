"""Pitch-space tracker, re-identification (stitching), teams and gap filling."""
import numpy as np

from soccercal.players import Identity, _trajectory
from soccercal.stitch import StitchConfig, Tracklet, stitch
from soccercal.teams import OTHER, decide, fit_team_model
from soccercal.timeline import DETECTED, EDGE, SHORT_GAP, sample_identity
from soccercal.tracker import PitchTracker, rts_smooth

FPS = 25.0


def _run_tracker(trajs, noise=0.15, seed=0, drop=None):
    rng = np.random.default_rng(seed)
    T = len(trajs[0])
    tr = PitchTracker(FPS)
    R = np.eye(2) * noise ** 2
    for k in range(T):
        z = np.array([t[k] for t in trajs]) + rng.normal(0, noise, (len(trajs), 2))
        keep = np.ones(len(trajs), bool) if drop is None else ~drop[k]
        tr.step(k, z[keep], np.repeat(R[None], keep.sum(), 0), np.full(keep.sum(), 0.9))
    return tr.close()


def test_crossing_players_keep_identity():
    """Two players cross (1 m apart at the crossing) and the tracker must not swap them."""
    t = np.arange(0, 4, 1 / FPS)
    a = np.c_[-10 + 5 * t, 0.5 + 0 * t]
    b = np.c_[10 - 5 * t, -0.5 + 0 * t]
    tracks = _run_tracker([a, b])
    assert len(tracks) == 2
    for tr in tracks:
        xs = np.array([o[1] for o in tr.obs])
        # each track stays on one side of y = 0 (its own player)
        assert np.all(np.sign(xs[:, 1]) == np.sign(np.median(xs[:, 1])))


def test_occlusion_is_bridged():
    t = np.arange(0, 4, 1 / FPS)
    a = np.c_[-10 + 4 * t, 3 + 0 * t]
    drop = np.zeros((len(t), 1), bool)
    drop[40:55] = True  # 0.6 s without detections
    tracks = _run_tracker([a], drop=drop)
    assert len(tracks) == 1


def test_rts_smoothing_reduces_noise():
    rng = np.random.default_rng(1)
    t = np.arange(0, 6, 1 / FPS)
    truth = np.c_[3 * t, np.sin(t)]
    z = truth + rng.normal(0, 0.3, truth.shape)
    fr, pos, vel = rts_smooth(np.arange(len(t)), z, np.repeat(np.eye(2)[None] * 0.09, len(t), 0), FPS)
    assert np.sqrt(np.mean((pos - truth) ** 2)) < 0.5 * np.sqrt(np.mean((z - truth) ** 2))
    assert abs(np.median(vel[:, 0]) - 3.0) < 0.2


def _tl(tid, f0, f1, p0, v, team=0):
    fr = np.arange(f0, f1)
    z = np.asarray(p0, float)[None] + np.outer((fr - f0) / FPS, v)
    return Tracklet(tid, fr, z, np.repeat(np.eye(2)[None] * 0.01, len(fr), 0), np.full(len(fr), 0.9),
                    np.zeros(len(fr), int), None, team, 0.9)


def test_stitch_links_continuation_and_respects_teams():
    a = _tl(1, 0, 50, [0, 0], [3, 0])  # ends at x = 6
    b = _tl(2, 75, 125, [9.2, 0], [3, 0])  # 1 s later, 3 m further: same player
    c = _tl(3, 75, 125, [9.2, 1.0], [3, 0], team=1)  # same place but other team
    chains = stitch([a, b, c], FPS)
    ids = {tuple(t.tid for t in ch) for ch in chains}
    assert (1, 2) in ids and (3,) in ids


def test_stitch_refuses_ambiguous_link():
    a = _tl(1, 0, 50, [0, 0], [0, 0])
    b = _tl(2, 200, 250, [0, 2.0], [0, 0])  # two equally plausible continuations 6 s later
    c = _tl(3, 200, 250, [0, -2.0], [0, 0])
    chains = stitch([a, b, c], FPS, StitchConfig())
    assert all(len(ch) == 1 for ch in chains)


def test_team_model_and_decide():
    rng = np.random.default_rng(0)
    D = 108

    def proto():
        h = np.zeros(D)
        h[rng.choice(D, 4, replace=False)] = 1
        return h / h.sum()

    pa, pb, pr = proto(), proto(), proto()

    def sample(p, n):
        x = p * 0.7 + 0.3 * rng.dirichlet(np.full(D, 0.3), n)
        x = np.sqrt(x / x.sum(1, keepdims=True))
        return x

    X = np.vstack([sample(pa, 400), sample(pb, 400), sample(pr, 40)])
    tm = fit_team_model(X)
    la, _ = decide(tm.distances(sample(pa, 20)))
    lb, _ = decide(tm.distances(sample(pb, 20)))
    lr, _ = decide(tm.distances(sample(pr, 20)))
    assert {la, lb} == {0, 1} and lr == OTHER


def test_gap_filling_kinds():
    a = _tl(1, 0, 50, [0, 0], [2, 0])
    b = _tl(2, 60, 110, [4.8, 0], [2, 0])
    ident = Identity(1, 0, "player", [a, b])
    _trajectory(ident, FPS)
    frame_t = np.arange(0, 200) / FPS
    out_t = np.arange(0, 7.9, 0.1)
    s = sample_identity(ident, frame_t, out_t, True, max_extrapolate_s=2.0)
    assert (s.kind[(out_t > 0.1) & (out_t < 1.9)] == DETECTED).all()
    assert (s.kind[(out_t > 2.05) & (out_t < 2.35)] == SHORT_GAP).all()
    assert (s.kind[(out_t > 4.5) & (out_t < 6.3)] == EDGE).all()
    assert np.isnan(s.pos[out_t > 6.5]).all()  # nothing beyond the extrapolation horizon
    gap = (out_t > 2.0) & (out_t < 2.4)
    assert np.allclose(s.pos[gap, 0], 2 * out_t[gap], atol=0.3)  # Hermite follows the motion


def test_other_team_kit_is_never_taken_over():
    """Two players of different teams cross at the same spot (worst case: positions alone
    cannot tell them apart); the kit labels must keep the tracks on their own players."""
    t = np.arange(0, 4, 1 / FPS)
    a = np.c_[-10 + 5 * t, 0.05 + 0 * t]
    b = np.c_[10 - 5 * t, -0.05 + 0 * t]
    rng = np.random.default_rng(3)
    tr = PitchTracker(FPS)
    R = np.eye(2) * 0.3 ** 2
    for k in range(len(t)):
        z = np.array([a[k], b[k]]) + rng.normal(0, 0.3, (2, 2))
        tr.step(k, z, np.repeat(R[None], 2, 0), np.full(2, 0.9), labels=np.array([0, 1]))
    for track in tr.close():
        assert len({o[3] for o in track.obs}) == 1  # detection 0 is always a, 1 always b


def test_split_by_team():
    from soccercal.players import split_by_team
    labs = np.array([1] * 30 + [-1, 0] + [1] * 10 + [0] * 25)  # a blip of 0 inside, then a real change
    parts = split_by_team(labs)
    assert len(parts) == 2 and parts[0][0] == 0 and parts[-1][1] == len(labs)
    assert 40 <= parts[0][1] <= 43
    assert split_by_team(np.array([-1] * 10)) == [(0, 10)]


def test_roster_fill_joins_fragments_to_ten_per_team():
    """12 fragments of 10 players (two leave the view and come back): 10 identities, and the
    returning fragments are joined to the right players."""
    from soccercal.config import Config
    from soccercal.players import fill_roster
    idents, pid = [], 1
    for p in range(10):
        y = -27.0 + 6 * p
        if p < 2:  # leaves the view at 4 s, back at 8 s, 2 m further along
            parts = [(0, 100, [-10.0, y]), (200, 300, [-10.0 + 2.0, y])]
        else:
            parts = [(0, 300, [-10.0, y])]
        for f0, f1, p0 in parts:
            tl = _tl(pid, f0, f1, p0, [0.0, 0.0], team=0)
            tl.player = p
            idents.append(Identity(pid, 0, "player", [tl]))
            pid += 1
    from soccercal.stitch import endpoints
    for i in idents:
        endpoints(i.tracklets[0], FPS)
    out = fill_roster(idents, Config(), FPS, None)
    assert len(out) == 10
    for i in out:
        assert len({t.player for t in i.tracklets}) == 1
