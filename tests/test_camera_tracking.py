"""Frame-to-frame camera motion (tripod model) and fusion with absolute calibrations."""
import numpy as np
from scipy.spatial.transform import Rotation

from soccercal.camtrack import Registration, fuse_segment, propagate, propagate_back, robust_center

from conftest import look_at


def _matches(c1, c2, n=200, seed=0):
    rng = np.random.default_rng(seed)
    # background points far away (crowd) and on the grass: any 3D point works for a pure rotation
    P = np.c_[rng.uniform(-60, 60, n), rng.uniform(-10, 60, n), rng.uniform(0, 15, n)]
    a, za = c1.project(P)
    b, zb = c2.project(P)
    ok = (za > 0) & (zb > 0) & (np.abs(a).max(1) < 3000) & (np.abs(b).max(1) < 3000)
    return Registration(a[ok].astype(np.float32), b[ok].astype(np.float32), 0.1)


def test_propagate_recovers_pan_and_zoom():
    c1 = look_at([0, -50, 12], [-30, 0, 0], 3000)
    c2 = look_at([0, -50, 12], [-27, 2, 0], 3150)
    p = propagate(c1, _matches(c1, c2))
    assert abs(p.f - c2.f) < 1.0
    assert np.degrees(Rotation.from_matrix(p.R @ c2.R.T).magnitude()) < 1e-3
    back = propagate_back(c2, _matches(c1, c2))
    assert abs(back.f - c1.f) < 1.0


def test_fusion_removes_keyframe_noise():
    rng = np.random.default_rng(0)
    n = 100
    truth = [look_at([0, -50, 12], [-30 + 0.1 * k, 0, 0], 3000 + 2 * k) for k in range(n)]
    # chain: exact relative motion but drifting (start 0.3 deg off)
    off = Rotation.from_rotvec(np.radians([0.3, 0.0, 0.0])).as_matrix()
    chain = [truth[0].copy(R=off @ truth[0].R)]
    for k in range(1, n):
        chain.append(propagate(chain[-1], _matches(truth[k - 1], truth[k], seed=k)))
    # noisy absolute calibrations every 5 frames
    keys = {}
    for k in range(0, n, 5):
        e = Rotation.from_rotvec(np.radians(rng.normal(0, 0.05, 3))).as_matrix()
        keys[k] = truth[k].copy(R=e @ truth[k].R, f=truth[k].f * np.exp(rng.normal(0, 0.01)))
    fused = fuse_segment(chain, keys, {k: 1.0 for k in keys}, smooth_lambda=5.0)
    err = lambda cs: np.mean([np.degrees(Rotation.from_matrix(c.R @ t.R.T).magnitude()) for c, t in zip(cs, truth)])  # noqa: E731
    err_keys = np.mean([np.degrees(Rotation.from_matrix(keys[k].R @ truth[k].R.T).magnitude()) for k in keys])
    assert err(fused) < 0.5 * err(chain)
    assert err(fused) < err_keys


def test_robust_center_ignores_outliers():
    good = [look_at([1 + 0.1 * i, -50, 12], [-30, 0, 0], 3000) for i in range(10)]
    bad = [look_at([30, -90, 40], [-30, 0, 0], 3000)]
    C = robust_center(good + bad)
    assert np.linalg.norm(C - [1.45, -50, 12]) < 0.5
