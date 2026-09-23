"""Pitch model, camera model and single-frame calibration (no network weights needed)."""
import numpy as np

from soccercal import pitch
from soccercal.calibrate import calibrate, quality_ok
from soccercal.camera import focal_from_homography, init_from_ground_points
from soccercal.field_model import FieldObservation
from soccercal.lines import LineMask, alignment

from conftest import look_at, render_lines


def test_pitch_conventions():
    assert pitch.KEYPOINTS.shape == (57, 3)
    assert np.allclose(pitch.KEYPOINTS[50], [0, 0, 0])  # 51: centre spot
    assert np.allclose(pitch.KEYPOINTS[44], [-41.5, 0, 0])  # 45: left penalty spot
    assert np.allclose(pitch.KEYPOINTS[0], [-52.5, 34, 0])  # 1: far-left corner (y up = far side)
    assert np.allclose(pitch.KEYPOINTS[27], [-52.5, -34, 0])  # 28: near-left corner
    assert np.allclose(pitch.KEYPOINTS[11], [-52.5, 3.66, 2.44])  # 12: top of far left post
    assert np.allclose(pitch.KEYPOINTS[51], [9.15, 0, 0])  # 52: fixed NBJW 61.5 typo
    # every keypoint defined as a line intersection lies on both lines
    for pid, la, lb in pitch.LINE_INTERSECTIONS:
        p = pitch.point_world(pid)
        for li in (la, lb):
            a, b = pitch.LINES[li]
            d = np.linalg.norm(np.cross(b - a, a - p)) / np.linalg.norm(b - a)
            assert d < 0.02, (pid, pitch.LINE_NAMES[li])


def test_projection_roundtrip(broadcast_cam):
    cam = broadcast_cam
    g = np.array([[x, y] for x in np.linspace(-52, 0, 9) for y in np.linspace(-30, 30, 7)])
    uv, z = cam.ground_to_image(g)
    back, ok = cam.image_to_ground(uv)
    assert ok.all() and np.allclose(back, g, atol=1e-6)
    # a point above the horizon does not hit the ground
    _, ok = cam.image_to_ground(np.array([[960.0, -5000.0]]))
    assert not ok[0]
    # expected height of a 1.8 m player is positive and shrinks with distance
    h_near, h_far = cam.pixel_height(np.array([[-20.0, -25.0], [-20.0, 25.0]]), 1.8)
    assert h_near > h_far > 0


def test_focal_from_homography(broadcast_cam):
    f = focal_from_homography(broadcast_cam.H, broadcast_cam.cx, broadcast_cam.cy)
    assert abs(f - broadcast_cam.f) / broadcast_cam.f < 1e-6


def test_init_from_ground_points(broadcast_cam):
    rng = np.random.default_rng(0)
    g = pitch.KEYPOINTS[pitch.KEYPOINTS[:, 2] == 0][:, :2]
    uv, z = broadcast_cam.ground_to_image(g)
    vis = (z > 0) & (uv[:, 0] > 0) & (uv[:, 0] < 1920) & (uv[:, 1] > 0) & (uv[:, 1] < 1080)
    cam, inl = init_from_ground_points(g[vis], uv[vis] + rng.normal(0, 1, uv[vis].shape), 1920, 1080)
    assert cam is not None
    assert abs(cam.f / broadcast_cam.f - 1) < 0.05
    assert np.linalg.norm(cam.C - broadcast_cam.C) < 3.0


def _observation(cam, noise=1.5, seed=0):
    rng = np.random.default_rng(seed)
    obs = FieldObservation(cam.width, cam.height)
    uv, z = cam.project(pitch.KEYPOINTS)
    for i, (p, zz) in enumerate(zip(uv, z)):
        if zz > 0 and 0 <= p[0] < cam.width and 0 <= p[1] < cam.height:
            obs.keypoints[i + 1] = (p[0] + rng.normal(0, noise), p[1] + rng.normal(0, noise), 0.8)
    return obs


def test_calibrate_rejects_outlier_and_recovers_camera(broadcast_cam):
    obs = _observation(broadcast_cam)
    obs.keypoints[51] = (60.0, 70.0, 0.9)  # gross outlier (centre spot is not even visible)
    cam = calibrate(obs)
    assert cam is not None and quality_ok(cam)
    assert 51 not in cam.info["point_ids"]
    g = np.array([[x, y] for x in np.linspace(-50, -10, 9) for y in np.linspace(-25, 25, 6)])
    uv, _ = broadcast_cam.ground_to_image(g)
    back, _ = cam.image_to_ground(uv)
    assert np.median(np.linalg.norm(back - g, axis=1)) < 0.3


def test_dense_line_refinement_improves_calibration(broadcast_cam):
    """Keypoints with 4 px noise -> the white-line chamfer term pulls the camera back."""
    img = render_lines(broadcast_cam)
    lm = LineMask.from_frame(img)
    obs = _observation(broadcast_cam, noise=4.0, seed=3)
    c_kp = calibrate(obs)
    c_dense = calibrate(obs, line_mask=lm)
    g = np.array([[x, y] for x in np.linspace(-50, -10, 9) for y in np.linspace(-25, 25, 6)])
    uv, _ = broadcast_cam.ground_to_image(g)
    err = lambda c: np.median(np.linalg.norm(c.image_to_ground(uv)[0] - g, axis=1))  # noqa: E731
    assert err(c_dense) < err(c_kp)
    assert err(c_dense) < 0.15
    assert alignment(c_dense, lm) > 0.9


def test_tripod_fit_uses_known_position(broadcast_cam):
    obs = _observation(broadcast_cam, noise=2.0, seed=1)
    cam = calibrate(obs, fix_center=broadcast_cam.C)
    assert np.allclose(cam.C, broadcast_cam.C)
    assert abs(cam.f / broadcast_cam.f - 1) < 0.02


def test_corners_projection_clips_far_corners():
    cam = look_at([0.0, -45.0, 12.0], [0.0, 30.0, 0.0], 1100.0)  # wide view reaching the horizon
    c = cam.corners_projection()
    assert set(c) == {f"{a}_{b}" for b in ("top_left", "bottom_left", "bottom_right", "top_right") for a in ("x", "y")}
    assert c["y_top_left"] <= 39.0 + 1e-6 and c["y_top_right"] <= 39.0 + 1e-6
    assert c["y_bottom_left"] < c["y_top_left"]


def test_line_mask_pack_roundtrip(broadcast_cam):
    lm = LineMask.from_frame(render_lines(broadcast_cam))
    lm2 = LineMask.unpack(lm.pack())
    assert (lm.mask == lm2.mask).all() and lm.mask.sum() > 1000
