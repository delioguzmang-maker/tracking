"""Shot types, off-pitch staff, shirt numbers and the ball trajectory."""
import numpy as np

from soccercal.analysis import Analysis, FrameData
from soccercal.ball import track_ball
from soccercal.cameras import shot_type
from soccercal.camtrack import Registration
from soccercal.config import Config
from soccercal.detect import BALL, PERSON, Detections
from soccercal.jersey import vote_number
from soccercal.players import (Identity, kit_display_colors, mark_assistant_referees, merge_by_number,
                               merge_unique_roles, off_pitch, outside_share)
from soccercal.stitch import Tracklet

from conftest import look_at


def _fd(boxes, scores, classes, t=0.0, grass=0.6):
    det = Detections(np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(scores, np.float32),
                     np.asarray(classes, int))
    n = int((det.classes == PERSON).sum())
    return FrameData(0, t, 0.0, grass, Registration(None, None), det, np.zeros((n, 108), np.uint8),
                     np.zeros((n, 3), np.uint8))


def test_shot_type():
    cfg = Config()
    wide = _fd([[100, 400, 130, 480]], [0.9], [PERSON])
    closeup = _fd([[500, 100, 900, 1080]], [0.9], [PERSON])  # a person fills the frame
    crowd = _fd([[100, 400, 130, 480]], [0.9], [PERSON], grass=0.05)
    assert shot_type(wide, 1920, 1080, cfg) == "wide"
    assert shot_type(closeup, 1920, 1080, cfg) == "closeup"
    assert shot_type(crowd, 1920, 1080, cfg) == "no_pitch"


def test_outside_share_flags_touchline_staff():
    cfg = Config()
    linesman = np.c_[np.linspace(-30, 0, 50), np.full(50, -35.5)]  # runs 1.5 m outside the touchline
    winger = np.c_[np.linspace(-30, 0, 50), np.full(50, -33.5)]  # on the pitch, near the line
    assert outside_share(linesman, cfg) > 0.9
    assert outside_share(winger, cfg) == 0.0


def test_assistant_referee_vs_referee():
    def ident(pid, y):
        tl = Tracklet(pid, np.arange(20), np.c_[np.linspace(-20, 0, 20), np.full(20, y)], np.zeros((20, 2, 2)),
                      np.ones(20), np.zeros(20, int))
        return Identity(pid, -1, "referee", [tl])
    ref, lines = ident(1, 3.0), ident(2, -34.3)
    mark_assistant_referees([ref, lines], Config())
    assert ref.role == "referee" and lines.role == "assistant_referee"


def test_coach_on_touchline_is_not_assistant_referee():
    """Colour group 2 = referee kit, 3 = coach's black jacket. A coach standing at the edge
    of the technical area is staff even if calibration noise puts him on the line."""
    def ident(pid, y, group, role="referee", x0=-20.0):
        tl = Tracklet(pid, np.arange(20), np.c_[np.linspace(x0, x0 + 20, 20), np.full(20, y)], np.zeros((20, 2, 2)),
                      np.ones(20), np.zeros(20, int))
        return Identity(pid, -1, role, [tl], kit_group=group)
    ref, lines, coach = ident(1, 3.0, 2), ident(2, -34.3, 2), ident(3, -34.4, 3)
    runner = ident(4, -35.6, 2, role="staff")  # assistant referee running just outside the line
    mark_assistant_referees([ref, lines, coach, runner], Config())
    assert ref.role == "referee" and lines.role == "assistant_referee"
    assert coach.role == "staff" and runner.role == "assistant_referee"
    rng = np.random.default_rng(1)
    at_edge = np.c_[rng.uniform(-5, 5, 60), -34.7 + rng.normal(0, 0.3, 60)]  # 0.7 m out, noisy
    assert outside_share(at_edge, Config()) < 0.6 and off_pitch(at_edge, Config())
    assert not off_pitch(np.c_[np.linspace(-30, 0, 50), np.full(50, -33.5)], Config())


def test_duplicate_referee_box_is_absorbed():
    def tl(tid, fr, y):
        fr = np.asarray(fr)
        return Tracklet(tid, fr, np.c_[fr * 0.1, np.full(len(fr), y)], np.zeros((len(fr), 2, 2)), np.ones(len(fr)),
                        np.zeros(len(fr), int))
    main = Identity(1, -1, "referee", [tl(1, range(0, 100), 0.0)])
    dup = Identity(2, -1, "referee", [tl(2, range(40, 55), 0.8)])  # partial box of the same man
    other = Identity(3, -1, "referee", [tl(3, range(40, 55), 12.0)])  # someone else (a misread player)
    out = merge_unique_roles([main, dup, other])
    assert {i.pid for i in out} == {1, 3} and len(out[0].absorbed) == 1 and len(out[0].tracklets) == 1


def test_vote_number_needs_agreement():
    assert vote_number([(17, 0.9), (17, 0.8), (12, 0.7)]) == (17, 2)
    assert vote_number([(17, 0.9)])[0] is None  # a single reading is not enough
    assert vote_number([(7, 0.9), (7, 0.9)])[0] is None  # one digit needs 3 votes (could be half of 17)
    assert vote_number([(7, 0.9)] * 3)[0] == 7


def _tl(tid, f0, f1):
    fr = np.arange(f0, f1)
    return Tracklet(tid, fr, np.zeros((len(fr), 2)), np.zeros((len(fr), 2, 2)), np.ones(len(fr)), np.zeros(len(fr), int))


def test_merge_by_number():
    a = Identity(1, 0, "player", [_tl(1, 0, 50)], number=17, number_votes=3)
    b = Identity(2, 0, "player", [_tl(2, 300, 350)], number=17, number_votes=2)  # later: same player
    c = Identity(3, 0, "player", [_tl(3, 10, 40)], number=17, number_votes=1)  # same time as a: misread
    d = Identity(4, 1, "player", [_tl(4, 300, 350)], number=17, number_votes=2)  # other team keeps its 17
    out = merge_by_number([a, b, c, d])
    ids = {i.pid: i for i in out}
    assert set(ids) == {1, 3, 4}
    assert len(ids[1].tracklets) == 2 and ids[3].number is None and ids[4].number == 17


def test_referee_fragments_merge():
    a = Identity(1, -1, "referee", [_tl(1, 0, 50)])
    b = Identity(2, -1, "referee", [_tl(2, 60, 90)])
    c = Identity(3, -1, "referee", [_tl(3, 40, 70)])  # overlaps both, 10 m away: a different person
    c.tracklets[0].z[:] = 10.0
    out = merge_unique_roles([a, b, c])
    assert {i.pid for i in out} == {1, 3} and len(out[0].tracklets) == 2


def test_kit_colours_are_distinct():
    red, dark = kit_display_colors([np.array([60, 60, 170]), np.array([40, 35, 30])])
    assert red[2] > 180 and max(dark) < 80  # saturated red vs near-black
    red_at_night, _ = kit_display_colors([np.array([53, 95, 143]), np.array([46, 73, 78])])  # measured: hue 14
    assert red_at_night[2] > 200 and red_at_night[1] < 40  # drawn red, not orange
    same = kit_display_colors([np.array([60, 60, 170]), np.array([65, 55, 175])])
    assert np.linalg.norm(np.subtract(same[0], same[1])) > 90  # falls back to distinguishable colours


def test_ball_trajectory_ignores_false_positives():
    """True ball: weak detections (0.05-0.1) moving 10 m/s; false positives: stronger but
    isolated and jumping around. The trajectory must follow the true ball."""
    rng = np.random.default_rng(0)
    cam = look_at([0.0, -50.0, 15.0], [-10.0, 0.0, 0.0], 2400.0)
    fps, n = 25.0, 100
    frames, truth = [], []
    for k in range(n):
        p = np.array([-25.0 + 10.0 * k / fps, -5.0 + 2.0 * k / fps])
        truth.append(p)
        boxes, scores = [], []
        if rng.random() < 0.7:  # the ball is missed 30 % of the time
            (u, v), = cam.project(np.array([[*p, 0.11]]))[0]
            s = cam.pixel_height(p[None], 0.22)[0]
            boxes.append([u - s / 2, v - s / 2, u + s / 2, v + s / 2])
            scores.append(rng.uniform(0.05, 0.1))
        if rng.random() < 0.4:  # a random false positive somewhere on the pitch
            q = rng.uniform([-40, -25], [20, 25])
            (u, v), = cam.project(np.array([[*q, 0.11]]))[0]
            s = cam.pixel_height(q[None], 0.22)[0]
            boxes.append([u - s / 2, v - s / 2, u + s / 2, v + s / 2])
            scores.append(rng.uniform(0.1, 0.3))
        frames.append(_fd(boxes, scores, [BALL] * len(scores), t=k / fps))
    an = Analysis("synthetic", fps, cam.width, cam.height, n, {"stride_used": 1}, frames)
    pos, det = track_ball(an, [cam] * n)
    truth = np.array(truth)
    ok = np.isfinite(pos).all(1)
    err = np.linalg.norm(pos[ok] - truth[ok], axis=1)
    assert ok.mean() > 0.8  # detected or interpolated most of the time
    assert np.median(err) < 0.3 and (err > 3).mean() < 0.05  # and it is the real ball


def test_real_fps_from_timestamps(tmp_path):
    import cv2

    from soccercal.analysis import auto_stride, iter_frames, probe_fps

    path = str(tmp_path / "clip.avi")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 50.0, (64, 48))
    for k in range(30):
        vw.write(np.full((48, 64, 3), k * 8, np.uint8))
    vw.release()
    fps = probe_fps(path)
    assert abs(fps - 50.0) < 0.5 and auto_stride(fps) == 2
    t = [s for _, s, _ in iter_frames(path, 0.0, 2, None, with_time=True, fps=fps)]
    assert len(t) == 15 and np.allclose(np.diff(t), 0.04, atol=1e-3)


def test_jersey_reader_reads_a_small_back_number():
    """A 70 px tall player seen from behind at 720p: red shirt, white '17' ~14 px tall."""
    from soccercal.jersey import JerseyReader
    reader = JerseyReader()
    if not reader.available:
        import pytest
        pytest.skip("rapidocr_onnxruntime not installed")
    frame = np.full((200, 200, 3), (60, 140, 60), np.uint8)  # grass
    box = np.array([80, 50, 110, 120], float)  # 30 x 70 px player
    cv2 = __import__("cv2")
    cv2.rectangle(frame, (80, 58), (110, 92), (40, 40, 200), -1)  # shirt
    cv2.rectangle(frame, (84, 92), (106, 120), (40, 40, 200), -1)  # shorts / legs
    cv2.putText(frame, "17", (83, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 2, cv2.LINE_AA)
    frame = cv2.GaussianBlur(frame, (3, 3), 0.8)
    r = reader.read_one(frame, box)
    assert r is not None and r[0] == 17


def test_ball_search_windows_follow_the_trajectory():
    from soccercal.refine import ball_search_windows
    cam = look_at([0.0, -50.0, 15.0], [0.0, 0.0, 0.0], 1500.0, W=1280, H=720)
    n = 30
    frames = [_fd([], [], [], t=k / 25.0) for k in range(n)]
    an = Analysis("synthetic", 25.0, 1280, 720, n, {"stride_used": 1}, frames)
    pos = np.full((n, 2), np.nan)
    det = np.zeros(n, bool)
    for k in (0, 1, 2, 20, 21):  # seen, lost for 17 frames, seen again
        pos[k], det[k] = [-5.0 + 0.5 * k, 0.0], True
    win = ball_search_windows(an, [cam] * n, pos, det, imgsz=1280)
    ks = sorted(win)
    assert set(ks) <= set(range(n)) - {0, 1, 2, 20, 21} and len(ks) >= 7
    assert min(np.diff(ks)) >= 3  # about every 0.1 s (the tracker interpolates in between)
    for k in ks:
        uv, _ = cam.project(np.array([[-5.0 + 0.5 * k, 0.0, 0.11]]))
        (x0, y0, x1, y1), = win[k]
        assert x0 <= uv[0, 0] <= x1 and y0 <= uv[0, 1] <= y1 and x1 - x0 == int(1280 / 2.4)
