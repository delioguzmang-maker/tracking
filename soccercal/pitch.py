"""Pitch model in SkillCorner coordinates.

Convention (identical to SkillCorner broadcast tracking data):

* origin at the centre spot, metres;
* ``x`` along the long side, positive to the right as seen from the main camera;
* ``y`` along the short side, positive towards the FAR touchline (top of the image);
* ``z`` up.

The keypoint/line vocabulary is the one the public NBJW / PnLCalib networks were
trained on (57 keypoints + 23 lines, SoccerNet naming). Their tables use a
corner origin with ``y`` pointing to the near touchline and ``z`` pointing down;
``_from_soccernet`` converts that to the convention above.
"""
from __future__ import annotations

import numpy as np

LENGTH = 105.0
WIDTH = 68.0
GOAL_HEIGHT = 2.44
GOAL_HALF_WIDTH = 3.66
CIRCLE_R = 9.15


def _from_soccernet(x: float, y: float, z_down: float = 0.0) -> tuple[float, float, float]:
    """SoccerNet/NBJW table (corner origin, y to near side, z down) -> SkillCorner."""
    return (x - LENGTH / 2, -(y - WIDTH / 2), -z_down)


# 57 keypoints predicted by the keypoint network, 1-based ids as in NBJW.
# (x, y, z_down) in SoccerNet table convention. Keypoint 52 is 61.65 in the
# real pitch (52.5 + 9.15); NBJW's table has a 61.5 typo, fixed here.
_KP_TABLE = [
    (0., 0., 0.), (52.5, 0., 0.), (105., 0., 0.), (0., 13.84, 0.), (16.5, 13.84, 0.), (88.5, 13.84, 0.),
    (105., 13.84, 0.), (0., 24.84, 0.), (5.5, 24.84, 0.), (99.5, 24.84, 0.), (105., 24.84, 0.),
    (0., 30.34, -2.44), (0., 30.34, 0.), (105., 30.34, 0.), (105., 30.34, -2.44),
    (0., 37.66, -2.44), (0., 37.66, 0.), (105., 37.66, 0.), (105., 37.66, -2.44),
    (0., 43.16, 0.), (5.5, 43.16, 0.), (99.5, 43.16, 0.), (105., 43.16, 0.), (0., 54.16, 0.), (16.5, 54.16, 0.),
    (88.5, 54.16, 0.), (105., 54.16, 0.), (0., 68., 0.), (52.5, 68., 0.), (105., 68., 0.),
    (16.5, 26.68, 0.), (52.5, 24.85, 0.), (88.5, 26.68, 0.), (16.5, 41.31, 0.), (52.5, 43.15, 0.),
    (88.5, 41.31, 0.), (19.99, 32.29, 0.), (43.68, 31.53, 0.), (61.31, 31.53, 0.), (85., 32.29, 0.),
    (19.99, 35.7, 0.), (43.68, 36.46, 0.), (61.31, 36.46, 0.), (85., 35.7, 0.), (11., 34., 0.),
    (16.5, 34., 0.), (20.15, 34., 0.), (46.03, 27.53, 0.), (58.97, 27.53, 0.), (43.35, 34., 0.),
    (52.5, 34., 0.), (61.65, 34., 0.), (46.03, 40.47, 0.), (58.97, 40.47, 0.), (84.85, 34., 0.),
    (88.5, 34., 0.), (94., 34., 0.),
]
KEYPOINTS = np.array([_from_soccernet(*p) for p in _KP_TABLE])  # (57, 3), index = id - 1

# Auxiliary keypoints (ids 58..73): intersections of non-adjacent lines.
_AUX_TABLE = [(5.5, 0), (16.5, 0), (88.5, 0), (99.5, 0), (5.5, 13.84), (99.5, 13.84), (16.5, 24.84),
              (88.5, 24.84), (16.5, 43.16), (88.5, 43.16), (5.5, 54.16), (99.5, 54.16), (5.5, 68),
              (16.5, 68), (88.5, 68), (99.5, 68)]
AUX_KEYPOINTS = np.array([_from_soccernet(x, y) for x, y in _AUX_TABLE])

# 23 lines predicted by the line network (1-based ids as in NBJW).
LINE_NAMES = [
    "Big rect. left bottom", "Big rect. left main", "Big rect. left top", "Big rect. right bottom",
    "Big rect. right main", "Big rect. right top", "Goal left crossbar", "Goal left post left",
    "Goal left post right", "Goal right crossbar", "Goal right post left", "Goal right post right",
    "Middle line", "Side line bottom", "Side line left", "Side line right", "Side line top",
    "Small rect. left bottom", "Small rect. left main", "Small rect. left top", "Small rect. right bottom",
    "Small rect. right main", "Small rect. right top",
]
_LINE_TABLE = [
    ((0., 54.16, 0.), (16.5, 54.16, 0.)), ((16.5, 13.84, 0.), (16.5, 54.16, 0.)),
    ((16.5, 13.84, 0.), (0., 13.84, 0.)), ((88.5, 54.16, 0.), (105., 54.16, 0.)),
    ((88.5, 13.84, 0.), (88.5, 54.16, 0.)), ((88.5, 13.84, 0.), (105., 13.84, 0.)),
    ((0., 37.66, -2.44), (0., 30.34, -2.44)), ((0., 37.66, 0.), (0., 37.66, -2.44)),
    ((0., 30.34, 0.), (0., 30.34, -2.44)), ((105., 37.66, -2.44), (105., 30.34, -2.44)),
    ((105., 30.34, 0.), (105., 30.34, -2.44)), ((105., 37.66, 0.), (105., 37.66, -2.44)),
    ((52.5, 0., 0.), (52.5, 68., 0.)), ((0., 68., 0.), (105., 68., 0.)), ((0., 0., 0.), (0., 68., 0.)),
    ((105., 0., 0.), (105., 68., 0.)), ((0., 0., 0.), (105., 0., 0.)), ((0., 43.16, 0.), (5.5, 43.16, 0.)),
    ((5.5, 43.16, 0.), (5.5, 24.84, 0.)), ((5.5, 24.84, 0.), (0., 24.84, 0.)),
    ((99.5, 43.16, 0.), (105., 43.16, 0.)), ((99.5, 43.16, 0.), (99.5, 24.84, 0.)),
    ((99.5, 24.84, 0.), (105., 24.84, 0.)),
]
LINES = np.array([[_from_soccernet(*a), _from_soccernet(*b)] for a, b in _LINE_TABLE])  # (23, 2, 3)

# Keypoints 1..30 are intersections of two lines (used when the keypoint net misses them).
_KP_FROM_LINES = [
    ("Side line top", "Side line left"), ("Side line top", "Middle line"), ("Side line right", "Side line top"),
    ("Side line left", "Big rect. left top"), ("Big rect. left top", "Big rect. left main"),
    ("Big rect. right top", "Big rect. right main"), ("Side line right", "Big rect. right top"),
    ("Side line left", "Small rect. left top"), ("Small rect. left top", "Small rect. left main"),
    ("Small rect. right top", "Small rect. right main"), ("Side line right", "Small rect. right top"),
    ("Goal left crossbar", "Goal left post right"), ("Side line left", "Goal left post right"),
    ("Side line right", "Goal right post left"), ("Goal right crossbar", "Goal right post left"),
    ("Goal left crossbar", "Goal left post left"), ("Side line left", "Goal left post left"),
    ("Side line right", "Goal right post right"), ("Goal right crossbar", "Goal right post right"),
    ("Side line left", "Small rect. left bottom"), ("Small rect. left bottom", "Small rect. left main"),
    ("Small rect. right bottom", "Small rect. right main"), ("Side line right", "Small rect. right bottom"),
    ("Side line left", "Big rect. left bottom"), ("Big rect. left bottom", "Big rect. left main"),
    ("Big rect. right main", "Big rect. right bottom"), ("Side line right", "Big rect. right bottom"),
    ("Side line left", "Side line bottom"), ("Side line bottom", "Middle line"),
    ("Side line bottom", "Side line right"),
]
_AUX_FROM_LINES = [
    ("Small rect. left main", "Side line top"), ("Big rect. left main", "Side line top"),
    ("Big rect. right main", "Side line top"), ("Small rect. right main", "Side line top"),
    ("Small rect. left main", "Big rect. left top"), ("Big rect. right top", "Small rect. right main"),
    ("Small rect. left top", "Big rect. left main"), ("Small rect. right top", "Big rect. right main"),
    ("Small rect. left bottom", "Big rect. left main"), ("Small rect. right bottom", "Big rect. right main"),
    ("Small rect. left main", "Big rect. left bottom"), ("Small rect. right main", "Big rect. right bottom"),
    ("Small rect. left main", "Side line bottom"), ("Big rect. left main", "Side line bottom"),
    ("Big rect. right main", "Side line bottom"), ("Small rect. right main", "Side line bottom"),
]
_LID = {n: i for i, n in enumerate(LINE_NAMES)}
# (point id 1-based, line index a, line index b); ids 58..73 are the aux keypoints.
LINE_INTERSECTIONS = [(k + 1, _LID[a], _LID[b]) for k, (a, b) in enumerate(_KP_FROM_LINES)] + \
                     [(58 + k, _LID[a], _LID[b]) for k, (a, b) in enumerate(_AUX_FROM_LINES)]


def point_world(pid: int) -> np.ndarray:
    """World coordinates of keypoint id (1..57) or aux keypoint id (58..73)."""
    return KEYPOINTS[pid - 1] if pid <= 57 else AUX_KEYPOINTS[pid - 58]


def polylines(length: float = LENGTH, width: float = WIDTH, n_arc: int = 48) -> list[np.ndarray]:
    """All pitch markings (ground plane plus goal frames) as 3D polylines, for drawing."""
    L, W = length / 2, width / 2
    out = []

    def seg(*pts):
        out.append(np.array([(p[0], p[1], p[2] if len(p) > 2 else 0.0) for p in pts], float))

    seg((-L, -W), (L, -W), (L, W), (-L, W), (-L, -W))
    seg((0, -W), (0, W))
    for s in (-1, 1):
        bx = s * (L - 16.5)
        seg((s * L, -20.16), (bx, -20.16), (bx, 20.16), (s * L, 20.16))
        sx = s * (L - 5.5)
        seg((s * L, -9.16), (sx, -9.16), (sx, 9.16), (s * L, 9.16))
        g = GOAL_HALF_WIDTH
        seg((s * L, -g, 0), (s * L, -g, GOAL_HEIGHT), (s * L, g, GOAL_HEIGHT), (s * L, g, 0))
        # penalty arc: part of the circle around the spot that lies outside the box
        spot = s * (L - 11.0)
        a0 = np.arccos(5.5 / CIRCLE_R)
        ang = np.linspace(-a0, a0, n_arc // 2)
        seg(*[(spot - s * CIRCLE_R * np.cos(a), CIRCLE_R * np.sin(a)) for a in ang])
    ang = np.linspace(0, 2 * np.pi, n_arc + 1)
    seg(*[(CIRCLE_R * np.cos(a), CIRCLE_R * np.sin(a)) for a in ang])
    return out


def inside_pitch(xy: np.ndarray, margin: float = 0.0, length: float = LENGTH, width: float = WIDTH) -> np.ndarray:
    xy = np.asarray(xy, float).reshape(-1, 2)
    return (np.abs(xy[:, 0]) <= length / 2 + margin) & (np.abs(xy[:, 1]) <= width / 2 + margin)
