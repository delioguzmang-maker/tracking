"""Single-frame camera calibration from pitch keypoints and line pixels.

Pipeline (per frame):

1. keypoints from the HRNet keypoint network (57 SoccerNet pitch points);
2. RANSAC homography on the ground-plane keypoints -> initial focal length and pose
   (closed form + planar PnP over a focal grid, best reprojection wins);
3. robust refinement on keypoints (incl. the goal-frame points 2.44 m up) plus, if a
   ``LineMask`` is given, the chamfer distance of all visible pitch markings to the
   white line pixels of the image (dense, thousands of constraints);
4. inlier re-selection and a second refinement; quality metrics for gating.

``fix_center`` turns it into the tripod fit (known camera position), which is what
makes video calibration stable (see ``camtrack.py``).
"""
from __future__ import annotations

import numpy as np

from . import pitch
from .camera import Camera, Correspondences, init_from_ground_points, refine, residual_norms
from .field_model import FieldObservation
from .lines import LineMask, alignment, visible_model_points


def _line_eq(a, b):
    return np.cross([a[0], a[1], 1.0], [b[0], b[1], 1.0])


def line_intersections(obs: FieldObservation) -> dict:
    """Pitch points from intersecting pairs of lines found by the (optional) line network."""
    out = {}
    w, h = obs.width, obs.height
    for pid, la, lb in pitch.LINE_INTERSECTIONS:
        if la not in obs.lines or lb not in obs.lines:
            continue
        A, B = obs.lines[la], obs.lines[lb]
        x = np.cross(_line_eq(A[0], A[1]), _line_eq(B[0], B[1]))
        if abs(x[2]) < 1e-9:
            continue
        u, v = x[0] / x[2], x[1] / x[2]
        d1 = np.array(A[1][:2]) - np.array(A[0][:2])
        d2 = np.array(B[1][:2]) - np.array(B[0][:2])
        sin = abs(d1[0] * d2[1] - d1[1] * d2[0]) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-9)
        if sin < 0.15:  # near-parallel lines -> unstable intersection
            continue
        if -0.5 * w < u < 1.5 * w and -0.5 * h < v < 1.5 * h:
            out[pid] = (float(u), float(v), 0.5 * min(A[0][2], A[1][2], B[0][2], B[1][2]) * min(1.0, sin / 0.5))
    return out


def build_correspondences(obs: FieldObservation, use_lines: bool = True) -> tuple[Correspondences, dict]:
    pts = dict(obs.keypoints)
    if use_lines and obs.lines:
        for pid, val in line_intersections(obs).items():
            pts.setdefault(pid, val)
    ids = np.array(sorted(pts), int)
    world = np.array([pitch.point_world(i) for i in ids]).reshape(-1, 3)
    img = np.array([pts[i][:2] for i in ids], float).reshape(-1, 2)
    wts = np.array([pts[i][2] for i in ids], float)
    if len(img):  # markings cut by the image border are localised worse
        m = 0.02 * obs.width
        edge = (img[:, 0] < m) | (img[:, 0] > obs.width - m) | (img[:, 1] < m) | (img[:, 1] > obs.height - m)
        wts = np.where(edge, 0.5 * wts, wts)
    lw, li, lwt, lid = [], [], [], []
    if use_lines:
        for k, (a, b) in obs.lines.items():
            for p in (a, b):
                lw.append(pitch.LINES[k])
                li.append(p[:2])
                lwt.append(p[2])
                lid.append(k)
    corr = Correspondences(world, img, wts, ids,
                           np.array(lw, float).reshape(-1, 2, 3), np.array(li, float).reshape(-1, 2),
                           np.array(lwt, float), np.array(lid, int))
    return corr, pts


def _subset(c: Correspondences, pm: np.ndarray, lm: np.ndarray) -> Correspondences:
    return Correspondences(c.pts_world[pm], c.pts_image[pm], c.pts_weight[pm], c.pts_id[pm],
                           c.line_world[lm], c.line_image[lm], c.line_weight[lm], c.line_id[lm])


def reseat(cam: Camera, C: np.ndarray) -> Camera:
    """Move a camera to position C, re-aiming it at the ground point it was looking at."""
    C = np.asarray(C, float)
    fwd_old = cam.R[2]
    s = -cam.C[2] / fwd_old[2] if fwd_old[2] < -1e-6 else 50.0
    look = cam.C + max(s, 1.0) * fwd_old
    fwd = look - C
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    f = cam.f * np.linalg.norm(look - C) / np.linalg.norm(look - cam.C)
    return cam.copy(C=C, R=np.stack([right, down, fwd]), f=f)


def calibrate(obs: FieldObservation, line_mask: LineMask | None = None, fix_center: np.ndarray | None = None,
              prior: Camera | None = None, use_lines: bool = False, min_points: int = 4) -> Camera | None:
    """Camera for one frame, or ``None`` when the evidence is insufficient/inconsistent.

    line_mask: white line pixels of the frame (enables the dense refinement).
    fix_center: known camera position (tripod constraint).
    prior: camera of a neighbouring frame, tried as an extra initialisation.
    use_lines: also use the line network's extremities (off: unreliable on real footage).
    """
    W, H = obs.width, obs.height
    corr, _ = build_correspondences(obs, use_lines=use_lines)
    ground = np.abs(corr.pts_world[:, 2]) < 1e-6
    cands = []
    cam0, inl = init_from_ground_points(corr.pts_world[ground, :2], corr.pts_image[ground], W, H)
    if cam0 is not None:
        cands.append(cam0)
    if prior is not None:
        cands.append(prior.copy(info={}))
    if not cands:
        return None
    pm0 = np.ones(len(corr.pts_world), bool)
    if cam0 is not None:  # RANSAC rejects start as outliers
        gidx = np.nonzero(ground)[0]
        pm0[gidx[~inl]] = False
    lm0 = np.ones(len(corr.line_world), bool)
    fixed = fix_center is not None
    thr = max(4.0, 0.006 * W)

    best, best_key = None, None
    for c0 in cands:
        cam = reseat(c0, fix_center) if fixed else c0
        pm, lm = pm0, lm0
        for it in range(2):
            lp = visible_model_points(cam) if line_mask is not None else None
            cam = refine(cam, _subset(corr, pm, lm), fix_center=fixed, line_mask=line_mask, line_points=lp)
            rp, rl = residual_norms(cam, corr)
            pm, lm = rp < (2.5 * thr if it == 0 else thr), rl < (2.5 * thr if it == 0 else thr)
        n_ev = int(pm.sum() + lm.sum() // 2)
        if n_ev < min_points and line_mask is None:
            continue
        res = np.concatenate([rp[pm], rl[lm]])
        rms = float(np.sqrt(np.mean(res ** 2))) if len(res) else np.inf
        al = alignment(cam, line_mask) if line_mask is not None else np.nan
        key = (0.0 if np.isnan(al) else al, n_ev, -rms)
        if best is None or key > best_key:
            best, best_key = cam, key
            best.info = _quality(cam, corr, pm, lm, rms, al)
    if best is None or not best.is_plausible():
        return None
    return best


def _quality(cam: Camera, corr: Correspondences, pin: np.ndarray, lin: np.ndarray, rms: float, align: float) -> dict:
    img = np.concatenate([corr.pts_image[pin], corr.line_image[lin]], 0)
    span = (np.ptp(img[:, 0]) / cam.width) * (np.ptp(img[:, 1]) / cam.height) if len(img) >= 3 else 0.0
    return {
        "rms_px": rms,
        "rms_1080": rms / cam.width * 1920.0,  # comparable across resolutions
        "n_points": int(pin.sum()),
        "n_line_pts": int(lin.sum()),
        "coverage": float(span),
        "align": align,
        "point_ids": corr.pts_id[pin].tolist(),
    }


def quality_ok(cam: Camera | None, max_rms_1080: float = 12.0, min_points: int = 4, min_align: float = 0.35) -> bool:
    """Accept a calibration for tracking. Uses the line alignment when available."""
    if cam is None:
        return False
    q = cam.info
    if q.get("rms_1080", np.inf) > max_rms_1080:
        return False
    al = q.get("align", np.nan)
    if not np.isnan(al):
        return bool(al >= min_align and q.get("n_points", 0) >= 3)
    return bool(q.get("n_points", 0) + q.get("n_line_pts", 0) // 2 >= min_points and q.get("coverage", 0) > 0.03)
