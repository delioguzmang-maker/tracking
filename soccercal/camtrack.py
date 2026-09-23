"""Camera tracking over a video: shot cuts, frame-to-frame registration, fusion.

Why: calibrating every frame independently from keypoints jitters by a few pixels,
which becomes 0.3-1 m of noise on the far side of the pitch and ruins speeds.
Broadcast cameras sit on a tripod: between frames they only pan / tilt / roll /
zoom, so consecutive images are related *exactly* by the infinite homography
``K2 R K1^-1``, which sparse optical flow on the background measures to a fraction
of a pixel. That relative motion is precise but drifts; the keypoint calibrations
are noisy but unbiased. ``fuse_segment`` combines them (a small linear smoother on
the drift correction), after fixing the camera position per shot.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.sparse import diags
from scipy.optimize import least_squares
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation

from .camera import Camera

REG_WIDTH = 640  # registration runs on a downscaled grey image


# ------------------------------------------------------------------ shot cuts
def frame_signature(frame_bgr: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    return (h / (h.sum() + 1e-9)).astype(np.float32)


def signature_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Hellinger distance between two frame signatures (0 = identical, 1 = disjoint)."""
    return float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(a * b)))))


def grass_fraction(frame_bgr: np.ndarray) -> float:
    small = cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    g = (hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 50) & (hsv[..., 2] > 40)
    return float(g.mean())


# ------------------------------------------------------- frame registration
@dataclass
class Registration:
    """Background point matches between consecutive frames (full-resolution pixels)."""
    a: np.ndarray | None  # (N,2) points in the previous frame
    b: np.ndarray | None  # (N,2) same points in the current frame
    residual_px: float = np.inf  # RANSAC homography residual (quality)

    @property
    def ok(self) -> bool:
        return self.a is not None and len(self.a) >= 12


class Registrar:
    """Tracks background features (crowd, ads, grass texture) between frames."""

    def __init__(self, width: int, height: int, max_corners: int = 600, keep: int = 160):
        self.scale = REG_WIDTH / float(width)
        self.size = (REG_WIDTH, int(round(height * self.scale)))
        self.max_corners = max_corners
        self.keep = keep
        self.prev = None
        self.prev_mask = None

    def gray(self, frame_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(cv2.resize(frame_bgr, self.size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)

    def mask(self, boxes: np.ndarray | None) -> np.ndarray:
        """Features on moving players / the ball would bias the camera motion: mask them out."""
        m = np.full(self.size[::-1], 255, np.uint8)
        if boxes is not None:
            for x1, y1, x2, y2 in np.asarray(boxes).reshape(-1, 4) * self.scale:
                pad = 0.15 * (y2 - y1) + 2
                cv2.rectangle(m, (int(x1 - pad), int(y1 - pad)), (int(x2 + pad), int(y2 + pad)), 0, -1)
        return m

    def reset(self):
        self.prev = None

    def step(self, frame_bgr: np.ndarray, boxes: np.ndarray | None = None) -> Registration:
        g = self.gray(frame_bgr)
        mk = self.mask(boxes)
        prev, prev_mask = self.prev, self.prev_mask
        self.prev, self.prev_mask = g, mk
        if prev is None:
            return Registration(None, None)
        p0 = cv2.goodFeaturesToTrack(prev, self.max_corners, 0.01, 8, mask=prev_mask, blockSize=7)
        if p0 is None or len(p0) < 12:
            return Registration(None, None)
        lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None, **lk)
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(g, prev, p1, None, **lk)
        fb = np.linalg.norm((p0 - pb).reshape(-1, 2), axis=1)
        good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 0.5)
        a, b = p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]
        if len(a) < 12:
            return Registration(None, None)
        # Static overlays (score bug, channel logo) do not move with the camera.
        flow = np.linalg.norm(b - a, axis=1)
        if np.median(flow) > 1.0:
            keep = flow > 0.2
            if keep.sum() >= 12:
                a, b = a[keep], b[keep]
        H, inl = cv2.findHomography(a, b, cv2.RANSAC, 1.0, maxIters=1000, confidence=0.995)
        if H is None or inl.sum() < 12:
            return Registration(None, None)
        inl = inl.ravel().astype(bool)
        a, b = a[inl], b[inl]
        pa = cv2.perspectiveTransform(a[None].astype(np.float64), H)[0]
        res = float(np.sqrt(np.mean(np.sum((pa - b) ** 2, 1)))) / self.scale
        if len(a) > self.keep:  # spatially spread subsample, enough for a 4-parameter fit
            idx = np.linspace(0, len(a) - 1, self.keep).astype(int)
            a, b = a[idx], b[idx]
        return Registration((a / self.scale).astype(np.float32), (b / self.scale).astype(np.float32), res)


def propagate(cam: Camera, reg: Registration) -> Camera:
    """Camera at the next frame from background matches (tripod: rotation + zoom only).

    Fits 4 parameters (3 rotation + log zoom) instead of a free 8-parameter homography:
    the extra homography freedoms only absorb noise, which showed up as tilt jitter.
    """
    c = np.array([cam.cx, cam.cy])
    f1 = cam.f
    a, b = reg.a.astype(np.float64), reg.b.astype(np.float64)
    r1 = np.hstack([(a - c) / f1, np.ones((len(a), 1))])

    def fun(p):
        R = Rotation.from_rotvec(p[:3]).as_matrix()
        r2 = r1 @ R.T
        q = f1 * np.exp(p[3]) * r2[:, :2] / r2[:, 2:3] + c
        return (q - b).ravel()

    sol = least_squares(fun, np.zeros(4), loss="soft_l1", f_scale=0.7, max_nfev=50)
    Rr = Rotation.from_rotvec(sol.x[:3]).as_matrix()
    return cam.copy(R=Rr @ cam.R, f=float(f1 * np.exp(sol.x[3])), info={"propagated": True})


def propagate_back(cam: Camera, reg: Registration) -> Camera:
    """Camera at the previous frame (inverse of ``propagate``)."""
    return propagate(cam, Registration(reg.b, reg.a, reg.residual_px))


# ------------------------------------------------------------------- fusion
def _smooth_1d(n: int, obs_idx: np.ndarray, obs_val: np.ndarray, obs_w: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """argmin_x sum_k w_k (x[i_k] - v_k)^2 + sum_t lam_t (x[t] - x[t-1])^2  (tridiagonal solve)."""
    d = np.zeros(n)
    rhs = np.zeros(n)
    np.add.at(d, obs_idx, obs_w)
    np.add.at(rhs, obs_idx, obs_w * obs_val)
    lam = np.asarray(lam, float)
    main = d.copy()
    main[1:] += lam
    main[:-1] += lam
    main += 1e-9
    if n == 1:
        return rhs / main
    A = diags([main, -lam, -lam], [0, -1, 1], format="csc")
    return spsolve(A, rhs)


def fuse_segment(chain: list[Camera], keys: dict[int, Camera], weights: dict[int, float],
                 smooth_lambda: float = 50.0, iters: int = 3) -> list[Camera]:
    """Fuse a registration chain with absolute calibrations at some frames.

    chain[t]: camera obtained by propagating frame 0 through the registrations
    (smooth, precise frame to frame, but drifting). keys[t]: independent absolute
    calibrations (same camera centre). Solves for a slowly varying correction
    (rotation vector + log focal) that pulls the chain onto the absolute
    measurements, with Huber re-weighting against bad calibrations.
    """
    n = len(chain)
    if not keys:
        return chain
    idx = np.array(sorted(keys))
    base_w = np.array([weights.get(i, 1.0) for i in idx])
    # residual of each absolute calibration w.r.t. the chain, as a 4-vector
    meas = []
    for i in idx:
        dR = Rotation.from_matrix(keys[i].R @ chain[i].R.T).as_rotvec()
        meas.append(np.r_[dR, np.log(keys[i].f / chain[i].f)])
    meas = np.array(meas)
    lam = np.full(n - 1, smooth_lambda)
    w = base_w.copy()
    corr = np.zeros((n, 4))
    for _ in range(iters):
        for c in range(4):
            corr[:, c] = _smooth_1d(n, idx, meas[:, c], w, lam)
        # Huber on the angular residual (degrees) of each calibration
        r = np.degrees(np.linalg.norm(meas[:, :3] - corr[idx, :3], axis=1)) + 10 * np.abs(meas[:, 3] - corr[idx, 3])
        k = 0.15
        w = base_w * np.where(r <= k, 1.0, k / np.maximum(r, 1e-9))
    out = []
    for t in range(n):
        R = Rotation.from_rotvec(corr[t, :3]).as_matrix() @ chain[t].R
        c = chain[t].copy(R=R, f=float(chain[t].f * np.exp(corr[t, 3])))
        c.info = {"fused": True}
        out.append(c)
    return out


def robust_center(cams: list[Camera], weights: list[float] | None = None) -> np.ndarray:
    """Tripod position of a shot: weighted geometric median of per-frame camera centres."""
    P = np.array([c.C for c in cams])
    w = np.ones(len(P)) if weights is None else np.asarray(weights, float)
    x = np.median(P, 0)
    for _ in range(50):
        d = np.linalg.norm(P - x, axis=1) + 1e-6
        ww = w / d
        x_new = (ww[:, None] * P).sum(0) / ww.sum()
        if np.linalg.norm(x_new - x) < 1e-4:
            break
        x = x_new
    return x
