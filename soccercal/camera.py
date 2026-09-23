"""Pinhole broadcast camera: projection, ground back-projection, robust fitting.

World frame = SkillCorner frame (see ``pitch.py``). Model: square pixels, principal
point at the image centre, no skew, no lens distortion; that is accurate for
broadcast main cameras and keeps the fit well conditioned (7 parameters:
focal length, rotation, translation; 4 if the camera position is known).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from . import pitch


@dataclass
class Camera:
    f: float
    R: np.ndarray  # world -> camera rotation
    C: np.ndarray  # camera centre, world metres
    width: int
    height: int
    cx: float | None = None
    cy: float | None = None
    info: dict = field(default_factory=dict)  # fit diagnostics

    def __post_init__(self):
        self.R = np.asarray(self.R, float)
        self.C = np.asarray(self.C, float).reshape(3)
        if self.cx is None:
            self.cx = self.width / 2.0
        if self.cy is None:
            self.cy = self.height / 2.0

    # ------------------------------------------------------------------ matrices
    @property
    def K(self) -> np.ndarray:
        return np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1.0]])

    @property
    def t(self) -> np.ndarray:
        return -self.R @ self.C

    @property
    def P(self) -> np.ndarray:
        return self.K @ np.hstack([self.R, self.t[:, None]])

    @property
    def H(self) -> np.ndarray:
        """Homography ground plane (x, y, 1) -> image."""
        return self.P[:, [0, 1, 3]]

    @property
    def rvec(self) -> np.ndarray:
        return cv2.Rodrigues(self.R)[0].ravel()

    def copy(self, **kw) -> "Camera":
        d = dict(f=self.f, R=self.R.copy(), C=self.C.copy(), width=self.width, height=self.height,
                 cx=self.cx, cy=self.cy, info=dict(self.info))
        d.update(kw)
        return Camera(**d)

    # --------------------------------------------------------------- projection
    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points (N,3) -> pixels (N,2) and depth (N,). Depth <= 0 means behind."""
        X = np.asarray(X, float).reshape(-1, 3)
        Xc = (X - self.C) @ self.R.T
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = np.stack([self.f * Xc[:, 0] / z + self.cx, self.f * Xc[:, 1] / z + self.cy], 1)
        return uv, z

    def ground_to_image(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy = np.asarray(xy, float).reshape(-1, 2)
        return self.project(np.hstack([xy, np.zeros((len(xy), 1))]))

    def rays(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, float).reshape(-1, 2)
        d = np.stack([(uv[:, 0] - self.cx) / self.f, (uv[:, 1] - self.cy) / self.f, np.ones(len(uv))], 1)
        return d @ self.R  # camera -> world direction (R^T d)

    def image_to_plane(self, uv: np.ndarray, z: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Back-project pixels onto the horizontal plane at height ``z``.

        Returns (N,2) ground coordinates and a validity mask (ray hits the plane in front
        of the camera). Invalid rows are NaN.
        """
        d = self.rays(uv)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (z - self.C[2]) / d[:, 2]
        ok = np.isfinite(s) & (s > 0)
        P = self.C[None, :] + s[:, None] * d
        out = P[:, :2].copy()
        out[~ok] = np.nan
        return out, ok

    def image_to_ground(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.image_to_plane(uv, 0.0)

    def ground_jacobian(self, xy: np.ndarray) -> np.ndarray:
        """d(image uv)/d(ground xy) at ground points: (N, 2, 2). Used for measurement noise."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        H = self.H
        p = np.hstack([xy, np.ones((len(xy), 1))]) @ H.T
        w = p[:, 2]
        J = np.empty((len(xy), 2, 2))
        for i in range(2):
            for j in range(2):
                J[:, i, j] = (H[i, j] * w - p[:, i] * H[2, j]) / w ** 2
        return J

    def pixel_height(self, xy: np.ndarray, height: float = 1.8) -> np.ndarray:
        """Expected image height (px) of a vertical segment of ``height`` metres standing at xy."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        a, za = self.project(np.hstack([xy, np.zeros((len(xy), 1))]))
        b, zb = self.project(np.hstack([xy, np.full((len(xy), 1), height)]))
        h = np.linalg.norm(a - b, axis=1)
        h[(za <= 0) | (zb <= 0)] = np.nan
        return h

    # ------------------------------------------------------------ view footprint
    def _edge_ground_points(self, n: int = 64) -> list[np.ndarray]:
        w, h = self.width - 1, self.height - 1
        t = np.linspace(0, 1, n)
        edges = [np.stack([t * w, np.full(n, h)], 1),       # bottom, left -> right
                 np.stack([np.full(n, w), h - t * h], 1),   # right, bottom -> top
                 np.stack([w - t * w, np.zeros(n)], 1),     # top, right -> left
                 np.stack([np.zeros(n), t * h], 1)]         # left, top -> bottom
        return [self.image_to_ground(e)[0] for e in edges]

    def footprint(self, max_dist: float = 150.0) -> np.ndarray:
        """Visible ground area as a polygon (M,2), clipped at ``max_dist`` metres from the camera."""
        pts = np.concatenate(self._edge_ground_points(), 0)
        ok = np.isfinite(pts).all(1)
        pts = pts[ok]
        d = pts - self.C[:2]
        r = np.linalg.norm(d, axis=1)
        far = r > max_dist
        pts[far] = self.C[:2] + d[far] / r[far, None] * max_dist
        # rays above the horizon: close the polygon along the max_dist arc
        return pts

    def corners_projection(self, length: float = pitch.LENGTH, width: float = pitch.WIDTH,
                           margin_x: float = 5.0, margin_y: float = 5.0) -> dict:
        """SkillCorner ``image_corners_projection``: the image corners on the ground.

        A corner above the horizon (or beyond the box ``|x| <= L/2+mx, |y| <= W/2+my``) is
        replaced by the first point down the image side edge that lands inside the box,
        which is how SkillCorner's own files clip the far corners (``|y| = W/2 + 5``).
        """
        bx, by = length / 2 + margin_x, width / 2 + margin_y
        w, h = self.width - 1.0, self.height - 1.0

        def ground(u, v):
            g, _ = self.image_to_ground(np.array([[u, v]], float))
            return g[0]

        def inside(p):
            return bool(np.isfinite(p).all() and abs(p[0]) <= bx + 1e-6 and abs(p[1]) <= by + 1e-6)

        out = {}
        for key, u in (("left", 0.0), ("right", w)):
            vs = np.linspace(0.0, h, 256)
            ok = [inside(ground(u, v)) for v in vs]
            top = bottom = None
            if any(ok):
                i = ok.index(True)
                if i == 0:
                    top = ground(u, 0.0)
                else:  # bisect the boundary between vs[i-1] (outside) and vs[i] (inside)
                    lo, hi = vs[i - 1], vs[i]
                    for _ in range(30):
                        mid = 0.5 * (lo + hi)
                        lo, hi = (lo, mid) if inside(ground(u, mid)) else (mid, hi)
                    top = ground(u, hi)
                j = len(ok) - 1 - ok[::-1].index(True)
                if j == len(ok) - 1:
                    bottom = ground(u, h)
                else:
                    lo, hi = vs[j], vs[j + 1]
                    for _ in range(30):
                        mid = 0.5 * (lo + hi)
                        lo, hi = (mid, hi) if inside(ground(u, mid)) else (lo, mid)
                    bottom = ground(u, lo)
            for name, p in (("top_" + key, top), ("bottom_" + key, bottom)):
                out[f"x_{name}"] = None if p is None else round(float(p[0]), 2)
                out[f"y_{name}"] = None if p is None else round(float(p[1]), 2)
        order = ("top_left", "bottom_left", "bottom_right", "top_right")
        return {f"{c}_{k}": out[f"{c}_{k}"] for k in order for c in ("x", "y")}

    # ---------------------------------------------------------------- summaries
    def pan_tilt_roll(self) -> tuple[float, float, float]:
        """Degrees. pan: heading of the optical axis in the ground plane (0 = +y, positive
        towards +x); tilt: angle below the horizon; roll: rotation about the optical axis."""
        fwd = self.R[2]  # optical axis in world
        pan = np.degrees(np.arctan2(fwd[0], fwd[1]))
        tilt = np.degrees(np.arcsin(-fwd[2]))
        right = self.R[0]
        horiz = np.cross(fwd, [0, 0, 1.0])
        n = np.linalg.norm(horiz)
        roll = 0.0 if n < 1e-9 else np.degrees(np.arctan2(np.dot(np.cross(horiz / n, right), fwd),
                                                          np.dot(horiz / n, right)))
        return float(pan), float(tilt), float(roll)

    def is_plausible(self) -> bool:
        """Physically sensible broadcast camera (above the ground, looking down, small roll)."""
        pan, tilt, roll = self.pan_tilt_roll()
        return bool(2.0 < self.C[2] < 150.0 and -5.0 < tilt < 80.0 and abs(roll) < 12.0
                    and 0.2 * self.width < self.f < 40 * self.width)

    def to_dict(self) -> dict:
        pan, tilt, roll = self.pan_tilt_roll()
        return {"f": float(self.f), "cx": float(self.cx), "cy": float(self.cy),
                "rvec": self.rvec.tolist(), "C": self.C.tolist(), "width": self.width, "height": self.height,
                "pan": pan, "tilt": tilt, "roll": roll}

    @staticmethod
    def from_dict(d: dict) -> "Camera":
        return Camera(f=d["f"], R=cv2.Rodrigues(np.asarray(d["rvec"], float))[0], C=np.asarray(d["C"]),
                      width=int(d["width"]), height=int(d["height"]), cx=d.get("cx"), cy=d.get("cy"))


# ======================================================================= fitting

def rotation_from_rvec(rvec) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]


def focal_from_homography(H: np.ndarray, cx: float, cy: float) -> float | None:
    """Closed-form focal length from a plane->image homography (square pixels, known pp)."""
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    h = T @ H
    h1, h2 = h[:, 0], h[:, 1]
    a1 = h1[0] * h2[0] + h1[1] * h2[1]
    b1 = h1[2] * h2[2]
    a2 = h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2
    b2 = h1[2] ** 2 - h2[2] ** 2
    # a/f^2 + b = 0  ->  u = 1/f^2 = -b/a, least squares over both constraints
    den = a1 * a1 + a2 * a2
    if den <= 0:
        return None
    u = -(a1 * b1 + a2 * b2) / den
    if not np.isfinite(u) or u <= 0:
        return None
    return float(1.0 / np.sqrt(u))


def pose_from_homography(H: np.ndarray, f: float, cx: float, cy: float):
    """Rotation / translation from ground homography and known intrinsics."""
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    M = np.linalg.solve(K, H)
    lam = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    M = M * lam
    if M[2, 2] < 0:  # pitch origin must be in front of the camera
        M = -M
    r1, r2, t = M[:, 0], M[:, 1], M[:, 2]
    R = np.stack([r1, r2, np.cross(r1, r2)], 1)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        return None
    return R, t


@dataclass
class Correspondences:
    """Image observations for a fit (pixels in full-resolution image coordinates)."""
    pts_world: np.ndarray  # (N,3)
    pts_image: np.ndarray  # (N,2)
    pts_weight: np.ndarray  # (N,)
    pts_id: np.ndarray  # (N,) keypoint ids
    line_world: np.ndarray  # (M,2,3) world line through two points
    line_image: np.ndarray  # (M,2) observed image point lying on that line
    line_weight: np.ndarray  # (M,)
    line_id: np.ndarray  # (M,)

    @staticmethod
    def empty() -> "Correspondences":
        return Correspondences(np.zeros((0, 3)), np.zeros((0, 2)), np.zeros(0), np.zeros(0, int),
                               np.zeros((0, 2, 3)), np.zeros((0, 2)), np.zeros(0), np.zeros(0, int))


def _residuals(cam: Camera, obs: Correspondences) -> tuple[np.ndarray, np.ndarray]:
    rp = np.zeros(0)
    if len(obs.pts_world):
        uv, z = cam.project(obs.pts_world)
        d = uv - obs.pts_image
        d[z <= 0] = 1e4
        rp = d
    rl = np.zeros(0)
    if len(obs.line_world):
        a, za = cam.project(obs.line_world[:, 0])
        b, zb = cam.project(obs.line_world[:, 1])
        ab = b - a
        n = np.linalg.norm(ab, axis=1) + 1e-9
        q = obs.line_image - a
        rl = (ab[:, 0] * q[:, 1] - ab[:, 1] * q[:, 0]) / n
        rl[(za <= 0) | (zb <= 0)] = 1e4
    return rp, rl


def residual_norms(cam: Camera, obs: Correspondences) -> tuple[np.ndarray, np.ndarray]:
    rp, rl = _residuals(cam, obs)
    return (np.linalg.norm(rp, axis=1) if len(rp) else rp), np.abs(rl)


def refine(cam: Camera, obs: Correspondences, fix_center: bool = False, loss_scale_px: float | None = None,
           f_prior: tuple[float, float] | None = None, line_mask=None, line_points: np.ndarray | None = None,
           line_weight: float = 30.0, max_nfev: int = 100) -> Camera:
    """Robust least-squares refinement of focal length + pose.

    Residuals: keypoint reprojection errors, point-on-line errors (``obs.line_*``) and,
    when ``line_mask`` is given, the chamfer distance of every pitch-marking point in
    ``line_points`` to the detected line pixels (weighted to count as ``line_weight``
    keypoints in total). ``fix_center`` keeps the camera position (tripod model).
    ``f_prior``: optional (f0, sigma_log) soft prior on the focal length.
    """
    scale = loss_scale_px or max(2.0, 0.0015 * cam.width)
    wp = np.sqrt(obs.pts_weight)[:, None] if len(obs.pts_weight) else np.zeros((0, 1))
    wl = np.sqrt(obs.line_weight) if len(obs.line_weight) else np.zeros(0)
    use_chamfer = line_mask is not None and line_points is not None and len(line_points) >= 20
    wc = np.sqrt(line_weight / len(line_points)) if use_chamfer else 0.0
    cap = 5.0 * scale

    def unpack(p):
        f = float(np.exp(p[0]))
        R = rotation_from_rvec(p[1:4])
        C = cam.C if fix_center else -R.T @ p[4:7]
        return cam.copy(f=f, R=R, C=C)

    def fun(p):
        c = unpack(p)
        rp, rl = _residuals(c, obs)
        r = [(rp * wp).ravel(), rl * wl]
        if use_chamfer:
            uv, z = c.project(line_points)
            d = line_mask.distances(uv, cap)
            d[z <= 0] = cap
            r.append(wc * d)
        if f_prior is not None:
            r.append(np.array([(p[0] - np.log(f_prior[0])) / f_prior[1] * scale]))
        return np.concatenate(r)

    p0 = [np.log(cam.f), *cam.rvec]
    if not fix_center:
        p0 += list(cam.t)
    p0 = np.asarray(p0, float)
    n_res = 2 * len(obs.pts_world) + len(obs.line_world) + (20 if use_chamfer else 0)
    if n_res < len(p0):
        return cam
    try:
        sol = least_squares(fun, p0, loss="soft_l1", f_scale=scale, max_nfev=max_nfev, x_scale="jac")
    except (ValueError, np.linalg.LinAlgError):
        return cam
    return unpack(sol.x)


def init_from_ground_points(world_xy: np.ndarray, image_uv: np.ndarray, width: int, height: int,
                            ransac_px: float | None = None) -> tuple[Camera | None, np.ndarray]:
    """Initial camera from >= 4 ground-plane correspondences (RANSAC homography + focal search)."""
    n = len(world_xy)
    inl = np.zeros(n, bool)
    if n < 4 or _degenerate(world_xy):
        return None, inl
    thr = ransac_px or max(3.0, 0.008 * width)
    # Classic RANSAC on purpose: OpenCV 5's USAC/MAGSAC variants return empty inlier
    # masks for these small point sets (verified), OpenCV 4's work; RANSAC works on both.
    H, mask = cv2.findHomography(world_xy.astype(np.float64), image_uv.astype(np.float64), cv2.RANSAC, thr,
                                 maxIters=2000, confidence=0.999)
    if H is None or mask is None:
        return None, inl
    inl = mask.ravel().astype(bool)
    if inl.sum() < 4 or _degenerate(world_xy[inl]):
        return None, inl
    cx, cy = width / 2.0, height / 2.0
    cands = []
    f0 = focal_from_homography(H, cx, cy)
    if f0 is not None and 0.2 * width < f0 < 40 * width:
        cands.append(f0)
    cands += list(width * np.geomspace(0.4, 12.0, 14))
    obj = np.hstack([world_xy[inl], np.zeros((inl.sum(), 1))]).astype(np.float64)
    img = image_uv[inl].astype(np.float64)
    best, best_err = None, np.inf
    for f in cands:
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        pose = pose_from_homography(H, f, cx, cy)
        sols = []
        if pose is not None:
            sols.append((cv2.Rodrigues(pose[0])[0], pose[1].reshape(3, 1)))
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
        if ok:
            sols.append((rvec, tvec))
        for rvec, tvec in sols:
            R = cv2.Rodrigues(rvec)[0]
            cam = Camera(f=f, R=R, C=(-R.T @ tvec).ravel(), width=width, height=height)
            if cam.C[2] <= 0:
                continue
            uv, z = cam.project(obj)
            if (z <= 0).any():
                continue
            err = np.sqrt(np.mean(np.sum((uv - img) ** 2, 1)))
            if err < best_err:
                best, best_err = cam, err
    return best, inl


def _degenerate(xy: np.ndarray) -> bool:
    """True if ground points are (nearly) collinear -> homography ill-posed."""
    if len(xy) < 4:
        return True
    c = xy - xy.mean(0)
    s = np.linalg.svd(c, compute_uv=False)
    return bool(s[1] < 0.5 * np.sqrt(len(xy)))  # < ~0.5 m spread off the main axis


def relative_rotation_angle(R1: np.ndarray, R2: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(R1 @ R2.T).magnitude()))
