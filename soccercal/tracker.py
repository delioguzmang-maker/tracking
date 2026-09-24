"""Multi-player tracking on the pitch plane (metres), not in image space.

Broadcast cameras pan and zoom constantly, so image-space trackers (SORT/ByteTrack)
see every player "move" whenever the camera does. After calibration each detection
is a point on the pitch with a known, anisotropic uncertainty (large along the
viewing direction for far players), so we track there:

* constant-velocity Kalman filter per player, white-noise acceleration;
* measurement covariance propagated from pixel noise through the camera Jacobian;
* ByteTrack-style two-stage association (confident, then weak detections),
  Mahalanobis gating + jersey-colour term, Hungarian assignment;
* a track that has shown one team's kit never takes a detection that clearly wears the
  other team's kit (two players crossing must not swap identities);
* Rauch-Tung-Striebel smoothing afterwards (offline) for positions and velocities.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

CHI2_99 = 9.21  # 2 dof


def cv_model(dt: float, sigma_acc: float) -> tuple[np.ndarray, np.ndarray]:
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    q = sigma_acc ** 2
    Q1 = q * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
    Q = np.zeros((4, 4))
    Q[np.ix_([0, 2], [0, 2])] = Q1
    Q[np.ix_([1, 3], [1, 3])] = Q1
    return F, Q


HM = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])


@dataclass
class Track:
    id: int
    x: np.ndarray
    P: np.ndarray
    start: int
    hits: int = 1
    lost: int = 0
    confirmed: bool = False
    app: np.ndarray | None = None
    # per processed frame: (frame, z (2,), R (2,2), det_index or -1, score)
    obs: list = field(default_factory=list)
    last_frame: int = 0
    team_votes: np.ndarray = field(default_factory=lambda: np.zeros(2))

    @property
    def team(self) -> int:
        """0 / 1 once the kit is clear (>= 3 votes, 80 % agreement), else -1."""
        n = self.team_votes.sum()
        if n >= 3 and self.team_votes.max() >= 0.8 * n:
            return int(np.argmax(self.team_votes))
        return -1

    def predict(self, F, Q):
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q


class PitchTracker:
    def __init__(self, fps: float, sigma_acc: float = 4.0, high: float = 0.45, low: float = 0.15,
                 gate: float = CHI2_99, max_lost_s: float = 1.0, min_hits: int = 3, app_weight: float = 0.5,
                 app_max: float = 0.55, init_speed_sigma: float = 3.0):
        self.dt = 1.0 / fps
        self.F, self.Q = cv_model(self.dt, sigma_acc)
        self.high, self.low = high, low
        self.gate = gate
        self.max_lost = max(1, int(round(max_lost_s * fps)))
        self.min_hits = min_hits
        self.app_weight = app_weight
        self.app_max = app_max
        self.init_speed_sigma = init_speed_sigma
        self.tracks: list[Track] = []
        self.finished: list[Track] = []
        self._next = 1

    # --------------------------------------------------------------------- core
    def _cost(self, tracks: list[Track], z: np.ndarray, R: np.ndarray, desc: np.ndarray | None, gate: float,
              labels: np.ndarray | None = None):
        n, m = len(tracks), len(z)
        C = np.full((n, m), np.inf)
        for i, t in enumerate(tracks):
            S = HM @ t.P @ HM.T + R  # (m,2,2)
            d = z - (HM @ t.x)[None]
            Si = np.linalg.inv(S)
            md = np.einsum("mi,mij,mj->m", d, Si, d)
            ok = md <= gate
            if labels is not None and t.team >= 0:
                ok &= ~((labels >= 0) & (labels != t.team))  # other team's kit: never the same player
            c = md / gate
            if desc is not None and t.app is not None:
                has = np.isfinite(desc[:, 0])
                ad = np.where(has, 1.0 - np.clip(desc @ np.nan_to_num(t.app), 0, 1), 0.0)
                ok &= ad <= self.app_max
                c = c + self.app_weight * ad
            C[i, ok] = c[ok]
        return C

    def _match(self, tracks, z, R, desc, gate, labels=None):
        if not tracks or not len(z):
            return [], list(range(len(tracks))), list(range(len(z)))
        C = self._cost(tracks, z, R, desc, gate, labels)
        big = 1e6
        Cf = np.where(np.isfinite(C), C, big)
        r, c = linear_sum_assignment(Cf)
        pairs = [(i, j) for i, j in zip(r, c) if Cf[i, j] < big]
        mt = {i for i, _ in pairs}
        md = {j for _, j in pairs}
        return pairs, [i for i in range(len(tracks)) if i not in mt], [j for j in range(len(z)) if j not in md]

    def _update(self, t: Track, frame: int, z, R, score, di, desc, label: int = -1):
        S = HM @ t.P @ HM.T + R
        K = t.P @ HM.T @ np.linalg.inv(S)
        t.x = t.x + K @ (z - HM @ t.x)
        t.P = (np.eye(4) - K @ HM) @ t.P
        t.hits += 1
        t.lost = 0
        t.last_frame = frame
        t.obs.append((frame, z.copy(), R.copy(), di, float(score)))
        if label >= 0:
            t.team_votes[label] += 1
        if desc is not None and np.isfinite(desc[0]):
            t.app = desc.copy() if t.app is None else _renorm(0.9 * t.app + 0.1 * desc)
        if not t.confirmed and t.hits >= self.min_hits:
            t.confirmed = True

    def step(self, frame: int, z: np.ndarray, R: np.ndarray, scores: np.ndarray, desc: np.ndarray | None = None,
             visible=None, labels: np.ndarray | None = None) -> dict[int, int]:
        """Advance one processed frame. z: (N,2) pitch positions, R: (N,2,2) covariances.

        labels: optional (N,) team of each detection from its kit colour (0 / 1, -1 = unsure).

        visible: optional callable (M,2) -> bool mask telling whether pitch points are in the
        camera view; a track that leaves the view is ended right away (it will be re-linked by
        the offline stitcher when the player comes back) instead of drifting on prediction.
        Returns {detection index: track id} for confirmed tracks.
        """
        for t in self.tracks:
            t.predict(self.F, self.Q)
        z = np.asarray(z, float).reshape(-1, 2)
        hi = np.nonzero(scores >= self.high)[0]
        lo = np.nonzero((scores >= self.low) & (scores < self.high))[0]
        d_all = desc
        lab = np.full(len(z), -1, int) if labels is None else np.asarray(labels, int)

        conf = [t for t in self.tracks if t.confirmed]
        tent = [t for t in self.tracks if not t.confirmed]
        # 1) confident detections vs confirmed tracks
        dsub = None if d_all is None else d_all[hi]
        pairs, um_t, um_d = self._match(conf, z[hi], R[hi], dsub, self.gate, lab[hi])
        for i, j in pairs:
            self._update(conf[i], frame, z[hi[j]], R[hi[j]], scores[hi[j]], int(hi[j]),
                         None if d_all is None else d_all[hi[j]], lab[hi[j]])
        rem_t = [conf[i] for i in um_t]
        rem_hi = hi[um_d]
        # 2) weak detections vs the remaining confirmed tracks (tighter gate, no appearance: weak boxes are partial)
        pairs, um_t2, _ = self._match(rem_t, z[lo], R[lo], None, self.gate * 0.5, lab[lo])
        for i, j in pairs:
            self._update(rem_t[i], frame, z[lo[j]], R[lo[j]], scores[lo[j]], int(lo[j]), None)
        # 3) remaining confident detections vs tentative tracks
        dsub = None if d_all is None else d_all[rem_hi]
        pairs, _, um_d3 = self._match(tent, z[rem_hi], R[rem_hi], dsub, self.gate, lab[rem_hi])
        for i, j in pairs:
            k = int(rem_hi[j])
            self._update(tent[i], frame, z[k], R[k], scores[k], k, None if d_all is None else d_all[k], lab[k])
        # 4) births
        for j in um_d3:
            k = int(rem_hi[j])
            P0 = np.zeros((4, 4))
            P0[:2, :2] = R[k]
            P0[2, 2] = P0[3, 3] = self.init_speed_sigma ** 2
            t = Track(self._next, np.r_[z[k], 0.0, 0.0], P0, start=frame, last_frame=frame)
            t.obs.append((frame, z[k].copy(), R[k].copy(), k, float(scores[k])))
            if lab[k] >= 0:
                t.team_votes[lab[k]] += 1
            if d_all is not None and np.isfinite(d_all[k][0]):
                t.app = d_all[k].copy()
            self._next += 1
            self.tracks.append(t)
        # 5) bookkeeping
        alive = []
        for t in self.tracks:
            if t.last_frame != frame:
                t.lost += 1
            gone = t.lost > (self.max_lost if t.confirmed else 1)
            if not gone and visible is not None and t.lost > 0:
                gone = not bool(visible(t.x[None, :2])[0])
            if gone:
                if t.confirmed:
                    self.finished.append(t)
            else:
                alive.append(t)
        self.tracks = alive
        return {o[3]: t.id for t in self.tracks if t.confirmed and t.last_frame == frame for o in t.obs[-1:]}

    def close(self) -> list[Track]:
        out = self.finished + [t for t in self.tracks if t.confirmed]
        self.tracks, self.finished = [], []
        return sorted(out, key=lambda t: t.id)


def _renorm(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ------------------------------------------------------------------ smoothing
def rts_smooth(frames: np.ndarray, z: np.ndarray, R: np.ndarray, fps: float, sigma_acc: float = 4.0,
               out_frames: np.ndarray | None = None, frame_step: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kalman filter + RTS smoother over a (possibly gappy) observation sequence.

    frames: (N,) processed-frame indices (increasing); z: (N,2); R: (N,2,2).
    Returns (out_frames, pos (T,2), vel (T,2)) for every frame between first and last
    observation with step ``frame_step`` (gaps are bridged by the motion model).
    """
    frames = np.asarray(frames, int)
    if out_frames is None:
        out_frames = np.arange(frames[0], frames[-1] + 1, frame_step)
    T = len(out_frames)
    F, Q = cv_model(frame_step / fps, sigma_acc)
    obs_at = {int(f): i for i, f in enumerate(frames)}
    xs = np.zeros((T, 4))
    Ps = np.zeros((T, 4, 4))
    xp = np.zeros((T, 4))
    Pp = np.zeros((T, 4, 4))
    x = np.r_[z[0], 0.0, 0.0]
    P = np.diag([R[0][0, 0], R[0][1, 1], 9.0, 9.0])
    for k, f in enumerate(out_frames):
        if k > 0:
            x = F @ x
            P = F @ P @ F.T + Q
        xp[k], Pp[k] = x, P
        i = obs_at.get(int(f))
        if i is not None:
            S = HM @ P @ HM.T + R[i]
            K = P @ HM.T @ np.linalg.inv(S)
            x = x + K @ (z[i] - HM @ x)
            P = (np.eye(4) - K @ HM) @ P
        xs[k], Ps[k] = x, P
    for k in range(T - 2, -1, -1):
        G = Ps[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xs[k] + G @ (xs[k + 1] - xp[k + 1])
        Ps[k] = Ps[k] + G @ (Ps[k + 1] - Pp[k + 1]) @ G.T
    return out_frames, xs[:, :2], xs[:, 2:]
