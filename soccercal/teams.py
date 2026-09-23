"""Team / referee assignment from jersey colour descriptors.

Descriptors (``detect.jersey_descriptor``) are clustered over the whole video with
k-means (k=8); clusters whose centroids are similar are merged, then the two largest
groups are the teams and everything else is "other" (referees, goalkeepers). A
goalkeeper is recognised afterwards by where he plays (``assign_goalkeepers``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OTHER = -1


def _kmeans(X: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(X)
    k = min(k, n)
    # k-means++ init
    C = [X[rng.integers(n)]]
    for _ in range(1, k):
        d = np.min(((X[:, None] - np.array(C)[None]) ** 2).sum(-1), 1)
        p = d / d.sum() if d.sum() > 0 else None
        C.append(X[rng.choice(n, p=p)])
    C = np.array(C)
    for _ in range(iters):
        lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
        newC = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
    return C, lab


@dataclass
class TeamModel:
    centroids: np.ndarray  # (2, D) unit vectors (Hellinger space)
    other: np.ndarray  # (K, D) centroids of the "other" groups
    spread: float  # typical distance of a team member to its centroid

    def distances(self, X: np.ndarray) -> np.ndarray:
        """(N, 2 + K): distance of each descriptor to team 0, team 1 and each 'other' group."""
        allc = np.vstack([self.centroids, self.other]) if len(self.other) else self.centroids
        return np.sqrt(np.maximum(((X[:, None] - allc[None]) ** 2).sum(-1), 0))

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Team label (0, 1 or OTHER) and a confidence in [0, 1] per descriptor."""
        d = self.distances(X)
        lab = np.argmin(d, 1)
        srt = np.sort(d, 1)
        margin = (srt[:, 1] - srt[:, 0]) / (srt[:, 1] + 1e-9) if d.shape[1] > 1 else np.ones(len(X))
        out = np.where(lab < 2, lab, OTHER)
        return out, np.clip(margin, 0, 1)

    def log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Unnormalised log-likelihood of (team0, team1, other) per descriptor."""
        d = self.distances(X)
        s2 = 2 * max(self.spread, 0.05) ** 2
        ll = -(d ** 2) / s2
        other = ll[:, 2:].max(1) if d.shape[1] > 2 else np.full(len(X), -9.0)
        return np.stack([ll[:, 0], ll[:, 1], other], 1)


def fit_team_model(X: np.ndarray, k: int = 8, merge_dist: float = 0.35, seed: int = 0) -> TeamModel | None:
    X = X[np.isfinite(X).all(1)]
    if len(X) < 10:
        return None
    C, lab = _kmeans(X, k, seed=seed)
    sizes = np.bincount(lab, minlength=len(C)).astype(float)
    # agglomerative merge of similar centroids (same kit under different light)
    groups = [[i] for i in range(len(C))]

    def gc(g):
        w = sizes[g]
        return (C[g] * w[:, None]).sum(0) / max(w.sum(), 1e-9)

    while True:
        best = None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                d = np.linalg.norm(gc(groups[a]) - gc(groups[b]))
                if d < merge_dist and (best is None or d < best[0]):
                    best = (d, a, b)
        if best is None:
            break
        _, a, b = best
        groups[a] += groups[b]
        groups.pop(b)
    gsize = np.array([sizes[g].sum() for g in groups])
    order = np.argsort(-gsize)
    if len(groups) < 2:
        return None
    teams = [gc(groups[order[0]]), gc(groups[order[1]])]
    others = [gc(groups[i]) for i in order[2:] if gsize[i] > 0]
    tm = TeamModel(np.array(teams), np.array(others).reshape(-1, X.shape[1]), 0.2)
    lab2, _ = tm.predict(X)
    dd = tm.distances(X)
    mem = lab2 >= 0
    if mem.any():
        tm.spread = float(np.median(dd[mem, lab2[mem]]))
    return tm


def decide(distances: np.ndarray, other_margin: float = 0.15, team_max: float = 0.55) -> tuple[int, float]:
    """Team of one tracklet/identity from its detections' distances to every colour group.

    distances: (N, 2 + K) as returned by ``TeamModel.distances``. A kit usually splits
    into sub-clusters (sun / shade); so "other" (referee, goalkeeper) is only chosen when
    the person is far from both teams *and* clearly closer to an "other" group.
    Returns (label, confidence in [0, 1])."""
    if len(distances) == 0:
        return OTHER, 0.0
    d = np.median(distances, 0)
    dt = d[:2]
    j = int(np.argmin(dt))
    if d.shape[0] > 2:
        do = d[2:].min()
        if dt[j] > team_max and do < dt[j] - other_margin:
            return OTHER, float(np.clip((dt[j] - do) / (dt[j] + 1e-9), 0, 1))
    conf = float(np.clip((dt[1 - j] - dt[j]) / (dt[1 - j] + 1e-9), 0, 1))
    return j, conf
