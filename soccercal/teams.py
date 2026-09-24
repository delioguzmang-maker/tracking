"""Team / referee assignment from jersey colour descriptors.

Descriptors (``detect.jersey_descriptor``) of the whole video are clustered with
k-means (k=10). A kit never forms one tidy cluster: sun and shade, near and far
players, the side with the number or the sponsor all give sub-clusters. So:

* sub-clusters are merged bottom-up (closest first) into groups, except that once only
  two big groups are left (the teams, each >= 15 % of all detections) they are never
  merged together (while there are three or more, one kit is split and the closest two
  merge), and nothing merges across a distance > ``merge_max`` (referee, goalkeepers,
  linesmen stay separate: their kits are far from both teams);
* a group keeps all its sub-centroids: the distance to a team is the distance to its
  *nearest* sub-cluster, so a player seen from close (number and sponsor visible) is
  still recognised;
* a person's team is voted over all his detections.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OTHER = -1


def _kmeans(X: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(X)
    k = min(k, n)
    C = [X[rng.integers(n)]]  # k-means++ init
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
    groups: list  # list of (m_i, D) arrays of sub-centroids; groups[0], groups[1] are the teams
    sizes: np.ndarray  # detections per group

    @property
    def centroids(self) -> np.ndarray:  # size-weighted team centres (for display / compatibility)
        return np.array([g.mean(0) for g in self.groups[:2]])

    def distances(self, X: np.ndarray) -> np.ndarray:
        """(N, n_groups): distance of each descriptor to the nearest sub-cluster of each group."""
        X = np.asarray(X, float).reshape(-1, self.groups[0].shape[1])
        out = np.empty((len(X), len(self.groups)))
        for j, g in enumerate(self.groups):
            out[:, j] = np.sqrt(np.maximum(((X[:, None] - g[None]) ** 2).sum(-1), 0)).min(1)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        lab = np.argmin(self.distances(X), 1)
        return np.where(lab < 2, lab, OTHER)


def fit_team_model(X: np.ndarray, k: int = 10, merge_max: float = 0.7, big: float = 0.15,
                   seed: int = 0) -> TeamModel | None:
    X = X[np.isfinite(X).all(1)]
    if len(X) < 20:
        return None
    C, lab = _kmeans(X, k, seed=seed)
    sizes = np.bincount(lab, minlength=len(C)).astype(float)
    keep = sizes > 0
    C, sizes = C[keep], sizes[keep]
    groups = [[i] for i in range(len(C))]
    total = sizes.sum()

    def gsize(g):
        return sizes[g].sum()

    def gdist(a, b):  # average linkage between sub-centroids
        d = np.linalg.norm(C[a][:, None] - C[b][None], axis=2)
        w = np.outer(sizes[a], sizes[b])
        return float((d * w).sum() / w.sum())

    while len(groups) > 2:
        best = None
        n_big = sum(gsize(g) >= big * total for g in groups)
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                if n_big <= 2 and gsize(groups[a]) >= big * total and gsize(groups[b]) >= big * total:
                    continue  # the two teams never merge (with 3+ big groups, one kit is split in two)
                d = gdist(groups[a], groups[b])
                if d <= merge_max and (best is None or d < best[0]):
                    best = (d, a, b)
        if best is None:
            break
        _, a, b = best
        groups[a] += groups[b]
        groups.pop(b)
    order = sorted(range(len(groups)), key=lambda i: -gsize(groups[i]))
    if len(order) < 2:
        return None
    return TeamModel([C[groups[i]] for i in order], np.array([gsize(groups[i]) for i in order]))


def decide(distances: np.ndarray, has_number: bool = False, other_share: float = 0.6) -> tuple[int, float]:
    """Team of a tracklet / identity from its detections' distances to the colour groups.

    Every detection votes for its nearest group. "Other" (referee, goalkeeper, linesman)
    needs a clear majority of votes for non-team groups and no shirt number read
    (referees do not wear numbers). Returns (0, 1 or OTHER, confidence in [0, 1])."""
    if len(distances) == 0:
        return OTHER, 0.0
    near = np.argmin(distances, 1)
    f = np.array([(near == 0).mean(), (near == 1).mean()])
    fo = 1.0 - f.sum()
    if fo >= other_share and not has_number:
        return OTHER, float(fo)
    if f.sum() == 0:  # only "other" votes but a number was read: nearest team by distance
        j = int(np.argmin(np.median(distances[:, :2], 0)))
        return j, 0.3
    j = int(np.argmax(f))
    return j, float(abs(f[0] - f[1]) / f.sum())
