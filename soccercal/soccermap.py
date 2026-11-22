"""SoccerMap (Fernández & Bornn, ECML-PKDD 2020): a fully convolutional network that turns the game
state (``tactics.soccermap_tensor``, 13 channels on a 104 x 68 grid) into a probability for every
cell of the pitch.

Architecture as described in the paper: at 1x, 1/2x and 1/4x scales, two 5x5 convolutions (ReLU,
symmetric padding); a prediction head per scale (32 1x1 filters + ReLU, one 1x1 linear filter);
coarse predictions are upsampled (2x nearest neighbour + 32 3x3 filters + ReLU + one 3x3 linear
filter) and fused with the finer one by a 1x1 convolution; a sigmoid gives the probability surface.
It is trained with the paper's target-location loss: only the cell where the pass went is scored.

``pass_success`` loads the weights trained by ``tools/train_soccermap.py``: first on StatsBomb 360
passes (players in the broadcast image, no velocities), then on open tracking data (SkillCorner
broadcast tracking, IDSSE / DFL and Metrica optical tracking; see the README for the measured
accuracy) and returns, for a frame, the probability that a pass from the ball to each cell
is completed. The network's logit is blended with the paper's logistic baseline (hand-crafted
features of each pass, ``pass_features``): weights fitted on validation matches, the blend beats both
models on held-out matches (README).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).with_name("data") / "soccermap_pass_success.pt"


def _net():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Features(nn.Module):
        def __init__(self, c_in):
            super().__init__()
            self.c1 = nn.Conv2d(c_in, 32, 5)
            self.c2 = nn.Conv2d(32, 64, 5)

        def forward(self, x):
            x = F.relu(self.c1(F.pad(x, (2, 2, 2, 2), mode="replicate")))
            return F.relu(self.c2(F.pad(x, (2, 2, 2, 2), mode="replicate")))

    class Predict(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(64, 32, 1)
            self.c2 = nn.Conv2d(32, 1, 1)

        def forward(self, x):
            return self.c2(F.relu(self.c1(x)))

    class Up(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(1, 32, 3)
            self.c2 = nn.Conv2d(32, 1, 3)

        def forward(self, x, size):
            x = F.interpolate(x, size=size, mode="nearest")
            x = F.relu(self.c1(F.pad(x, (1, 1, 1, 1), mode="replicate")))
            return self.c2(F.pad(x, (1, 1, 1, 1), mode="replicate"))

    class SoccerMapNet(nn.Module):
        def __init__(self, c_in: int = 13):
            super().__init__()
            self.f1, self.f2, self.f4 = Features(c_in), Features(64), Features(64)
            self.p1, self.p2, self.p4 = Predict(), Predict(), Predict()
            self.u2, self.u4 = Up(), Up()
            self.fuse24 = nn.Conv2d(2, 1, 1)
            self.fuse12 = nn.Conv2d(2, 1, 1)

        def forward(self, x):  # (B, 13, 68, 104) -> logits (B, 1, 68, 104)
            f1 = self.f1(x)
            f2 = self.f2(F.max_pool2d(f1, 2))
            f4 = self.f4(F.max_pool2d(f2, 2))
            p2 = self.fuse24(torch.cat([self.u4(self.p4(f4), f2.shape[-2:]), self.p2(f2)], 1))
            return self.fuse12(torch.cat([self.u2(p2, f1.shape[-2:]), self.p1(f1)], 1))

    return SoccerMapNet


def SoccerMapNet(c_in: int = 13):
    return _net()(c_in)


_MODEL = None
_BLEND = None


def load(path: str | Path = WEIGHTS):
    """The trained pass-success network (CPU, eval mode); cached."""
    global _MODEL, _BLEND
    if _MODEL is None or Path(path) != WEIGHTS:
        import torch

        state = torch.load(path, map_location="cpu", weights_only=True)
        net = SoccerMapNet(13)
        net.load_state_dict({k: v.float() for k, v in state["model"].items()})
        net.eval()
        if Path(path) != WEIGHTS:
            return net
        _MODEL, _BLEND = net, state.get("blend")
    return _MODEL


def load_blend() -> dict | None:
    """The logistic baseline and the blend weights stored with the network (None if absent)."""
    load()
    return _BLEND


def pass_features(o, d, opp, mate, pitch=(105.0, 68.0)) -> np.ndarray:
    """(n, 13) hand-crafted features of passes from ``o`` to each of ``d`` (n, 2), attacking towards
    +x, as in the paper's logistic baseline: origin and destination, length, angles to goal,
    opponents near the destination and along the pass line, teammates near the destination
    (``opp``, ``mate``: (k, 2) positions; ``mate`` without the passer)."""
    L, W = pitch
    o = np.asarray(o, float)
    d = np.atleast_2d(np.asarray(d, float))
    opp = np.asarray(opp, float).reshape(-1, 2)
    mate = np.asarray(mate, float).reshape(-1, 2)
    seg = d - o
    if len(opp):
        t = np.clip(((opp[None] - o) * seg[:, None]).sum(-1) / np.maximum((seg * seg).sum(1), 1e-6)[:, None], 0, 1)
        line = np.linalg.norm(opp[None] - (o + t[..., None] * seg[:, None]), axis=-1).min(1)
        dd_opp = np.linalg.norm(d[:, None] - opp[None], axis=-1)
    else:
        line, dd_opp = np.full(len(d), 10.0), np.full((len(d), 1), 15.0)
    dd_mate = np.linalg.norm(d[:, None] - mate[None], axis=-1) if len(mate) else np.full((len(d), 1), 15.0)
    ang = lambda q: np.abs(np.arctan2(-q[..., 1], L / 2 - q[..., 0]))  # noqa: E731
    return np.column_stack([np.full(len(d), o[0] / L), np.full(len(d), o[1] / W), d[:, 0] / L, d[:, 1] / W,
                            np.linalg.norm(seg, axis=1) / 30, np.full(len(d), ang(o)), ang(d),
                            np.minimum(line, 10) / 10, np.minimum(dd_opp.min(1), 15) / 15,
                            np.minimum(dd_mate.min(1), 15) / 15, (dd_opp < 5).sum(1) / 3, (dd_mate < 5).sum(1) / 3,
                            np.abs(d[:, 1]) / W]).astype(np.float32)


def logistic_logit(H: np.ndarray, blend: dict) -> np.ndarray:
    """Logit of the logistic baseline for features ``H`` (n, 13)."""
    return ((H - np.asarray(blend["mu"])) / np.asarray(blend["sd"])) @ np.asarray(blend["w"]) + blend["b"]


def blend_logits(z_net: np.ndarray, z_logistic: np.ndarray, blend: dict) -> np.ndarray:
    a, b = blend["coef"]
    return a * z_net + b * z_logistic + blend["bias"]


def pass_success(s, team: int | None = None, model=None, blend: dict | None = None):
    """Probability that a pass from the ball to each cell is completed (``tactics.Surface``, 68 x 104
    cells, pitch coordinates), for ``team`` (default: the team in possession). With the shipped
    weights the network is blended with the logistic baseline; a ``model`` passed in is used alone
    unless ``blend`` is given too."""
    import torch

    from .tactics import Surface, _team_or_attacking, soccermap_tensor

    notes: list = []
    team = _team_or_attacking(s, team, notes)
    X = soccermap_tensor(s, team=team)
    if model is None:
        model, blend = load(), load_blend()
    with torch.no_grad():
        z = model(torch.from_numpy(X[None])).numpy()[0, 0].astype(float)
    L, W = s.pitch
    ny, nx = z.shape
    xg = (np.arange(nx) + 0.5) * L / nx - L / 2
    yg = (np.arange(ny) + 0.5) * W / ny - W / 2
    if blend is not None:
        flip = 1.0 if s.attacks_right[team] else -1.0  # the tensor's frame: ``team`` attacks +x
        ok = np.isfinite(s.pos).all(1)
        pos, ball = s.pos * flip, s.ball * flip
        if not np.isfinite(ball).all():
            c = ok & (s.pid == s.carrier)
            ball = pos[c][0] if c.any() else np.zeros(2)
        CX, CY = np.meshgrid(xg, yg)
        H = pass_features(ball, np.c_[CX.ravel(), CY.ravel()], pos[ok & (s.team != team)],
                          pos[ok & (s.team == team) & (s.pid != s.carrier)], (L, W))
        z = blend_logits(z, logistic_logit(H, blend).reshape(ny, nx), blend)
    p = 1.0 / (1.0 + np.exp(-z))
    if not s.attacks_right[team]:  # the tensor was rotated so that ``team`` attacks +x: rotate back
        p = p[::-1, ::-1]
    if not np.isfinite(s.ball).all():
        notes.append("balón desconocido: se usa la posición de quien lo lleva")
    return Surface(np.ascontiguousarray(p), xg, yg, team, "soccermap_pass_success", notes)
