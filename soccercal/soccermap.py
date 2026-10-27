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
is completed.
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


def load(path: str | Path = WEIGHTS):
    """The trained pass-success network (CPU, eval mode); cached."""
    global _MODEL
    if _MODEL is None or Path(path) != WEIGHTS:
        import torch

        state = torch.load(path, map_location="cpu", weights_only=True)
        net = SoccerMapNet(13)
        net.load_state_dict({k: v.float() for k, v in state["model"].items()})
        net.eval()
        if Path(path) != WEIGHTS:
            return net
        _MODEL = net
    return _MODEL


def pass_success(s, team: int | None = None, model=None):
    """Probability that a pass from the ball to each cell is completed (``tactics.Surface``, 68 x 104
    cells, pitch coordinates), for ``team`` (default: the team in possession)."""
    import torch

    from .tactics import Surface, _team_or_attacking, soccermap_tensor

    notes: list = []
    team = _team_or_attacking(s, team, notes)
    X = soccermap_tensor(s, team=team)
    net = model or load()
    with torch.no_grad():
        p = torch.sigmoid(net(torch.from_numpy(X[None]))).numpy()[0, 0]
    if not s.attacks_right[team]:  # the tensor was rotated so that ``team`` attacks +x: rotate back
        p = p[::-1, ::-1]
    L, W = s.pitch
    ny, nx = p.shape
    xg = (np.arange(nx) + 0.5) * L / nx - L / 2
    yg = (np.arange(ny) + 0.5) * W / ny - W / 2
    if not np.isfinite(s.ball).all():
        notes.append("balón desconocido: se usa la posición de quien lo lleva")
    return Surface(np.ascontiguousarray(p), xg, yg, team, "soccermap_pass_success", notes)
