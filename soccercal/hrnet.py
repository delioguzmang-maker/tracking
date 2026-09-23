"""HRNetV2-W48 heatmap network used for pitch keypoints and pitch lines.

Written from the HRNet paper (Wang et al., 2019) so that the parameter names
match the public checkpoints released with "No Bells, Just Whistles" /
PnLCalib (Gutiérrez-Pérez & Agudo). Those checkpoints are downloaded at run
time, never shipped with this package.

Output: a softmax over ``num_joints`` channels (the last one is background)
at half of the input resolution. Input: RGB in [0, 1], ideally 960x540.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1

# Stage layout of HRNetV2-W48 (NUM_MODULES, NUM_BLOCKS per branch, NUM_CHANNELS per branch).
W48_STAGES = {
    2: (1, [4, 4], [48, 96]),
    3: (4, [4, 4, 4], [48, 96, 192]),
    4: (3, [4, 4, 4, 4], [48, 96, 192, 384]),
}


def _bn(c: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(c, momentum=BN_MOMENTUM)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin: int, c: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, c, 3, stride, 1, bias=False)
        self.bn1 = _bn(c)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.bn2 = _bn(c)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, cin: int, c: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, c, 1, bias=False)
        self.bn1 = _bn(c)
        self.conv2 = nn.Conv2d(c, c, 3, stride, 1, bias=False)
        self.bn2 = _bn(c)
        self.conv3 = nn.Conv2d(c, c * 4, 1, bias=False)
        self.bn3 = _bn(c * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


def _res_layer(block, cin: int, c: int, n: int) -> nn.Sequential:
    down = None
    if cin != c * block.expansion:
        down = nn.Sequential(nn.Conv2d(cin, c * block.expansion, 1, bias=False), _bn(c * block.expansion))
    layers = [block(cin, c, 1, down)]
    layers += [block(c * block.expansion, c) for _ in range(1, n)]
    return nn.Sequential(*layers)


class HRModule(nn.Module):
    """Parallel multi-resolution branches followed by all-to-all fusion."""

    def __init__(self, channels: list[int], blocks: list[int]):
        super().__init__()
        nb = len(channels)
        self.num_branches = nb
        self.branches = nn.ModuleList(
            [_res_layer(BasicBlock, channels[i], channels[i], blocks[i]) for i in range(nb)]
        )
        fuse = []
        for i in range(nb):
            row = []
            for j in range(nb):
                if j > i:  # lower resolution -> 1x1 conv, upsampled in forward
                    row.append(nn.Sequential(nn.Conv2d(channels[j], channels[i], 1, 1, 0, bias=False), _bn(channels[i])))
                elif j == i:
                    row.append(None)
                else:  # higher resolution -> strided 3x3 convs
                    convs = []
                    for k in range(i - j):
                        last = k == i - j - 1
                        cout = channels[i] if last else channels[j]
                        seq = [nn.Conv2d(channels[j], cout, 3, 2, 1, bias=False), _bn(cout)]
                        if not last:
                            seq.append(nn.ReLU(inplace=True))
                        convs.append(nn.Sequential(*seq))
                    row.append(nn.Sequential(*convs))
            fuse.append(nn.ModuleList(row))
        self.fuse_layers = nn.ModuleList(fuse)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        xs = [branch(x) for branch, x in zip(self.branches, xs)]
        out = []
        for i in range(self.num_branches):
            y = None
            for j in range(self.num_branches):
                if j == i:
                    z = xs[j]
                elif j > i:
                    z = F.interpolate(self.fuse_layers[i][j](xs[j]), size=xs[i].shape[2:], mode="bilinear",
                                      align_corners=False)
                else:
                    z = self.fuse_layers[i][j](xs[j])
                y = z if y is None else y + z
            out.append(self.relu(y))
        return out


class HRNetHeatmap(nn.Module):
    def __init__(self, num_joints: int):
        super().__init__()
        self.num_joints = num_joints
        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        self.bn1 = _bn(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = _bn(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = _res_layer(Bottleneck, 64, 64, 4)

        prev = [256]
        for s in (2, 3, 4):
            n_mod, blocks, chans = W48_STAGES[s]
            setattr(self, f"transition{s - 1}", self._transition(prev, chans))
            setattr(self, f"stage{s}", nn.Sequential(*[HRModule(chans, blocks) for _ in range(n_mod)]))
            prev = chans

        c = sum(prev) + 64
        self.head = nn.Sequential(nn.Sequential(
            nn.Conv2d(c, c, 1), _bn(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, num_joints, 1), nn.Softmax(dim=1)))

    @staticmethod
    def _transition(prev: list[int], cur: list[int]) -> nn.ModuleList:
        layers = []
        for i, c in enumerate(cur):
            if i < len(prev):
                layers.append(None if prev[i] == c else nn.Sequential(
                    nn.Conv2d(prev[i], c, 3, 1, 1, bias=False), _bn(c), nn.ReLU(inplace=True)))
            else:
                convs = []
                for j in range(i + 1 - len(prev)):
                    cout = c if j == i - len(prev) else prev[-1]
                    convs.append(nn.Sequential(nn.Conv2d(prev[-1], cout, 3, 2, 1, bias=False), _bn(cout),
                                               nn.ReLU(inplace=True)))
                layers.append(nn.Sequential(*convs))
        return nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        skip = x
        x = self.relu(self.bn1(x))
        x = self.relu(self.bn2(self.conv2(x)))
        ys = [self.layer1(x)]
        for s in (2, 3, 4):
            trans = getattr(self, f"transition{s - 1}")
            xs = [ys[i] if t is None else t(ys[min(i, len(ys) - 1)]) for i, t in enumerate(trans)]
            ys = getattr(self, f"stage{s}")(xs)
        h, w = ys[0].shape[2:]
        feats = [ys[0]] + [F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False) for y in ys[1:]]
        x = torch.cat(feats, 1)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.head(torch.cat([x, skip], 1))


def build_keypoint_net() -> HRNetHeatmap:
    """57 pitch keypoints + background."""
    return HRNetHeatmap(58)


def build_line_net() -> HRNetHeatmap:
    """23 pitch line extremity maps + background."""
    return HRNetHeatmap(24)
