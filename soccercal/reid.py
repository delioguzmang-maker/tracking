"""Appearance embeddings for re-identification: OSNet (Zhou et al., "Omni-Scale Feature Learning for
Person Re-Identification", ICCV 2019).

The architecture is written here from the paper / reference design so that only the published
weights are downloaded (``kaiyangzhou/osnet`` on HuggingFace, MIT licence; the MSMT17-trained
models). ``Embedder(crops)`` returns L2-normalised 512-d vectors: two crops of the same person are
close (cosine similarity near 1) even with the same kit, from build, hair, skin, boots and so on.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

URL = "https://huggingface.co/kaiyangzhou/osnet/resolve/main/{}"
FILES = {
    "x0_25": ("osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_"
              "jitter.pth", [16, 64, 96, 128]),
    "x1_0": ("osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_"
             "jitter.pth", [64, 256, 384, 512]),
}


def _build(channels):
    import torch.nn as nn
    import torch.nn.functional as F

    class ConvLayer(nn.Module):
        def __init__(self, cin, cout, k, stride=1, padding=0, groups=1):
            super().__init__()
            self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=padding, bias=False, groups=groups)
            self.bn = nn.BatchNorm2d(cout)

        def forward(self, x):
            return F.relu(self.bn(self.conv(x)))

    class Conv1x1(ConvLayer):
        def __init__(self, cin, cout, stride=1, groups=1):
            super().__init__(cin, cout, 1, stride, 0, groups)

    class Conv1x1Linear(nn.Module):
        def __init__(self, cin, cout, stride=1):
            super().__init__()
            self.conv = nn.Conv2d(cin, cout, 1, stride=stride, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(cout)

        def forward(self, x):
            return self.bn(self.conv(x))

    class LightConv3x3(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.conv1 = nn.Conv2d(cin, cout, 1, bias=False)
            self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False, groups=cout)
            self.bn = nn.BatchNorm2d(cout)

        def forward(self, x):
            return F.relu(self.bn(self.conv2(self.conv1(x))))

    class ChannelGate(nn.Module):
        def __init__(self, c, reduction=16):
            super().__init__()
            self.fc1 = nn.Conv2d(c, c // reduction, 1, bias=True)
            self.fc2 = nn.Conv2d(c // reduction, c, 1, bias=True)

        def forward(self, x):
            g = F.adaptive_avg_pool2d(x, 1)
            return x * self.fc2(F.relu(self.fc1(g))).sigmoid()

    class OSBlock(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            mid = cout // 4
            self.conv1 = Conv1x1(cin, mid)
            self.conv2a = LightConv3x3(mid, mid)
            self.conv2b = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid))
            self.conv2c = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(3)])
            self.conv2d = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(4)])
            self.gate = ChannelGate(mid)
            self.conv3 = Conv1x1Linear(mid, cout)
            self.downsample = Conv1x1Linear(cin, cout) if cin != cout else None

        def forward(self, x):
            x1 = self.conv1(x)
            x2 = self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1)) + self.gate(self.conv2c(x1)) + \
                self.gate(self.conv2d(x1))
            x3 = self.conv3(x2)
            return F.relu(x3 + (self.downsample(x) if self.downsample is not None else x))

    def layer(cin, cout, reduce):
        mods = [OSBlock(cin, cout), OSBlock(cout, cout)]
        if reduce:
            mods.append(nn.Sequential(Conv1x1(cout, cout), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*mods)

    class OSNet(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.conv1 = ConvLayer(3, c[0], 7, stride=2, padding=3)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            self.conv2 = layer(c[0], c[1], True)
            self.conv3 = layer(c[1], c[2], True)
            self.conv4 = layer(c[2], c[3], False)
            self.conv5 = Conv1x1(c[3], c[3])
            self.fc = nn.Sequential(nn.Linear(c[3], 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True))

        def forward(self, x):
            x = self.conv5(self.conv4(self.conv3(self.conv2(self.maxpool(self.conv1(x))))))
            return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))

    return OSNet(channels)


class Embedder:
    """OSNet appearance embeddings for player crops (BGR uint8 images of any size)."""

    def __init__(self, variant: str = "x0_25", cache_dir: str | Path | None = None, device: str = "cpu"):
        import torch

        from .weights import cache_dir as default_cache, download

        fname, channels = FILES[variant]
        path = Path(cache_dir or default_cache()) / f"osnet_{variant}_msmt17.pth"
        if not path.exists():
            download(URL.format(fname), path)
        self.net = _build(channels)
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        state = {k.replace("module.", ""): v for k, v in state.items() if not k.startswith(("classifier", "module.classifier"))}
        self.net.load_state_dict(state, strict=True)
        self.net.eval().to(device)
        self.device = device
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def __call__(self, crops: list, batch: int = 64) -> np.ndarray:
        import cv2
        import torch

        out = []
        for i in range(0, len(crops), batch):
            X = np.stack([(cv2.resize(c, (128, 256))[..., ::-1] / 255.0 - self.mean) / self.std
                          for c in crops[i:i + batch]]).astype(np.float32)
            with torch.no_grad():
                f = self.net(torch.from_numpy(X).permute(0, 3, 1, 2).to(self.device)).cpu().numpy()
            out.append(f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9))
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)
