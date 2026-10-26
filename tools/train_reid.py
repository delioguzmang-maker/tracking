"""Fine-tune OSNet to tell apart players of the SAME team (SoccerNet tracking ground truth).

    python tools/train_reid.py --train path/to/SNMOT-train-clips --test SNMOT-116 SNMOT-121 ...

Starts from the published OSNet x0.25 (MSMT17) weights. Every batch takes 8 players of one team in one
clip x 4 views each, so the batch-hard triplet loss works on teammates (same kit: what the tracker
must separate), plus an identity cross-entropy over all training players. Test: within-team rank-1
and switch-detection AUC on held-out clips (``tools/reid_benchmark.py``), before and after.
Writes ``soccercal/data/osnet_x0_25_soccer.pt``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reid_benchmark import evaluate, load_clip, switches  # noqa: E402
from soccercal.reid import Embedder  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_tensor(crop, train: bool, rng) -> np.ndarray:
    img = cv2.resize(crop, (128, 256))
    if train:
        if rng.random() < 0.5:
            img = img[:, ::-1]
        pad = np.pad(img, ((10, 10), (10, 10), (0, 0)), mode="edge")
        y, x = rng.integers(0, 21, 2)
        img = pad[y:y + 256, x:x + 128]
        img = np.clip(img.astype(np.float32) * rng.uniform(0.8, 1.2), 0, 255)
    return ((img[..., ::-1] / 255.0 - MEAN) / STD).astype(np.float32).transpose(2, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="folder with SNMOT-* training clips (gt, gameinfo, img1)")
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "soccercal" / "data" / "osnet_x0_25_soccer.pt"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    clips = sorted(Path(a.train).glob("SNMOT-*"))
    data, groups = [], {}
    for c in clips:
        crops = load_clip(c, every=1, per_track=24)
        for cr in crops:
            key = (c.name, cr["tid"])
            groups.setdefault((c.name, cr["team"]), {}).setdefault(key, []).append(len(data))
            data.append(cr["crop"])
        print(c.name, len(crops), "crops", flush=True)
    ids = sorted({k for g in groups.values() for k in g})
    label = {k: i for i, k in enumerate(ids)}
    lab = np.zeros(len(data), int)
    for g in groups.values():
        for k, idx in g.items():
            lab[idx] = label[k]
    teams = [g for g in groups.values() if sum(len(v) >= 4 for v in g.values()) >= 4]
    print(len(data), "crops,", len(ids), "players,", len(teams), "team-clips", flush=True)
    emb = Embedder("x0_25")
    net = emb.net
    head = torch.nn.Linear(512, len(ids))
    opt = torch.optim.AdamW(list(net.parameters()) + list(head.parameters()), lr=3e-4, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)

    def test():
        out = {}
        for d in a.test:
            crops = load_clip(Path(d))
            Fm = emb([c["crop"] for c in crops])
            out[Path(d).name] = {**evaluate(crops, Fm), **switches(crops, Fm)}
        return {m: round(float(np.mean([r[m] for r in out.values()])), 3)
                for m in ("auc_same_team", "rank1_same_team", "auc_switch_detection")}
    before = test()
    print("before", before, flush=True)
    t0 = time.time()
    for step in range(a.steps):
        net.train()
        g = teams[rng.integers(len(teams))]
        players = [k for k, v in g.items() if len(v) >= 4]
        pick = rng.choice(len(players), min(8, len(players)), replace=False)
        idx = np.concatenate([rng.choice(g[players[p]], 4, replace=False) for p in pick])
        X = torch.from_numpy(np.stack([to_tensor(data[i], True, rng) for i in idx]))
        y = torch.from_numpy(lab[idx])
        f = net(X)
        ce = F.cross_entropy(head(f), y, label_smoothing=0.1)
        fn = F.normalize(f, dim=1)
        D = 1 - fn @ fn.T
        same = y[:, None] == y[None]
        hard_pos = (D * same).max(1).values
        hard_neg = (D + 2.0 * same).min(1).values
        tri = F.relu(hard_pos - hard_neg + 0.3).mean()
        loss = ce + tri
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % 100 == 0:
            print(f"step {step + 1}: ce {float(ce):.3f} triplet {float(tri):.3f} ({time.time() - t0:.0f} s)", flush=True)
    net.eval()
    after = test()
    print("after", after, flush=True)
    torch.save({"model": {k: v.half() for k, v in net.state_dict().items()},
                "report": {"before": before, "after": after, "train_clips": len(clips), "players": len(ids),
                           "crops": len(data), "steps": a.steps}}, a.out)
    print(json.dumps({"before": before, "after": after}), "saved", a.out)


if __name__ == "__main__":
    main()
