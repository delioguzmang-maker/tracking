"""Train the "Wide Open Spaces" pitch-value network on open tracking data (see ``soccercal/space.py``).

    python tools/open_data.py --cache open_data
    python tools/train_pitch_value.py --cache open_data

As in Fernández & Bornn (2018, section 4): defensive situations every 3 s; target = the summed
influence of the defending players on a 21 x 15 grid, capped at 1; inputs = ball and grid location
(attacking towards +x); one hidden layer, sigmoid activations, mean squared error, Adam. Matches are
split into train / test. Writes ``soccercal/data/pitch_value.npz``.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soccercal.space import value_inputs  # noqa: E402
from soccercal.tactics import influence  # noqa: E402


def situations(match_pkl: Path, every_s: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    d = pickle.load(open(match_pkl, "rb"))
    m = d["m"]
    L, W = m.pitch_length, m.pitch_width
    xs = (np.arange(21) + 0.5) * L / 21 - L / 2
    ys = (np.arange(15) + 0.5) * W / 15 - W / 2
    P = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    tab = m.table.sort_values("frame", kind="stable").reset_index(drop=True)
    f = tab["frame"].values
    b = m.ball
    keep = b[(b["possession_team"] >= 0) & np.isfinite(b["x"]) & np.isfinite(b["y"])]
    X, Y, last = [], [], -np.inf
    for k, r in keep.iterrows():
        if (k - last) < every_s * m.fps:
            continue
        a0, a1 = np.searchsorted(f, k), np.searchsorted(f, k, side="right")
        g = tab.iloc[a0:a1]
        att = int(r["possession_team"])
        dfd = g[g["team"] != att]
        if len(dfd) < 9 or (g["team"] == att).sum() < 9:
            continue
        last = k
        sgn = 1.0 if m.attacks_right(att, int(g["period"].iloc[0])) else -1.0
        pos = dfd[["x", "y"]].values * sgn
        vel = np.nan_to_num(dfd[["vx", "vy"]].values) * sgn
        ball = np.array([r["x"], r["y"]]) * sgn
        D = np.minimum(influence(pos, vel, ball, P).sum(0), 1.0)
        X.append(value_inputs(ball, P, (L, W)))
        Y.append(D)
    return (np.concatenate(X).astype(np.float32), np.concatenate(Y).astype(np.float32)) if X else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="open_data")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "soccercal" / "data" / "pitch_value.npz"))
    a = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    files = sorted((Path(a.cache) / "matches").glob("*.pkl"))
    test = set()
    for src in sorted({p.stem.split("_")[0] for p in files}):  # ~20 % of each source's matches for testing
        ms = [p for p in files if p.stem.startswith(src)]
        rng.shuffle(ms)
        test |= set(ms[:max(1, round(0.2 * len(ms)))])
    cache = Path(a.cache) / "pitch_value_situations.npz"  # building them takes a while: kept
    if cache.exists():
        z = np.load(cache)
        Xtr, Ytr, Xte, Yte = (torch.from_numpy(z[k]) for k in ("Xtr", "Ytr", "Xte", "Yte"))
    else:
        data = {"train": ([], []), "test": ([], [])}
        for p in files:
            X, Y = situations(p)
            if X is None:
                continue
            part = "test" if p in test else "train"
            data[part][0].append(X)
            data[part][1].append(Y)
            print(p.stem, part, len(X) // 315, "situations", flush=True)
        Xtr, Ytr = (torch.from_numpy(np.concatenate(v)) for v in data["train"])
        Xte, Yte = (torch.from_numpy(np.concatenate(v)) for v in data["test"])
        np.savez(cache, Xtr=Xtr.numpy(), Ytr=Ytr.numpy(), Xte=Xte.numpy(), Yte=Yte.numpy())
    net = torch.nn.Sequential(torch.nn.Linear(4, a.hidden), torch.nn.Sigmoid(), torch.nn.Linear(a.hidden, 1), torch.nn.Sigmoid())
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    ckpt, first = Path(a.cache) / "pitch_value_ckpt.pt", 0  # resume after an interruption
    if ckpt.exists():
        c = torch.load(ckpt, weights_only=False)
        net.load_state_dict(c["net"])
        opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"])
        first = c["epoch"]
        print("resuming after epoch", first, flush=True)
    mse = float("nan")
    for ep in range(first, a.epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 1024):
            j = perm[i:i + 1024]
            loss = torch.mean((net(Xtr[j])[:, 0] - Ytr[j]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        with torch.no_grad():
            mse = float(torch.mean((net(Xte)[:, 0] - Yte) ** 2))
        print(f"epoch {ep + 1}: test MSE {mse:.4f}", flush=True)
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep + 1}, ckpt)
    with torch.no_grad():
        mse = float(torch.mean((net(Xte)[:, 0] - Yte) ** 2))
    # baselines: the overall mean, and the mean of every grid cell (ignoring where the ball is)
    cell_mean = Ytr.reshape(-1, 315).mean(0)
    report = {"situations": {"train": len(Xtr) // 315, "test": len(Xte) // 315},
              "test_matches": sorted(p.stem for p in test), "hidden": a.hidden,
              "test_mse": {"constant": float(torch.mean((Ytr.mean() - Yte) ** 2)),
                           "cell_mean_without_ball": float(torch.mean((cell_mean.repeat(len(Yte) // 315) - Yte) ** 2)),
                           "network": mse},
              "test_r2": 1 - mse / float(torch.var(Yte))}
    print(json.dumps(report, indent=1))
    W1, b1, W2, b2 = (t.detach().numpy() for t in (net[0].weight.T, net[0].bias, net[2].weight.T, net[2].bias))
    np.savez(a.out, W1=W1, b1=b1, W2=W2, b2=b2, report=json.dumps(report))
    print("saved", a.out)


if __name__ == "__main__":
    main()
