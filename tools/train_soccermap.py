"""Train SoccerMap's pass-success surface on open tracking + event data (see ``tools/open_data.py``).

    python tools/open_data.py --cache open_data
    python tools/train_soccermap.py --cache open_data

Matches (not passes) are split into train / validation / test, so the test score is on games the
network never saw. Baselines as in the paper: Naive (the average completion rate) and a logistic
model on handcrafted features. Writes ``soccercal/data/soccermap_pass_success.pt`` (float16 weights
+ the measured scores) and ``soccermap_report.json`` in the cache.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soccercal.soccermap import SoccerMapNet  # noqa: E402
from soccercal.tactics import Snapshot, soccermap_tensor  # noqa: E402

GRID = (68, 104)


# ------------------------------------------------------------------------------------------ samples
def samples_from(match_pkl: Path) -> list:
    """One record per pass with a usable snapshot: both teams with >= 7 players, origin and
    destination known."""
    d = pickle.load(open(match_pkl, "rb"))
    m, passes = d["m"], d["passes"]
    tab = m.table.sort_values("frame", kind="stable").reset_index(drop=True)
    f = tab["frame"].values
    c = {k: tab[k].values for k in ("x", "y", "vx", "vy", "team", "gk", "player_id", "period")}
    out = []
    for p in passes.itertuples(index=False):
        if not np.isfinite([p.x0, p.y0, p.x1, p.y1]).all():
            continue
        a, b = np.searchsorted(f, p.frame), np.searchsorted(f, p.frame, side="right")
        team = c["team"][a:b].astype(int)
        if b - a < 14 or min((team == 0).sum(), (team == 1).sum()) < 7:
            continue
        per = int(c["period"][a])
        out.append({"pos": np.stack([c["x"][a:b], c["y"][a:b]], 1).astype(np.float32),
                    "vel": np.nan_to_num(np.stack([c["vx"][a:b], c["vy"][a:b]], 1)).astype(np.float32),
                    "team": team, "gk": c["gk"][a:b].astype(bool), "pid": c["player_id"][a:b],
                    "ball": np.array([p.x0, p.y0], np.float32), "dest": np.array([p.x1, p.y1], np.float32),
                    "passer": int(p.passer), "att": int(p.team),
                    "right": (m.attacks_right(0, per), m.attacks_right(1, per)),
                    "pitch": (float(m.pitch_length), float(m.pitch_width)), "y": int(p.success),
                    "source": d["source"], "match": d["match"]})
    return out


def snapshot_of(r) -> Snapshot:
    n = len(r["team"])
    return Snapshot(0, 1, r["pos"].astype(float), r["vel"].astype(float), r["team"], r["gk"], np.ones(n, bool),
                    r["pid"], r["ball"].astype(float), r["att"], r["passer"], None, r["right"], r["pitch"])


def dest_cell(r) -> tuple[int, int]:
    """Destination cell in the tensor's orientation (attacking team towards +x)."""
    L, W = r["pitch"]
    x, y = r["dest"]
    if not r["right"][r["att"]]:
        x, y = -x, -y
    ny, nx = GRID
    return int(np.clip((y + W / 2) / W * ny, 0, ny - 1)), int(np.clip((x + L / 2) / L * nx, 0, nx - 1))


class PassData(torch.utils.data.Dataset):
    def __init__(self, recs, flip: bool):
        self.recs, self.flip = recs, flip

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        X = soccermap_tensor(snapshot_of(r), team=r["att"], dims=GRID)
        iy, ix = dest_cell(r)
        if self.flip and np.random.rand() < 0.5:  # mirror across the pitch's long axis
            X = X[:, ::-1].copy()
            X[[2, 5, 11]] *= -1  # vy, and the signed angle between velocities
            iy = GRID[0] - 1 - iy
        return torch.from_numpy(X), iy, ix, float(r["y"])


# ----------------------------------------------------------------------------------------- metrics
def metrics(p: np.ndarray, y: np.ndarray, bins: int = 10) -> dict:
    """Log-loss, Brier, the paper's ECE (10 bins) and AUC."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    k = np.minimum((p * bins).astype(int), bins - 1)
    ece = float(sum(abs(y[k == b].mean() - p[k == b].mean()) * (k == b).mean() for b in range(bins) if (k == b).any()))
    r = np.empty(len(p))
    r[np.argsort(p)] = np.arange(1, len(p) + 1)
    npos, nneg = y.sum(), len(y) - y.sum()
    auc = float((r[y == 1].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1))
    return {"log_loss": round(ll, 4), "brier": round(float(np.mean((p - y) ** 2)), 4), "ece": round(ece, 4),
            "auc": round(auc, 4), "n": int(len(y))}


def handcrafted(r) -> np.ndarray:
    """The paper's logistic-baseline features: origin and destination, distance, angles to goal,
    opponents near the destination and along the pass line (attacking towards +x)."""
    s = 1.0 if r["right"][r["att"]] else -1.0
    L, W = r["pitch"]
    o, d = r["ball"] * s, r["dest"] * s
    opp = r["pos"][r["team"] != r["att"]] * s
    mate = r["pos"][(r["team"] == r["att"]) & (r["pid"] != r["passer"])] * s
    seg = d - o
    t = np.clip(((opp - o) @ seg) / max(seg @ seg, 1e-6), 0, 1)
    line = np.linalg.norm(opp - (o + t[:, None] * seg), axis=1).min()
    goal = np.array([L / 2, 0.0])
    ang = lambda q: abs(np.arctan2(goal[1] - q[1], goal[0] - q[0]))  # noqa: E731
    dd_opp = np.linalg.norm(opp - d, axis=1)
    dd_mate = np.linalg.norm(mate - d, axis=1) if len(mate) else np.array([15.0])
    return np.array([o[0] / L, o[1] / W, d[0] / L, d[1] / W, np.linalg.norm(seg) / 30, ang(o), ang(d),
                     min(line, 10) / 10, min(dd_opp.min(), 15) / 15, min(dd_mate.min(), 15) / 15,
                     (dd_opp < 5).sum() / 3, (dd_mate < 5).sum() / 3, abs(d[1]) / W], np.float32)


def fit_logistic(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = torch.tensor((Xtr - mu) / sd)
    w = torch.zeros(A.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=500)
    yt = torch.tensor(ytr, dtype=torch.float32)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(A @ w + b, yt) + 1e-4 * (w * w).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return torch.sigmoid(torch.tensor((Xte - mu) / sd) @ w + b).numpy()


# ------------------------------------------------------------------------------------------ training
def predict(net, recs, batch=64):
    dl = torch.utils.data.DataLoader(PassData(recs, flip=False), batch_size=batch, num_workers=2)
    out = []
    net.eval()
    with torch.no_grad():
        for X, iy, ix, _ in dl:
            z = net(X.to(memory_format=torch.channels_last))[:, 0]
            out.append(torch.sigmoid(z[torch.arange(len(iy)), iy, ix]).numpy())
    return np.concatenate(out)


def split_matches(recs, seed: int) -> dict:
    """~70 % train, 15 % validation, 15 % test of the matches of each source."""
    rng = np.random.default_rng(seed)
    split = {}
    for src in sorted({r["source"] for r in recs}):
        ms = sorted({r["match"] for r in recs if r["source"] == src})
        rng.shuffle(ms)
        n_te = max(1, int(round(0.15 * len(ms))))
        n_va = max(1, int(round(0.15 * len(ms)))) if len(ms) > 2 else 0
        for k, mm in enumerate(ms):
            split[(src, mm)] = "test" if k < n_te else "val" if k < n_te + n_va else "train"
    part = {s: [r for r in recs if split[(r["source"], r["match"])] == s] for s in ("train", "val", "test")}
    return part


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="open_data")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "soccercal" / "data" / "soccermap_pass_success.pt"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    root = Path(a.cache)
    cache = root / "soccermap_samples.pkl"
    if cache.exists():
        recs = pickle.load(open(cache, "rb"))
    else:
        recs = []
        for p in sorted((root / "matches").glob("*.pkl")):
            r = samples_from(p)
            print(p.stem, len(r), "passes", flush=True)
            recs += r
        pickle.dump(recs, open(cache, "wb"))
    part = split_matches(recs, a.seed)
    for s, rr in part.items():
        print(s, len(rr), "passes,", round(float(np.mean([r["y"] for r in rr])), 3), "completed,",
              sorted({f"{r['source']}:{r['match']}" for r in rr}), flush=True)
    torch.set_num_threads(a.threads)
    net = SoccerMapNet(13).to(memory_format=torch.channels_last)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    dl = torch.utils.data.DataLoader(PassData(part["train"], flip=True), batch_size=a.batch, shuffle=True, num_workers=2,
                                     drop_last=True, persistent_workers=True)
    yv = np.array([r["y"] for r in part["val"]], float)
    best, best_state, bad, history = np.inf, None, 0, []
    for ep in range(a.epochs):
        net.train()
        t0, tot, n = time.time(), 0.0, 0
        for X, iy, ix, y in dl:
            z = net(X.to(memory_format=torch.channels_last))[:, 0]
            loss = F.binary_cross_entropy_with_logits(z[torch.arange(len(iy)), iy, ix], y.float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot, n = tot + float(loss) * len(y), n + len(y)
        mv = metrics(predict(net, part["val"]), yv)
        history.append({"epoch": ep + 1, "train_log_loss": round(tot / n, 4), **{f"val_{k}": v for k, v in mv.items()}})
        print(f"epoch {ep + 1}: train {tot / n:.4f}, validation {mv} ({time.time() - t0:.0f} s)", flush=True)
        if mv["log_loss"] < best - 1e-3:
            best, bad = mv["log_loss"], 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            torch.save({"model": {k: v.half() for k, v in best_state.items()}, "report": {"partial": history}}, a.out)
        else:
            bad += 1
            if bad >= a.patience:
                break
    net.load_state_dict(best_state)
    yt = np.array([r["y"] for r in part["test"]], float)
    p_sm = predict(net, part["test"])
    ytr = np.array([r["y"] for r in part["train"]], float)
    Htr = np.stack([handcrafted(r) for r in part["train"]])
    Hte = np.stack([handcrafted(r) for r in part["test"]])
    src = np.array([r["source"] for r in part["test"]])
    report = {
        "data": {s: {"passes": len(rr), "completed": round(float(np.mean([r["y"] for r in rr])), 3),
                     "matches": sorted({f"{r['source']}:{r['match']}" for r in rr})} for s, rr in part.items()},
        "test": {"naive": metrics(np.full(len(yt), ytr.mean()), yt), "logistic": metrics(fit_logistic(Htr, ytr, Hte), yt),
                 "soccermap": metrics(p_sm, yt)},
        "test_by_source": {s: metrics(p_sm[src == s], yt[src == s]) for s in sorted(set(src))},
        "parameters": int(n_par), "history": history,
    }
    print(json.dumps(report["test"], indent=1), json.dumps(report["test_by_source"], indent=1), flush=True)
    json.dump(report, open(root / "soccermap_report.json", "w"), indent=1)
    torch.save({"model": {k: v.half() for k, v in net.state_dict().items()}, "report": report}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
