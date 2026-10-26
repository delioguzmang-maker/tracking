"""Learn where unseen players stand (see ``soccercal/impute.py``) from open tracking data, and measure it.

    python tools/open_data.py --cache open_data
    python tools/train_impute.py --cache open_data

* Goalkeeper network: frames every second where the goalkeeper's position is real (optical data, or
  seen by the camera in SkillCorner's broadcast data); matches split into train / test.
* Formation library: every team's average shape per half (10 outfield places, relative to the
  team's centre and spread, attacking towards +x).
* Test: on held-out matches, one outfield player is hidden for 15 s and put back by matching the
  other nine to the library (and the goalkeeper by the network); errors are compared with simple
  baselines. Writes ``soccercal/data/impute.npz`` (network + library from all matches) and
  ``impute_report.json`` in the cache.
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
from soccercal.impute import gk_features, match_template  # noqa: E402


def team_halves(m):
    """Per (team, period): frames x 10 outfield players (attacking +x), player ids, goalkeeper track."""
    tab = m.table
    out = []
    for (team, per), g in tab.groupby(["team", "period"]):
        sgn = 1.0 if m.attacks_right(int(team), int(per)) else -1.0
        of = g[~g["gk"]]
        top = of["player_id"].value_counts().index[:10]
        if len(top) < 10:
            continue
        piv_x = of[of["player_id"].isin(top)].pivot_table(index="frame", columns="player_id", values="x")
        piv_y = of[of["player_id"].isin(top)].pivot_table(index="frame", columns="player_id", values="y")
        ok = piv_x.notna().all(axis=1) & piv_y.notna().all(axis=1)
        if ok.sum() < 300:  # under 30 s with the same ten players: no usable shape
            continue
        P = np.stack([piv_x[ok].values, piv_y[ok].values], -1) * sgn  # (T, 10, 2)
        det = None
        if "is_detected" in of:
            dd = of[of["player_id"].isin(top)].pivot_table(index="frame", columns="player_id", values="is_detected",
                                                            aggfunc="first")
            det = dd.reindex(index=piv_x[ok].index, columns=piv_x.columns).fillna(False).values.astype(bool)
        out.append({"team": int(team), "period": int(per), "frames": piv_x[ok].index.values, "P": P, "det": det,
                    "sgn": sgn})
    return out


def template(P: np.ndarray) -> np.ndarray:
    """Average normalised places (10, 2) of a (T, 10, 2) team track."""
    c = P.mean(1, keepdims=True)
    s = np.maximum(P.std(1, keepdims=True), 3.0)
    return ((P - c) / s).mean(0)


def gk_samples(m, every: int = 10):
    tab = m.table
    X, Y = [], []
    L, W = m.pitch_length, m.pitch_width
    by_f = dict(tuple(tab.groupby("frame")))
    for k in m.frames[::every]:
        g = by_f.get(k)
        if g is None:
            continue
        per = int(g["period"].iloc[0])
        b = m.ball.loc[k] if k in m.ball.index else None
        for team in (0, 1):
            gk = g[(g["team"] == team) & g["gk"]]
            of = g[(g["team"] == team) & ~g["gk"]]
            if len(gk) != 1 or len(of) < 8 or ("is_detected" in gk and not bool(gk["is_detected"].iloc[0])):
                continue
            sgn = 1.0 if m.attacks_right(team, per) else -1.0  # attacking +x = defending the -x goal
            P = of[["x", "y"]].values * sgn
            if b is None or not np.isfinite([b["x"], b["y"]]).all():
                continue
            ball = np.array([b["x"], b["y"]]) * sgn
            pt = int(b["possession_team"])
            own = 0 if pt < 0 else (1 if pt == team else -1)
            X.append(gk_features(ball, [own], [P[:, 0].mean()], [P[:, 0].min()], (L, W))[0])
            Y.append(gk[["x", "y"]].values[0] * sgn / np.array([L / 2, W / 2]))
    return X, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="open_data")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "soccercal" / "data" / "impute.npz"))
    ap.add_argument("--windows", type=int, default=30)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    files = sorted((Path(a.cache) / "matches").glob("*.pkl"))
    test = set()
    for src in sorted({p.stem.split("_")[0] for p in files}):
        ms = [p for p in files if p.stem.startswith(src)]
        rng.shuffle(ms)
        test |= set(ms[:max(1, round(0.2 * len(ms)))])
    lib = {"train": [], "all": []}
    halves_test, gk = {"train": ([], []), "test": ([], [])}, []
    for p in files:
        d = pickle.load(open(p, "rb"))
        m = d["m"]
        th = team_halves(m)
        for h in th:
            t = template(h["P"])
            lib["all"].append(t)
            if p not in test:
                lib["train"].append(t)
            else:
                gk.append((d["source"], m.pitch_length, m.pitch_width, h))
        X, Y = gk_samples(m)
        part = "test" if p in test else "train"
        halves_test[part][0].extend(X)
        halves_test[part][1].extend(Y)
        print(p.stem, part, len(th), "team halves,", len(X), "goalkeeper samples", flush=True)
    # ------------------------------------------------------------------ goalkeeper network
    Xtr, Ytr = (torch.tensor(np.array(v), dtype=torch.float32) for v in halves_test["train"])
    Xte, Yte = (torch.tensor(np.array(v), dtype=torch.float32) for v in halves_test["test"])
    net = torch.nn.Sequential(torch.nn.Linear(5, 32), torch.nn.Tanh(), torch.nn.Linear(32, 2))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for ep in range(60):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 256):
            j = perm[i:i + 256]
            loss = torch.mean((net(Xtr[j]) - Ytr[j]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
    scale = torch.tensor([52.5, 34.0])
    with torch.no_grad():
        err_net = torch.linalg.norm((net(Xte) - Yte) * scale, dim=1).numpy()
    const = Ytr.mean(0)
    err_const = torch.linalg.norm((const - Yte) * scale, dim=1).numpy()
    # heuristic: on the line goal-ball, 8 m out
    goal = torch.tensor([-1.0, 0.0])
    bl = Xte[:, :2] * scale - goal * scale
    heur = goal * scale + bl / torch.linalg.norm(bl, dim=1, keepdim=True) * 8.0
    err_heur = torch.linalg.norm(heur - Yte * scale, dim=1).numpy()
    rep_gk = {"samples": {"train": len(Xtr), "test": len(Xte)},
              "median_error_m": {"network": float(np.median(err_net)), "fixed_average_position": float(np.median(err_const)),
                                 "8m_out_towards_ball": float(np.median(err_heur))},
              "p90_error_m": {"network": float(np.percentile(err_net, 90)), "fixed_average_position": float(np.percentile(err_const, 90))}}
    print(json.dumps(rep_gk, indent=1), flush=True)
    # ------------------------------------------------------------- outfield, held-out test
    T_train = np.array(lib["train"])
    errs = {"library": [], "team_centre": [], "by_source": {}}
    for src, L, W, h in gk:
        T = len(h["frames"])
        if T < 200:
            continue
        for _ in range(a.windows):
            s0 = rng.integers(0, T - 150)
            P = h["P"][s0:s0 + 150]
            if (np.diff(h["frames"][s0:s0 + 150]) != 1).any():
                continue
            j = rng.integers(0, 10)
            if h["det"] is not None and h["det"][s0:s0 + 150, j].mean() < 0.8:
                continue  # in broadcast data, only test against positions that were really seen
            keep = np.setdiff1d(np.arange(10), [j])
            seen = P[:, keep]
            c = seen.mean(1, keepdims=True)
            s = np.maximum(seen.std(1, keepdims=True), 3.0)
            obs = ((seen - c) / s).mean(0)
            slot, _ = match_template(obs, T_train, 1)
            est = c[:, 0] + slot[0] * s[:, 0]
            e_lib = np.linalg.norm(est - P[:, j], axis=1).mean()
            e_c = np.linalg.norm(c[:, 0] - P[:, j], axis=1).mean()
            errs["library"].append(e_lib)
            errs["team_centre"].append(e_c)
            errs["by_source"].setdefault(src, []).append(e_lib)
    rep_of = {"windows": len(errs["library"]),
              "median_error_m": {"formation_library": float(np.median(errs["library"])),
                                 "team_centre": float(np.median(errs["team_centre"]))},
              "p90_error_m": {"formation_library": float(np.percentile(errs["library"], 90)),
                              "team_centre": float(np.percentile(errs["team_centre"], 90))},
              "median_by_source": {k: float(np.median(v)) for k, v in errs["by_source"].items()}}
    print(json.dumps(rep_of, indent=1), flush=True)
    report = {"test_matches": sorted(p.stem for p in test), "goalkeeper": rep_gk, "outfield": rep_of,
              "library_size": len(lib["all"])}
    json.dump(report, open(Path(a.cache) / "impute_report.json", "w"), indent=1)
    W1, b1, W2, b2 = (t.detach().numpy() for t in (net[0].weight.T, net[0].bias, net[2].weight.T, net[2].bias))
    np.savez(a.out, gk_W1=W1, gk_b1=b1, gk_W2=W2, gk_b2=b2, templates=np.array(lib["all"], np.float32),
             report=json.dumps(report))
    print("saved", a.out)


if __name__ == "__main__":
    main()
