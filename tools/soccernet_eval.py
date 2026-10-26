"""soccercal on SoccerNet tracking clips, scored against their ground truth.

    python tools/soccernet_eval.py path/to/SNMOT-116 [more clips] --work out_dir

Each clip (SoccerNet-Tracking 2023: 30 s of broadcast video at 25 fps, 1080p, every player boxed
with a persistent identity, his team and shirt number) is turned into a video and run through
``soccercal.run`` (analysis cached in ``--work``). Our person boxes are matched to the ground-truth
boxes frame by frame (IoU >= 0.5), and:

* IDF1 / ID precision / ID recall (Ristani et al. 2016): identities matched over the whole clip;
* ID switches: times a ground-truth player's matched identity changes;
* fragments: identities per ground-truth player (1 = never lost);
* team accuracy (our two teams mapped to left / right) and shirt numbers read right / wrong.

Only players and goalkeepers are scored (referees, staff and the ball are not players).
"""
from __future__ import annotations

import argparse
import configparser
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def clip_to_video(d: Path, out: Path, fps: float = 25.0) -> Path:
    if out.exists():
        return out
    frames = sorted((d / "img1").glob("*.jpg"))
    h, w = cv2.imread(str(frames[0])).shape[:2]
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.imread(str(f)))
    vw.release()
    return out


def ground_truth(d: Path):
    ini = configparser.ConfigParser()
    ini.read(d / "gameinfo.ini")
    info = {}
    for k, v in ini["Sequence"].items():
        if k.startswith("trackletid_"):
            role, _, num = v.strip().partition(";")
            if "team left" in role or "team right" in role:
                info[int(k.split("_")[1])] = ("left" if "left" in role else "right",
                                              int(num) if num.strip().isdigit() else None, "goalkeeper" in role)
    gt = np.loadtxt(d / "gt" / "gt.txt", delimiter=",")
    gt = gt[np.isin(gt[:, 1].astype(int), list(info))]
    return gt, info


def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])  # noqa: E731
    return inter / (ar(a)[:, None] + ar(b)[None] - inter + 1e-9)


def score(res, gt, info, min_iou: float = 0.5) -> dict:
    an = res.analysis
    ident = {i.pid: i for i in res.tracking.identities}
    pairs = []  # (frame, gt id, our pid)
    n_gt = 0
    n_ours = 0
    for i, fd in enumerate(an.frames):
        f = fd.index + 1  # SoccerNet frames are 1-based
        g = gt[gt[:, 0] == f]
        n_gt += len(g)
        boxes = fd.det.persons().boxes
        asg = res.tracking.assignments[i] if res.cameras[i] is not None else {}
        ours = [(j, pid) for j, pid in asg.items() if ident[pid].role in ("player", "goalkeeper")]
        n_ours += len(ours)
        if not len(g) or not ours:
            continue
        gb = g[:, 2:6].copy()
        gb[:, 2:] += gb[:, :2]
        ob = boxes[[j for j, _ in ours]]
        M = iou(gb, ob)
        r, c = linear_sum_assignment(-M)
        for a, b in zip(r, c):
            if M[a, b] >= min_iou:
                pairs.append((f, int(g[a, 1]), ours[b][1]))
    # IDF1: one-to-one map between ground-truth and predicted identities maximising matched frames
    cnt = Counter((g, p) for _, g, p in pairs)
    gids = sorted({g for g, _ in cnt})
    pids = sorted({p for _, p in cnt})
    C = np.zeros((len(gids), len(pids)))
    for (g, p), n in cnt.items():
        C[gids.index(g), pids.index(p)] = n
    r, c = linear_sum_assignment(-C)
    idtp = C[r, c].sum()
    idf1 = 2 * idtp / max(n_gt + n_ours, 1)
    # switches and fragments per ground-truth player
    by_g = defaultdict(list)
    for f, g, p in sorted(pairs):
        by_g[g].append(p)
    switches = sum(sum(1 for a, b in zip(v[:-1], v[1:]) if a != b) for v in by_g.values())
    frags = np.mean([len(set(v)) for v in by_g.values()]) if by_g else np.nan
    # teams: our team 0/1 against left/right (best of the two mappings)
    tm = Counter((info[g][0], ident[p].team) for _, g, p in pairs)
    a1 = tm[("left", 0)] + tm[("right", 1)]
    a2 = tm[("left", 1)] + tm[("right", 0)]
    team_acc = max(a1, a2) / max(sum(tm.values()), 1)
    # shirt numbers: each identity with a number against the ground-truth player it covers most
    right = wrong = 0
    owner = defaultdict(Counter)
    for _, g, p in pairs:
        owner[p][g] += 1
    for p, c_ in owner.items():
        num = ident[p].number
        if num is None:
            continue
        g = c_.most_common(1)[0][0]
        if info[g][1] is None:
            continue
        right += num == info[g][1]
        wrong += num != info[g][1]
    known = sum(1 for g in by_g if info[g][1] is not None)
    return {"IDF1": round(idf1, 3), "ID_precision": round(idtp / max(n_ours, 1), 3), "ID_recall": round(idtp / max(n_gt, 1), 3),
            "ID_switches": int(switches), "identities_per_player": round(float(frags), 2), "players_in_gt": len(by_g),
            "our_identities": len({p for _, _, p in pairs}), "team_accuracy": round(team_acc, 3),
            "numbers_right": int(right), "numbers_wrong": int(wrong), "players_with_number_in_gt": known}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--work", default="soccernet_eval")
    a = ap.parse_args()
    import soccercal
    from soccercal.config import Config

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    report = {}
    for d in map(Path, a.clips):
        video = clip_to_video(d, work / f"{d.name}.mp4")
        res = soccercal.run(video, work / d.name, Config(players_per_team=11), render=False, progress=False)
        gt, info = ground_truth(d)
        report[d.name] = score(res, gt, info)
        print(d.name, json.dumps(report[d.name]), flush=True)
    keys = ["IDF1", "ID_precision", "ID_recall", "identities_per_player", "team_accuracy"]
    report["mean"] = {k: round(float(np.mean([r[k] for r in report.values()])), 3) for k in keys}
    report["total"] = {k: int(sum(r[k] for n, r in report.items() if n != "mean"))
                       for k in ("ID_switches", "numbers_right", "numbers_wrong")}
    print(json.dumps(report, indent=1))
    json.dump(report, open(work / "report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
