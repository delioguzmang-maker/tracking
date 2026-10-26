"""Can appearance tell two players of the SAME team apart? Measured on SoccerNet tracking clips.

    python tools/reid_benchmark.py path/to/SNMOT-116 path/to/SNMOT-121 ...

Clips: SoccerNet-Tracking 2023 test set (broadcast, 1080p, 30 s, ground-truth identities, team and
shirt number per track; https://huggingface.co/datasets/SoccerNet/SN-Tracking-2023). Crops of
players, at least 1 s apart, are compared within each team: AUC of "same player" against "other
player of the same team", and rank-1 (the most similar crop of a teammate at another moment is the
same player). Descriptors: soccercal's jersey colour histogram (what the tracker used so far) and
OSNet (``soccercal.reid``). Also simulates identity switches inside a track (player A, then player
B of the same team) and measures how well the drop in similarity between the two halves of a
window finds them.
"""
from __future__ import annotations

import argparse
import configparser
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soccercal.detect import DESC_DIM, grass_reference, jersey_descriptor  # noqa: E402
from soccercal.reid import Embedder  # noqa: E402


def load_clip(d: Path, min_h: int = 50, max_iou: float = 0.2, every: int = 5, per_track: int = 30):
    ini = configparser.ConfigParser()
    ini.read(d / "gameinfo.ini")
    roles = {}
    for k, v in ini["Sequence"].items():
        if k.startswith("trackletid_"):
            role, _, num = v.strip().partition(";")
            if "team left" in role or "team right" in role:
                roles[int(k.split("_")[1])] = ("L" if "left" in role else "R", num)
    gt = np.loadtxt(d / "gt" / "gt.txt", delimiter=",")
    frames = sorted(int(p.stem) for p in (d / "img1").glob("*.jpg"))
    crops = []
    for f in frames:
        rows = gt[gt[:, 0] == f]
        boxes = rows[:, 2:6].copy()
        boxes[:, 2:] += boxes[:, :2]
        img = None
        for r, b in zip(rows, boxes):
            tid = int(r[1])
            if tid not in roles or b[3] - b[1] < min_h:
                continue
            others = np.delete(boxes, np.nonzero((boxes == b).all(1))[0][0], 0)
            x1, y1 = np.maximum(b[:2], others[:, :2]).T if len(others) else (np.zeros(0), np.zeros(0))
            x2, y2 = np.minimum(b[2:], others[:, 2:]).T if len(others) else (np.zeros(0), np.zeros(0))
            inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
            if len(inter) and (inter / ((b[2] - b[0]) * (b[3] - b[1]))).max() > max_iou:
                continue  # overlapped by somebody else: not a clean view
            if img is None:
                img = cv2.imread(str(d / "img1" / f"{f:06d}.jpg"))
                grass = grass_reference(img)
            bi = b.astype(int)
            col = jersey_descriptor(img, b, grass)
            crops.append({"frame": f, "tid": tid, "team": roles[tid][0], "crop": img[bi[1]:bi[3], bi[0]:bi[2]].copy(),
                          "colour": col if col is not None else np.zeros(DESC_DIM)})
    # at most ``per_track`` crops per track, spread over the clip
    out = []
    for tid in {c["tid"] for c in crops}:
        cs = [c for c in crops if c["tid"] == tid]
        idx = np.linspace(0, len(cs) - 1, min(per_track, len(cs))).round().astype(int)
        out += [cs[i] for i in sorted(set(idx))]
    return out


def auc(pos, neg):
    s = np.r_[pos, neg]
    r = np.empty(len(s))
    r[np.argsort(s)] = np.arange(1, len(s) + 1)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / max(len(pos) * len(neg), 1))


def evaluate(crops, F, min_dt: int = 25):
    """Within-team AUC and rank-1 with descriptor matrix F (n, d), rows L2-normalised."""
    S = F @ F.T
    fr = np.array([c["frame"] for c in crops])
    tid = np.array([c["tid"] for c in crops])
    team = np.array([c["team"] for c in crops])
    far = np.abs(fr[:, None] - fr[None]) >= min_dt
    same_team = team[:, None] == team[None]
    iu = np.triu(np.ones_like(S, bool), 1)
    pos = S[iu & far & same_team & (tid[:, None] == tid[None])]
    neg = S[iu & far & same_team & (tid[:, None] != tid[None])]
    hits, n = 0, 0
    for i in range(len(crops)):
        cand = far[i] & same_team[i]
        if not cand.any() or not (cand & (tid == tid[i])).any():
            continue
        j = np.nonzero(cand)[0][np.argmax(S[i, cand])]
        hits += tid[j] == tid[i]
        n += 1
    return {"auc_same_team": round(auc(pos, neg), 3), "rank1_same_team": round(hits / max(n, 1), 3),
            "pairs": [int(len(pos)), int(len(neg))]}


def switches(crops, F, w: int = 4, rng=None):
    """Synthetic switches: a track of player A whose last ``w`` views are player B (same team), against
    tracks of A only. Score = 1 - similarity between the mean of the first and the last w views."""
    rng = rng or np.random.default_rng(0)
    tid = np.array([c["tid"] for c in crops])
    team = np.array([c["team"] for c in crops])
    fr = np.array([c["frame"] for c in crops])
    sw, ok = [], []
    for a in np.unique(tid):
        ia = np.nonzero(tid == a)[0]
        ia = ia[np.argsort(fr[ia])]
        if len(ia) < 2 * w:
            continue
        mates = [b for b in np.unique(tid[team == team[ia[0]]]) if b != a and (tid == b).sum() >= w]
        first = F[ia[:w]].mean(0)
        ok.append(1 - first @ F[ia[-w:]].mean(0) / np.linalg.norm(first) / np.linalg.norm(F[ia[-w:]].mean(0)))
        for b in rng.choice(mates, min(3, len(mates)), replace=False) if mates else []:
            ib = np.nonzero(tid == b)[0][:w]
            m = F[ib].mean(0)
            sw.append(1 - first @ m / np.linalg.norm(first) / np.linalg.norm(m))
    return {"auc_switch_detection": round(auc(np.array(sw), np.array(ok)), 3), "n": [len(sw), len(ok)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--variants", default="x0_25,x1_0")
    a = ap.parse_args()
    emb = {v: Embedder(v) for v in a.variants.split(",")}
    report = {}
    allc = {k: [] for k in ["colour", *emb]}
    for d in map(Path, a.clips):
        crops = load_clip(d)
        Fs = {"colour": np.array([c["colour"] for c in crops], float)}
        Fs["colour"] /= np.maximum(np.linalg.norm(Fs["colour"], axis=1, keepdims=True), 1e-9)
        for v, e in emb.items():
            Fs[v] = e([c["crop"] for c in crops])
        report[d.name] = {k: {**evaluate(crops, F), **switches(crops, F)} for k, F in Fs.items()}
        print(d.name, len(crops), "crops", json.dumps(report[d.name]), flush=True)
        for k in allc:
            allc[k].append(report[d.name][k])
    report["mean"] = {k: {m: round(float(np.mean([r[m] for r in v])), 3)
                          for m in ("auc_same_team", "rank1_same_team", "auc_switch_detection")} for k, v in allc.items()}
    print(json.dumps(report["mean"], indent=1))


if __name__ == "__main__":
    main()
