"""Thresholds of the appearance model, measured on SoccerNet clips with ground truth.

    python tools/reid_calibrate.py path/to/SNMOT-187 [more clips]

Run it on validation clips only (here: SoccerNet test-split game 11, SNMOT-187/190/193/196), never
on the clips used for the final score (SNMOT-116/126/141, games 7 and 8): a threshold chosen on the
test clips makes the test score optimistic. Crops every 0.2 s (as ``Config.reid_every_s``) from
the ground-truth boxes not overlapped by anybody else, embedded with the soccer OSNet, give:

* the cosine between two views of one player and between two teammates: below ``reid_min_cos`` two
  fragments are never joined (how many same-player pairs that loses, how many teammates it excludes);
* the distance between the mean of 4 views and the next 4 views of one player (no switch) and of a
  player then a teammate (a switch): above ``reid_split`` a track is cut where a teammate was near.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def window_distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.mean(0), b.mean(0)
    return float(1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    from reid_benchmark import load_clip

    from soccercal.reid import Embedder

    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--window", type=int, default=4, help="views averaged on each side (players.split_by_appearance)")
    a = ap.parse_args()
    w = a.window
    emb = Embedder("soccer")
    rng = np.random.default_rng(0)
    same, mate, gen, sw = [], [], [], []
    for d in map(Path, a.clips):
        crops = load_clip(d, min_h=30, per_track=1000)
        F = emb([x["crop"] for x in crops])
        tid = np.array([x["tid"] for x in crops])
        team = np.array([x["team"] for x in crops])
        fr = np.array([x["frame"] for x in crops])
        S = F @ F.T
        far = np.abs(fr[:, None] - fr[None]) >= 25  # at least 1 s apart
        iu = np.triu(np.ones_like(S, bool), 1)
        same += list(S[iu & far & (tid[:, None] == tid[None])])
        mate += list(S[iu & far & (team[:, None] == team[None]) & (tid[:, None] != tid[None])])
        seqs = {t: np.nonzero(tid == t)[0][np.argsort(fr[tid == t])] for t in np.unique(tid)}
        for t, i in seqs.items():
            if len(i) < 2 * w + 2:
                continue
            mates = [u for u in seqs if u != t and team[seqs[u][0]] == team[i[0]] and len(seqs[u]) >= w]
            for k in range(w, len(i) - w, w):
                gen.append(window_distance(F[i[k - w:k]], F[i[k:k + w]]))
                if mates:
                    j = seqs[mates[rng.integers(len(mates))]]
                    s0 = rng.integers(0, len(j) - w + 1)
                    sw.append(window_distance(F[i[k - w:k]], F[j[s0:s0 + w]]))
        print(f"{d.name}: {len(crops)} views of {len(seqs)} players", flush=True)
    same, mate, gen, sw = map(np.array, (same, mate, gen, sw))
    print(f"\ncosine, same player (>= 1 s apart): p1 {np.percentile(same, 1):.2f}  p5 {np.percentile(same, 5):.2f}  "
          f"median {np.median(same):.2f}   ({len(same)} pairs)")
    print(f"cosine, two teammates:              median {np.median(mate):.2f}  p95 {np.percentile(mate, 95):.2f}  "
          f"p99 {np.percentile(mate, 99):.2f}   ({len(mate)} pairs)")
    print("\nreid_min_cos  same-player pairs refused  teammate pairs refused")
    for c in (0.2, 0.3, 0.4, 0.5, 0.6):
        print(f"   {c:.1f}            {np.mean(same < c) * 100:6.2f} %              {np.mean(mate < c) * 100:5.1f} %")
    print(f"\nwindow distance ({w} views | {w} views): one player median {np.median(gen):.3f} p99 {np.percentile(gen, 99):.3f};"
          f" a switch median {np.median(sw):.3f} p10 {np.percentile(sw, 10):.3f}   ({len(gen)} / {len(sw)} windows)")
    print("\nreid_split   one-player windows cut   switches found")
    for thr in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35):
        print(f"   {thr:.2f}          {np.mean(gen > thr) * 100:6.2f} %            {np.mean(sw > thr) * 100:5.1f} %")


if __name__ == "__main__":
    main()
