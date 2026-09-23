"""Tracking benchmark with ground truth: real player trajectories, simulated broadcast.

Uses Metrica Sports' public sample game (real 22-player trajectories, 25 fps) and a
simulated main camera that follows the ball with pan/zoom. Detections are the true
feet projected into the image with pixel noise, random misses, occlusions (a player
hidden behind a nearer one is usually not detected), false positives and noisy jersey
descriptors. The tracker, re-identification and gap filling then run exactly as on a
real video, and everything is compared to the truth:

* IDF1 / identity purity / identities per real player (re-identification quality),
* position error of detected rows and of extrapolated (off-screen) rows.

It does not test the neural networks (those need labelled video); it tests
everything after detection + calibration, which is where identities are made.

    python tools/synthetic_benchmark.py --minutes 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from soccercal.analysis import Analysis, FrameData  # noqa: E402
from soccercal.camera import Camera  # noqa: E402
from soccercal.camtrack import Registration  # noqa: E402
from soccercal.config import Config  # noqa: E402
from soccercal.detect import DESC_DIM, Detections  # noqa: E402
from soccercal.players import track_players  # noqa: E402
from soccercal.timeline import sample_identity  # noqa: E402
from soccercal.weights import cache_dir, download  # noqa: E402

METRICA = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_1/Sample_Game_1_RawTrackingData_{}_Team.csv"
W, H = 1920, 1080


def load_metrica():
    d = cache_dir() / "metrica"
    teams = {}
    for team in ("Home", "Away"):
        p = d / f"{team}.csv"
        if not p.exists():
            download(METRICA.format(team), p)
        df = pd.read_csv(p, skiprows=2)
        cols = df.columns
        t = df["Time [s]"].values
        pl = {}
        for i, c in enumerate(cols):
            if c.startswith("Player"):
                x, y = df.iloc[:, i].values.astype(float), df.iloc[:, i + 1].values.astype(float)
                pl[f"{team[0]}{c[6:]}"] = np.stack([(x - 0.5) * 105.0, -(y - 0.5) * 68.0], 1)
            if c == "Ball":
                ball = np.stack([(df.iloc[:, i].values - 0.5) * 105.0, -(df.iloc[:, i + 1].values - 0.5) * 68.0], 1)
        teams[team] = pl
    return t, teams["Home"], teams["Away"], ball


def look_at(C, target, f):
    fwd = np.asarray(target, float) - C
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    return Camera(f=f, R=np.stack([right, np.cross(fwd, right), fwd]), C=np.asarray(C, float), width=W, height=H)


def team_protos(rng):
    protos = []
    for _ in range(4):  # team A, team B, GK A, GK B
        h = np.zeros(DESC_DIM)
        h[rng.choice(DESC_DIM, 4, replace=False)] = rng.uniform(0.5, 1.0, 4)
        protos.append(h / h.sum())
    return protos


def noisy_desc(proto, rng, noise=0.35):
    h = proto * (1 - noise) + noise * rng.dirichlet(np.full(DESC_DIM, 0.3))
    v = np.sqrt(h / h.sum())
    return np.clip(np.round(v * 255), 0, 255).astype(np.uint8)


def simulate(start_s=600.0, minutes=3.0, seed=0, fps=25.0, miss=0.05, fp_rate=0.3, px_noise=1.0,
             calib_noise_deg=0.0):
    rng = np.random.default_rng(seed)
    t, home, away, ball = load_metrica()
    i0 = int(np.searchsorted(t, start_s))
    n = int(minutes * 60 * fps)
    sl = slice(i0, i0 + n)
    players = {**{k: v[sl] for k, v in home.items()}, **{k: v[sl] for k, v in away.items()}}
    ball = ball[sl]
    # goalkeepers: the player closest to his goal on average
    gk = set()
    for team in ("H", "A"):
        ks = [k for k in players if k[0] == team and np.isfinite(players[k]).all(1).mean() > 0.5]
        gk.add(min(ks, key=lambda k: -abs(np.nanmean(players[k][:, 0]))))
    protos = team_protos(rng)
    proto_of = {k: protos[(0 if k[0] == "H" else 1) + (2 if k in gk else 0)] for k in players}
    # camera follows the (smoothed) ball, zooms between 35 and 55 m of view width
    C = np.array([0.0, -50.0, 16.0])
    aim = np.zeros((n, 2))
    b = np.nan_to_num(ball[0], nan=0.0)
    for k in range(n):
        if np.isfinite(ball[k]).all():
            b = b + (ball[k] - b) * (1 / (0.8 * fps))
        aim[k] = b
    zoom = 45 + 10 * np.sin(np.arange(n) / fps / 20.0)
    frames, cams, det_gt = [], [], []
    for k in range(n):
        tgt = np.array([np.clip(aim[k, 0], -40, 40), np.clip(aim[k, 1], -20, 20), 0.0])
        dist = np.linalg.norm(tgt - C)
        cam = look_at(C, tgt, W * dist / zoom[k])
        est = cam
        if calib_noise_deg > 0:  # slowly varying calibration error
            from scipy.spatial.transform import Rotation
            e = calib_noise_deg * np.array([np.sin(k / 40.0), np.cos(k / 55.0), np.sin(k / 70.0)])
            est = cam.copy(R=Rotation.from_rotvec(np.radians(e)).as_matrix() @ cam.R)
        est.info = {"segment": 0}
        boxes, gts, descs, depth = [], [], [], []
        for pid, traj in players.items():
            p = traj[k]
            if not np.isfinite(p).all():
                continue
            uv, z = cam.ground_to_image(p)
            h = cam.pixel_height(p, 1.8)[0] * rng.uniform(0.93, 1.05)
            w = 0.38 * h
            u, v = uv[0]
            if not (w / 2 < u < W - w / 2 and h < v < H):
                continue
            boxes.append([u - w / 2, v - h, u + w / 2, v])
            gts.append(pid)
            descs.append(noisy_desc(proto_of[pid], rng))
            depth.append(z[0])
        boxes = np.array(boxes).reshape(-1, 4)
        keep = np.ones(len(boxes), bool)
        sc = rng.uniform(0.5, 0.95, len(boxes))
        order = np.argsort(depth)
        for a_i, a in enumerate(order):  # occlusion by nearer players
            for bb in order[:a_i]:
                ix = max(0, min(boxes[a, 2], boxes[bb, 2]) - max(boxes[a, 0], boxes[bb, 0]))
                iy = max(0, min(boxes[a, 3], boxes[bb, 3]) - max(boxes[a, 1], boxes[bb, 1]))
                cover = ix * iy / ((boxes[a, 2] - boxes[a, 0]) * (boxes[a, 3] - boxes[a, 1]))
                if cover > 0.6 and rng.random() < 0.8:
                    keep[a] = False
                elif cover > 0.3:
                    sc[a] = min(sc[a], rng.uniform(0.2, 0.5))
        keep &= rng.random(len(boxes)) > miss
        # pixel noise on the box (feet localisation)
        bh = boxes[:, 3] - boxes[:, 1]
        du = rng.normal(0, 0.04 * bh * 0.38 + px_noise, len(boxes))
        dv = rng.normal(0, 0.03 * bh + px_noise, len(boxes))
        boxes = boxes + np.stack([du, dv, du, dv], 1)
        boxes, sc = boxes[keep], sc[keep]
        gts = [g for g, kk in zip(gts, keep) if kk]
        descs = [d for d, kk in zip(descs, keep) if kk]
        # false positives on the visible pitch
        for _ in range(rng.poisson(fp_rate)):
            u, v = rng.uniform(100, W - 100), rng.uniform(H * 0.3, H - 10)
            g, ok = cam.image_to_ground(np.array([[u, v]]))
            if not ok[0]:
                continue
            h = cam.pixel_height(g[0], 1.8)[0]
            boxes = np.vstack([boxes, [u - 0.19 * h, v - h, u + 0.19 * h, v]])
            sc = np.r_[sc, rng.uniform(0.1, 0.45)]
            gts.append(None)
            descs.append(noisy_desc(rng.dirichlet(np.full(DESC_DIM, 0.3)), rng, 0.0))
        det = Detections(boxes.astype(np.float32), sc.astype(np.float32), np.zeros(len(sc), int))
        desc = np.array(descs, np.uint8).reshape(-1, DESC_DIM)
        frames.append(FrameData(k, k / fps, 0.0, 0.6, Registration(np.zeros((20, 2)), np.zeros((20, 2)), 0.3),
                                det, desc, np.zeros((len(sc), 3), np.uint8)))
        cams.append(est)
        det_gt.append(gts)
    an = Analysis("synthetic", fps, W, H, n, Config().to_dict(), frames)
    return an, cams, {"players": players, "det_gt": det_gt, "gk": gk, "fps": fps}


def evaluate(an, cams, gt, cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    trk = track_players(an, cams, cfg)
    det_gt = gt["det_gt"]
    # --- detection-level identity metrics
    pairs = {}
    n_gt_dets = sum(sum(g is not None for g in gs) for gs in det_gt)
    n_pred = 0
    for i, asg in enumerate(trk.assignments):
        for j, pid in asg.items():
            n_pred += 1
            g = det_gt[i][j]
            if g is not None:
                pairs[(pid, g)] = pairs.get((pid, g), 0) + 1
    pids = sorted({p for p, _ in pairs})
    gids = sorted({g for _, g in pairs})
    M = np.zeros((len(pids), len(gids)))
    for (p, g), c in pairs.items():
        M[pids.index(p), gids.index(g)] = c
    r, c = linear_sum_assignment(-M)
    idtp = M[r, c].sum()
    match = {pids[a]: gids[b] for a, b in zip(r, c)}
    purity = M.max(1).sum() / max(M.sum(), 1)
    per_gt = (M > 0.05 * M.sum(0, keepdims=True)).sum(0)
    # --- positions at 10 fps
    fps = gt["fps"]
    frame_t = np.array([fd.t for fd in an.frames])
    out_t = np.arange(frame_t[0], frame_t[-1], 1 / cfg.out_fps)
    err_det, err_ext = [], []
    for ident in trk.identities:
        g = match.get(ident.pid)
        s = sample_identity(ident, frame_t, out_t, True, cfg.max_extrapolate_s, motion=trk.motion)
        if g is None:
            continue
        truth = gt["players"][g][np.clip(np.round(out_t * fps).astype(int), 0, len(frame_t) - 1)]
        ok = np.isfinite(s.pos).all(1) & np.isfinite(truth).all(1)
        e = np.linalg.norm(s.pos - truth, axis=1)
        err_det += e[ok & s.detected].tolist()
        err_ext += e[ok & ~s.detected].tolist()
    return {
        "gt_players": len(gids),
        "identities": len(trk.identities),
        "identities_per_player_mean": round(float(per_gt.mean()), 2),
        "IDF1": round(float(2 * idtp / (n_gt_dets + n_pred)), 3),
        "identity_purity": round(float(purity), 3),
        "pos_err_detected_median_m": round(float(np.median(err_det)), 3) if err_det else None,
        "pos_err_detected_p90_m": round(float(np.percentile(err_det, 90)), 3) if err_det else None,
        "pos_err_extrapolated_median_m": round(float(np.median(err_ext)), 2) if err_ext else None,
        "pos_err_extrapolated_p90_m": round(float(np.percentile(err_ext, 90)), 2) if err_ext else None,
        "n_rows_detected": len(err_det),
        "n_rows_extrapolated": len(err_ext),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--start", type=float, default=600.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--calib-noise-deg", type=float, default=0.0)
    ap.add_argument("--miss", type=float, default=0.05)
    args = ap.parse_args()
    rows = []
    for s in range(args.seeds):
        an, cams, gt = simulate(args.start + 300 * s, args.minutes, seed=s, miss=args.miss,
                                calib_noise_deg=args.calib_noise_deg)
        m = evaluate(an, cams, gt)
        m["seed"] = s
        print(json.dumps(m))
        rows.append(m)
    df = pd.DataFrame(rows)
    print(df.drop(columns=["seed"]).mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
