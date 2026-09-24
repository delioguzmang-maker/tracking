"""Calibration quality on a real clip without ground truth.

Metrics (computed against white line pixels found by a top-hat filter, independent
of the pipeline's own line mask):

* ``align@3px`` / ``align@1px``: fraction of visible pitch-marking points whose
  projection lands within 3 / 1 px (1080p) of a white line pixel. Players and worn
  paint hide lines, so ~0.8 is excellent;
* jitter: mean absolute second difference of pan / tilt / roll (degrees) and log
  focal length between consecutive frames. A real camera moves smoothly, so this is
  essentially calibration noise.

Variants: (A) per-frame keypoint-only calibration (what NBJW / the sportlight
baseline do), (B) + dense line refinement, (C) + tripod (fixed camera position),
(D) the full pipeline: keyframes every 5 frames + registration + fusion.

    python tools/calibration_benchmark.py            # sample clip
    python tools/calibration_benchmark.py video.mp4 --max-seconds 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from soccercal import pitch  # noqa: E402
from soccercal.analysis import Analysis, analyze, iter_frames  # noqa: E402
from soccercal.calibrate import calibrate, quality_ok  # noqa: E402
from soccercal.cameras import solve_cameras  # noqa: E402
from soccercal.camtrack import robust_center  # noqa: E402
from soccercal.config import Config  # noqa: E402
from soccercal.lines import LineMask  # noqa: E402
from soccercal.weights import get_sample_video  # noqa: E402


def white_mask(fr):
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    return (th > 25).astype(np.uint8)


def _model_points(step=0.25):
    pts = []
    for pl in pitch.polylines():
        if np.abs(pl[:, 2]).max() > 0:
            continue
        for a, b in zip(pl[:-1], pl[1:]):
            t = np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / step)))[:, None]
            pts.append(a[None] * (1 - t) + b[None] * t)
    return np.concatenate(pts)


MP = _model_points()


def align(cam, m, tol):
    s = cam.width / 1920.0
    t = max(1, int(round(tol * s)))
    dm = cv2.dilate(m, np.ones((2 * t + 1, 2 * t + 1), np.uint8))
    uv, z = cam.project(MP)
    ok = (z > 0) & (uv[:, 0] >= 5) & (uv[:, 0] < m.shape[1] - 5) & (uv[:, 1] >= 5) & (uv[:, 1] < m.shape[0] - 5)
    q = uv[ok].astype(int)
    return float(dm[q[:, 1], q[:, 0]].mean()) if len(q) else np.nan


def jitter(cams):
    ang = np.array([c.pan_tilt_roll() for c in cams])
    lf = np.log([c.f for c in cams])
    return np.abs(np.diff(ang, 2, axis=0)).mean(0), float(np.abs(np.diff(lf, 2)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=None)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache", default="calib_bench_analysis.pkl.gz")
    args = ap.parse_args()
    video = args.video or str(get_sample_video())
    cfg1 = Config(device=args.device, keyframe_every=1, max_seconds=args.max_seconds)
    if Path(args.cache).exists():
        an = Analysis.load(args.cache)
    else:
        an = analyze(video, cfg1)
        an.save(args.cache)
    frames = [fr for _, fr in iter_frames(video, 0.0, 1, len(an.frames), fps=an.fps)]
    wm = [white_mask(f) for f in frames]
    n = len(frames)
    rows = {}
    # A / B: per-frame, free
    A = [calibrate(fd.field) for fd in an.frames]
    B = [calibrate(fd.field, line_mask=LineMask.unpack(fd.lines)) for fd in an.frames]
    good = [c for c in B if quality_ok(c)]
    C0 = robust_center(good, [c.info["align"] ** 2 for c in good])
    Cc = [calibrate(fd.field, line_mask=LineMask.unpack(fd.lines), fix_center=C0, prior=b)
          for fd, b in zip(an.frames, B)]
    # D: full pipeline using only every 5th frame's network output
    an5 = Analysis(an.video, an.fps, an.width, an.height, an.n_video_frames, dict(an.config), [])
    for i, fd in enumerate(an.frames):
        fd5 = type(fd)(**{k: getattr(fd, k) for k in fd.__dataclass_fields__})
        if i % 5:
            fd5.field, fd5.lines = None, None
        an5.frames.append(fd5)
    D, _ = solve_cameras(an5, Config(keyframe_every=5))
    for name, cams in (("A per-frame keypoints (baseline)", A), ("B + dense line refinement", B),
                       ("C + tripod (fixed position)", Cc), ("D full: keyframes/5 + registration + fusion", D)):
        ok = [i for i, c in enumerate(cams) if c is not None]
        a3 = np.nanmean([align(cams[i], wm[i], 3) for i in ok])
        a1 = np.nanmean([align(cams[i], wm[i], 1) for i in ok])
        runs = [cams[i] for i in ok]
        (jp, jt, jr), jf = jitter(runs) if len(runs) > 3 else ((np.nan,) * 3, np.nan)
        rows[name] = (len(ok), a3, a1, jp, jt, jr, jf)
    print(f"\n{n} frames of {Path(video).name}\n")
    print(f"{'variant':46s} {'calibrated':>10s} {'align@3px':>9s} {'align@1px':>9s} {'jit pan':>8s} {'jit tilt':>8s} {'jit roll':>8s} {'jit zoom':>8s}")
    for k, (m, a3, a1, jp, jt, jr, jf) in rows.items():
        print(f"{k:46s} {m:>6d}/{n:<3d} {a3:9.3f} {a1:9.3f} {jp:8.4f} {jt:8.4f} {jr:8.4f} {jf:8.4f}")


if __name__ == "__main__":
    main()
