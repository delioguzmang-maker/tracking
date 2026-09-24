"""Pass 2a: a camera for every analysed frame (or None when the view is unusable).

1. Split the video into *segments*: runs of frames linked by a successful background
   registration and no shot cut.
2. Free calibration (keypoints + dense line refinement) of every keyframe.
3. Tripod position: geometric median of good free calibrations; the main camera's
   position is shared by all its shots, other cameras keep their own.
4. Re-calibrate keyframes with the position fixed (only pan/tilt/roll/zoom left).
5. Registration chain through the segment + fusion with the keyframe calibrations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analysis import Analysis
from .calibrate import calibrate, quality_ok
from .camera import Camera
from .camtrack import fuse_segment, propagate, propagate_back, robust_center
from .config import Config
from .lines import LineMask


@dataclass
class Segment:
    start: int  # analysed-frame index (inclusive)
    end: int  # exclusive
    center: np.ndarray | None = None
    main_camera: bool = False
    n_good_keys: int = 0


def shot_type(fd, width: int, height: int, cfg: Config) -> str:
    """'wide' (usable for tracking) or why not: 'closeup' (a person fills the image: a player,
    the coach, a fan) or 'no_pitch' (crowd, bench, graphics: little grass).

    SkillCorner-style data only comes from wide views of the pitch; everything else must not
    produce positions even if a calibration happened to succeed on it."""
    if fd.grass < cfg.min_grass:
        return "no_pitch"
    p = fd.det.persons()
    if len(p):
        h = (p.boxes[:, 3] - p.boxes[:, 1]) / height
        big = (h > cfg.closeup_person_frac) & (p.scores > 0.4)
        if big.any():
            return "closeup"
    return "wide"


def split_segments(an: Analysis, cfg: Config) -> list[Segment]:
    segs, s = [], 0
    for i in range(1, len(an.frames)):
        fd = an.frames[i]
        if fd.sig_dist > cfg.cut_threshold or not fd.reg.ok:
            segs.append(Segment(s, i))
            s = i
    if an.frames:
        segs.append(Segment(s, len(an.frames)))
    return segs


def solve_cameras(an: Analysis, cfg: Config | None = None, progress: bool = False) -> tuple[list, list[Segment]]:
    cfg = cfg or Config()
    n = len(an.frames)
    cams: list[Camera | None] = [None] * n
    segs = split_segments(an, cfg)
    shots = [shot_type(fd, an.width, an.height, cfg) for fd in an.frames]
    masks: dict[int, LineMask] = {}

    def mask(i):
        if i not in masks and an.frames[i].lines is not None:
            masks[i] = LineMask.unpack(an.frames[i].lines)
        return masks.get(i)

    # --- free calibrations
    free: dict[int, Camera] = {}
    for i, fd in enumerate(an.frames):
        if fd.field is None or len(fd.field.keypoints) < 4 or shots[i] != "wide":
            continue
        c = calibrate(fd.field, line_mask=mask(i))
        if quality_ok(c, min_align=cfg.min_align):
            free[i] = c

    # --- tripod positions
    main_C = None
    if free:
        allc = list(free.values())
        main_C = robust_center(allc, [c.info.get("align", 0.5) ** 2 for c in allc])
    for sg in segs:
        ks = [free[i] for i in range(sg.start, sg.end) if i in free]
        sg.n_good_keys = len(ks)
        if not ks:
            continue
        C = robust_center(ks, [c.info.get("align", 0.5) ** 2 for c in ks])
        if main_C is not None and np.linalg.norm(C - main_C) < 8.0:
            sg.center, sg.main_camera = main_C, True
        elif len(ks) >= 3 and not cfg.main_camera_only:
            sg.center = C
        # a segment with 1-2 calibrations far from the main camera is not trusted

    # --- tripod re-calibration + chain + fusion per segment
    lam = 25.0 / max(1, cfg.keyframe_every)
    max_gap = cfg.max_keyframe_gap_s * an.proc_fps
    for sg in segs:
        if sg.center is None:
            continue
        keys, wts = {}, {}
        for i in range(sg.start, sg.end):
            fd = an.frames[i]
            if fd.field is None or shots[i] != "wide":
                continue
            c = calibrate(fd.field, line_mask=mask(i), fix_center=sg.center, prior=free.get(i))
            if quality_ok(c, min_align=cfg.min_align):
                keys[i - sg.start] = c
                wts[i - sg.start] = max(0.05, c.info.get("align", 0.5)) ** 2
        if not keys:
            continue
        L = sg.end - sg.start
        anchor = max(keys, key=lambda k: keys[k].info.get("align", 0))
        chain: list[Camera | None] = [None] * L
        chain[anchor] = keys[anchor]
        for k in range(anchor + 1, L):
            chain[k] = propagate(chain[k - 1], an.frames[sg.start + k].reg)
        for k in range(anchor - 1, -1, -1):
            chain[k] = propagate_back(chain[k + 1], an.frames[sg.start + k + 1].reg)
        fused = fuse_segment(chain, keys, wts, smooth_lambda=lam)
        kidx = np.array(sorted(keys))
        for k in range(L):
            gap = np.min(np.abs(kidx - k))
            c = fused[k]
            if gap <= max_gap and c.is_plausible() and shots[sg.start + k] == "wide":
                c.info = {"segment": segs.index(sg), "keyframe_gap": int(gap), "main": sg.main_camera,
                          "align": keys[k].info.get("align") if k in keys else None}
                cams[sg.start + k] = c
    for i, s in enumerate(shots):
        an.frames[i].shot = s  # kept for the verification video / summary
    return cams, segs
