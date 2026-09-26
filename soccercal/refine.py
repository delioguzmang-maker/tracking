"""Second look at the video once the first result is known (pass 1b).

The first pass runs every network once per frame on the full image. Two things
are then worth a targeted second look, and only where they are missing:

* **Ball**: in a full 720p frame the ball is 5-8 px and the detector misses it.
  In frames where the ball was not found, the detector is run again on a crop
  around where the ball should be (interpolated between the detections before
  and after, projected with the calibrated camera), upscaled ~2.4x by the
  detector: a 7 px ball becomes 17 px. Where there is no nearby detection to
  predict from, the whole frame is searched in overlapping tiles.
* **Shirt numbers**: with identities known, each player's best views across
  the whole clip (largest, unoccluded boxes) are read with OCR, instead of only
  the few largest players on keyframes.

Results are stored in the ``Analysis`` (extra ball boxes, extra number reads),
so they are cached with it and the next ``build`` uses them.
"""
from __future__ import annotations

import numpy as np

from collections import Counter

from .analysis import Analysis, iter_frames
from .detect import BALL, Detections, ball_colour


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def ball_search_windows(an: Analysis, cams: list, ball_pos: np.ndarray, ball_det: np.ndarray, imgsz: int = 1280,
                        max_pred_s: float = 2.0, zoom: float = 2.4, min_dt: float = 0.1,
                        tile_dt: float = 0.3) -> dict[int, list]:
    """{analysed frame: [(x0, y0, x1, y1) crop windows]} for frames with a camera and no ball,
    at most every ``min_dt`` s (the tracker interpolates in between); whole-frame tiles (6x the
    cost) at most every ``tile_dt`` s."""
    t = np.array([fd.t for fd in an.frames])
    S = int(min(imgsz / zoom, an.width, an.height))
    det_idx = np.nonzero(ball_det)[0]
    out, last, last_tiles = {}, -np.inf, -np.inf
    for i, cam in enumerate(cams):
        if cam is None or ball_det[i] or t[i] - last < min_dt - 1e-6:
            continue
        k = np.searchsorted(det_idx, i)
        a = det_idx[k - 1] if k > 0 else None
        b = det_idx[k] if k < len(det_idx) else None
        near_a = a is not None and t[i] - t[a] <= max_pred_s
        near_b = b is not None and t[b] - t[i] <= max_pred_s
        if near_a or near_b:
            if near_a and near_b:
                s = (t[i] - t[a]) / max(t[b] - t[a], 1e-6)
                p = ball_pos[a] * (1 - s) + ball_pos[b] * s
            else:
                p = ball_pos[a] if near_a else ball_pos[b]
            uv, z = cam.project(np.array([[p[0], p[1], 0.11]]))
            if z[0] <= 0 or not np.isfinite(uv).all():
                continue
            cx = int(np.clip(uv[0, 0] - S / 2, 0, an.width - S))
            cy = int(np.clip(uv[0, 1] - S / 2, 0, an.height - S))
            out[i] = [(cx, cy, cx + S, cy + S)]
            last = t[i]
        else:
            if t[i] - last_tiles < tile_dt - 1e-6:
                continue
            last = last_tiles = t[i]
            nx = int(np.ceil((an.width - S) / (0.8 * S))) + 1
            ny = int(np.ceil((an.height - S) / (0.8 * S))) + 1
            xs = np.linspace(0, an.width - S, nx).astype(int)
            ys = np.linspace(0, an.height - S, ny).astype(int)
            out[i] = [(x, y, x + S, y + S) for y in ys for x in xs]
    return out


def refine_ball(video: str, an: Analysis, cams: list, ball_pos: np.ndarray, ball_det: np.ndarray, detector,
                progress: bool = True, batch: int = 8) -> int:
    """Search the ball again in high resolution where it is missing. Adds the new ball boxes
    to the frames' detections. Returns the number of frames where a ball candidate was added."""
    win = ball_search_windows(an, cams, ball_pos, ball_det, imgsz=detector.imgsz)
    if not win:
        return 0
    by_index = {fd.index: i for i, fd in enumerate(an.frames)}
    bar = None
    if progress:
        try:
            from tqdm.auto import tqdm
            bar = tqdm(total=len(win), unit="frame", desc="Buscando el balón (alta resolución)")
        except ImportError:
            pass
    added = 0
    pending = []  # (frame i, frame image, window)

    def flush():
        nonlocal added
        if not pending:
            return
        dets = detector([fr[y0:y1, x0:x1] for _, fr, (x0, y0, x1, y1) in pending])
        new: dict[int, list] = {}
        for (i, fr, (x0, y0, _, _)), d in zip(pending, dets):
            bl = d.balls()
            if len(bl):
                new.setdefault(i, [fr, [], []])
                new[i][1].append(bl.boxes + np.array([x0, y0, x0, y0], np.float32))
                new[i][2].append(bl.scores)
        for i, (fr, bx, sc) in new.items():
            bx, sc = np.concatenate(bx), np.concatenate(sc)
            fd = an.frames[i]
            old = fd.det.balls()
            keep = np.ones(len(bx), bool)
            if len(old):
                keep &= _iou(bx, old.boxes).max(1) < 0.3  # already known
            order = np.argsort(-sc)  # overlapping windows see the same ball twice
            for a_ in order:
                if keep[a_]:
                    dup = _iou(bx[a_:a_ + 1], bx)[0] > 0.3
                    dup[a_] = False
                    keep &= ~dup | (np.arange(len(bx)) == a_)
            if not keep.any():
                continue
            bx, sc = bx[keep], sc[keep]
            fd.det = Detections(np.concatenate([fd.det.boxes, bx]).astype(np.float32),
                                np.concatenate([fd.det.scores, sc]).astype(np.float32),
                                np.concatenate([fd.det.classes, np.full(len(sc), BALL, int)]))
            fd.ball_sv = ball_colour(fr, fd.det.balls().boxes)  # persons keep their order and indices
            added += 1
        pending.clear()

    for idx, fr in iter_frames(an.video, an.config.get("start_s", 0.0), an.stride, len(an.frames), fps=an.fps):
        i = by_index.get(idx)
        if i is None or i not in win:
            continue
        for w in win[i]:
            pending.append((i, fr, w))
        if len(pending) >= batch:
            flush()
        if bar is not None:
            bar.update(1)
    flush()
    if bar is not None:
        bar.close()
    return added


def number_views(an: Analysis, tracking, per_identity: int = 80, min_height_frac: float = 0.055,
                 min_gap: int = 2, max_overlap: float = 0.15, known_votes: int = 6) -> dict[int, list[int]]:
    """{analysed frame: [person detection indices]} to read: each player's best views over the
    whole clip (largest boxes, not cut by the image border, not overlapping another person,
    a few frames apart), skipping boxes already read in the first pass and players whose
    number is already sure (``known_votes`` readings)."""
    meas = tracking.measurements
    H = an.height
    out: dict[int, list[tuple[int, int]]] = {}
    for ident in tracking.identities:
        if ident.role not in ("player", "goalkeeper") or (ident.number is not None and ident.number_votes >= known_votes):
            continue
        cand = []
        for t in ident.tracklets:
            for f, d in zip(t.frames, t.det_idx):
                f, j = int(f), int(meas[f].det_idx[d])
                fd = an.frames[f]
                boxes = fd.det.persons().boxes
                b = boxes[j]
                h = b[3] - b[1]
                if h < min_height_frac * H or b[1] <= 1 or b[3] >= H - 2 or (fd.jersey and j in fd.jersey):
                    continue
                if len(boxes) > 1:
                    others = np.delete(boxes, j, 0)
                    if _iou(b[None], others).max() > max_overlap:
                        continue
                cand.append((h, f, j))
        cand.sort(reverse=True)
        taken: list[int] = []
        for h, f, j in cand:
            if all(abs(f - g) >= min_gap for g in taken):
                out.setdefault(f, []).append((j, len(taken)))  # (detection, rank among this player's views)
                taken.append(f)
                if len(taken) >= per_identity:
                    break
    return out


def read_numbers(an: Analysis, tracking, reader, progress: bool = True, per_identity: int = 80,
                 det_views: int = 40, enough: int = 6) -> int:
    """OCR each player's best views (see ``number_views``); readings are added to the frames'
    ``jersey`` dicts, so the next build votes over them. The (slower) text detector runs on the
    ``det_views`` largest views of each player, the digit-blob reader on all of them; a player
    whose number is already clear (``enough`` agreeing readings, 80 %) is not read further.
    Returns the number of readings."""
    views = number_views(an, tracking, per_identity=per_identity)
    if not views or reader is None or not reader.available:
        return 0
    meas = tracking.measurements
    owner = {}  # (frame, detection) -> identity pid
    got: dict[int, list] = {}
    for ident in tracking.identities:
        for t in ident.tracklets:
            for f, d in zip(t.frames, t.det_idx):
                owner[(int(f), int(meas[f].det_idx[d]))] = ident.pid
            got.setdefault(ident.pid, []).extend(t.jersey_reads)

    def clear(pid):
        c = Counter(n for n, _ in got.get(pid, []))
        if not c:
            return False
        n, k = c.most_common(1)[0]
        return k >= enough and k >= 0.8 * sum(c.values())
    by_index = {fd.index: i for i, fd in enumerate(an.frames)}
    bar = None
    if progress:
        try:
            from tqdm.auto import tqdm
            bar = tqdm(total=sum(len(v) for v in views.values()), unit="recorte", desc="Leyendo dorsales")
        except ImportError:
            pass
    n = 0
    for idx, fr in iter_frames(an.video, an.config.get("start_s", 0.0), an.stride, len(an.frames), fps=an.fps):
        i = by_index.get(idx)
        if i is None or i not in views:
            continue
        fd = an.frames[i]
        boxes = fd.det.persons().boxes
        for j, rank in views[i]:
            pid = owner.get((i, j))
            if pid is not None and clear(pid):
                if bar is not None:
                    bar.update(1)
                continue
            r = reader.read_one(fr, boxes[j], use_det=rank < det_views)
            if r is not None:
                fd.jersey = dict(fd.jersey or {})
                fd.jersey[j] = r
                got.setdefault(pid, []).append(r)
                n += 1
            if bar is not None:
                bar.update(1)
    if bar is not None:
        bar.close()
    return n
