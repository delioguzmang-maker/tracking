"""Verification video and pitch plots.

Top: the broadcast with the calibrated pitch drawn on it (if the lines sit on the
real lines, the calibration is right) and every tracked person labelled with its
identity. Bottom: the 2D pitch, SkillCorner viewer style: filled dots = detected,
hollow = extrapolated, grey polygon = what the camera sees.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import pitch

PANEL_W = 1280


def team_color(team: int, colors) -> tuple:
    if team in (0, 1):
        return tuple(int(c) for c in colors[team])
    return (40, 40, 40)


def _ensure_contrast(bgr):
    """Very dark kits are drawn as dark grey so they stay visible on the pitch."""
    if max(bgr) < 60:
        return (30, 30, 30)
    return bgr


class PitchCanvas:
    def __init__(self, width: int = PANEL_W, height: int = 600, length: float = pitch.LENGTH,
                 wdt: float = pitch.WIDTH, margin: float = 5.0):
        self.W, self.H = width, height
        self.L, self.Wd = length, wdt
        self.s = min(width / (length + 2 * margin), height / (wdt + 2 * margin))
        self.ox = width / 2
        self.oy = height / 2
        self.base = np.zeros((height, width, 3), np.uint8)
        self.base[:] = (60, 130, 60)
        for pl in pitch.polylines(length, wdt):
            if np.abs(pl[:, 2]).max() > 0:
                continue
            cv2.polylines(self.base, [self.px(pl[:, :2])], False, (235, 235, 235), 2, cv2.LINE_AA)

    def px(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2)
        return np.stack([self.ox + xy[:, 0] * self.s, self.oy - xy[:, 1] * self.s], 1).astype(np.int32)

    def draw(self, players: list, ball=None, footprint=None, trails: dict | None = None) -> np.ndarray:
        img = self.base.copy()
        if footprint is not None and len(footprint) >= 3:
            ov = img.copy()
            cv2.fillPoly(ov, [self.px(footprint)], (200, 200, 200))
            img = cv2.addWeighted(ov, 0.25, img, 0.75, 0)
        if trails:
            for tr, col in trails.values():
                if len(tr) > 1:
                    cv2.polylines(img, [self.px(tr)], False, col, 1, cv2.LINE_AA)
        for x, y, pid, col, detected in players:
            c = tuple(self.px([[x, y]])[0])
            if detected:
                cv2.circle(img, c, 9, col, -1, cv2.LINE_AA)
                cv2.circle(img, c, 9, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(img, c, 8, col, 2, cv2.LINE_AA)
            cv2.putText(img, str(pid), (c[0] + 9, c[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        if ball is not None and np.isfinite(ball).all():
            c = tuple(self.px([ball])[0])
            cv2.circle(img, c, 6, (0, 220, 255), -1, cv2.LINE_AA)
            cv2.circle(img, c, 6, (0, 0, 0), 1, cv2.LINE_AA)
        return img


def draw_calibration(frame: np.ndarray, cam, color=(255, 0, 255), thickness: int = 2) -> np.ndarray:
    """Project every pitch marking with ``cam`` onto ``frame``."""
    img = frame
    for pl in pitch.polylines():
        dense = []
        for a, b in zip(pl[:-1], pl[1:]):
            n = max(2, int(np.linalg.norm(b - a) / 0.5))
            t = np.linspace(0, 1, n)[:, None]
            dense.append(a[None] * (1 - t) + b[None] * t)
        P = np.concatenate(dense)
        uv, z = cam.project(P)
        ok = z > 0
        # draw only runs in front of the camera
        runs = np.split(np.arange(len(P)), np.nonzero(np.diff(ok.astype(int)))[0] + 1)
        for r in runs:
            if ok[r[0]] and len(r) > 1:
                q = uv[r]
                if np.abs(q).max() < 1e5:
                    cv2.polylines(img, [q.astype(np.int32)], False, color, thickness, cv2.LINE_AA)
    return img


def render_video(res, path: str | Path, progress: bool = True, max_frames: int | None = None,
                 show_calibration: bool = True) -> str:
    from .analysis import iter_frames

    an = res.analysis
    cfg = res.config
    colors = [_ensure_contrast(c) for c in res.tracking.team_colors]
    ident = {i.pid: i for i in res.tracking.identities}
    canvas = PitchCanvas(length=cfg.pitch_length, wdt=cfg.pitch_width)
    tab = res.table
    by_k = {k: g for k, g in tab.groupby("frame")} if len(tab) else {}
    scale = PANEL_W / an.width
    top_h = int(round(an.height * scale))
    out_size = (PANEL_W, top_h + canvas.H)
    path = str(path)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), an.proc_fps, out_size)
    by_index = {fd.index: i for i, fd in enumerate(an.frames)}
    trails: dict[int, list] = {}
    stride = an.config.get("stride", 1)
    bar = None
    if progress:
        try:
            from tqdm.auto import tqdm
            bar = tqdm(total=len(an.frames), unit="frame", desc="Renderizando verificación")
        except ImportError:
            pass
    n = 0
    for idx, fr in iter_frames(an.video, an.config.get("start_s", 0.0), stride, len(an.frames)):
        i = by_index.get(idx)
        if i is None:
            continue
        fd, cam = an.frames[i], res.cameras[i]
        img = fr.copy()
        if cam is not None and show_calibration:
            draw_calibration(img, cam, (255, 0, 255), max(1, int(round(an.width / 960))))
        p = fd.det.persons()
        asg = res.tracking.assignments[i]
        for j, b in enumerate(p.boxes):
            x1, y1, x2, y2 = b.astype(int)
            pid = asg.get(j)
            if pid is None:
                cv2.rectangle(img, (x1, y1), (x2, y2), (160, 160, 160), 1)
                continue
            idn = ident[pid]
            col = team_color(idn.team, colors)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            lab = f"{pid}" + ("GK" if idn.role == "goalkeeper" else "R" if idn.role == "referee" else "")
            cv2.putText(img, lab, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(img, lab, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
        if res.ball_det[i] and cam is not None:
            uv, z = cam.project(np.array([[*res.ball_pos[i], 0.11]]))
            if z[0] > 0:
                cv2.circle(img, tuple(uv[0].astype(int)), 12, (0, 220, 255), 2, cv2.LINE_AA)
        if cam is None:
            cv2.putText(img, "sin calibracion (plano no utilizable)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                        (0, 0, 255), 3, cv2.LINE_AA)
        top = cv2.resize(img, (PANEL_W, top_h), interpolation=cv2.INTER_AREA)
        # minimap at the nearest 10 fps output frame
        k = int(np.argmin(np.abs(res.out_t - fd.t))) if len(res.out_t) else -1
        players = []
        g = by_k.get(k)
        if g is not None:
            for r in g.itertuples(index=False):
                idn = ident[int(r.player_id)]
                col = team_color(idn.team, colors)
                players.append((r.x, r.y, int(r.player_id), col, bool(r.is_detected)))
                tr = trails.setdefault(int(r.player_id), [])
                tr.append((r.x, r.y))
                del tr[:-20]
        fp = cam.footprint() if cam is not None else None
        if fp is not None:
            fp = np.clip(fp, [-cfg.pitch_length / 2 - 5, -cfg.pitch_width / 2 - 5], [cfg.pitch_length / 2 + 5, cfg.pitch_width / 2 + 5])
        ball = res.ball_pos[i] if res.ball_det[i] else None
        tr_draw = {pid: (np.array(tr), team_color(ident[pid].team, colors)) for pid, tr in trails.items()}
        mini = canvas.draw(players, ball, fp, tr_draw)
        cv2.putText(mini, f"t={fd.t:6.2f}s  frame {idx}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        vw.write(np.vstack([top, mini]))
        n += 1
        if bar is not None:
            bar.update(1)
        if max_frames is not None and n >= max_frames:
            break
    vw.release()
    if bar is not None:
        bar.close()
    return path


def plot_frame(res, k: int, ax=None):
    """Matplotlib pitch plot of output frame ``k`` (for notebooks)."""
    import matplotlib.pyplot as plt

    cfg = res.config
    if ax is None:
        _, ax = plt.subplots(figsize=(10.5, 6.8))
    ax.set_facecolor((0.24, 0.51, 0.24))
    for pl in pitch.polylines(cfg.pitch_length, cfg.pitch_width):
        if np.abs(pl[:, 2]).max() == 0:
            ax.plot(pl[:, 0], pl[:, 1], color="white", lw=1.2)
    colors = res.tracking.team_colors
    ident = {i.pid: i for i in res.tracking.identities}
    g = res.table[res.table["frame"] == k]
    for r in g.itertuples(index=False):
        idn = ident[int(r.player_id)]
        c = np.array(team_color(idn.team, colors))[::-1] / 255.0
        ax.scatter(r.x, r.y, s=120, color=c if r.is_detected else "none", edgecolors=c if not r.is_detected else "white",
                   linewidths=2 if not r.is_detected else 1, zorder=3)
        ax.text(r.x + 0.8, r.y + 0.8, str(r.player_id), color="white", fontsize=8, zorder=4)
    ax.set_xlim(-cfg.pitch_length / 2 - 5, cfg.pitch_length / 2 + 5)
    ax.set_ylim(-cfg.pitch_width / 2 - 5, cfg.pitch_width / 2 + 5)
    ax.set_aspect("equal")
    ax.set_title(f"frame {k}  (relleno = detectado, hueco = extrapolado)")
    return ax
