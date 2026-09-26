"""Pass 1: the only step that runs neural networks over the video.

Everything the later stages need is stored compactly per analysed frame, so
calibration / tracking / export can be re-run with other settings in seconds
without touching the video again (``Analysis.save`` / ``Analysis.load``).
"""
from __future__ import annotations

import gzip
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .camtrack import Registrar, Registration, frame_signature, grass_fraction, signature_distance
from .config import Config
from .detect import DESC_DIM, BALL, PERSON, Detections, PlayerDetector, ball_colour, grass_reference, jersey_descriptor
from .field_model import FieldDetector, FieldObservation
from .jersey import JerseyReader
from .lines import LineMask


@dataclass
class FrameData:
    index: int  # frame number in the video
    t: float  # seconds from the video start
    sig_dist: float  # signature distance to the previous analysed frame
    grass: float  # grass fraction of the image
    reg: Registration  # background motion from the previous analysed frame
    det: Detections
    desc: np.ndarray  # (n_person, DESC_DIM) uint8, jersey colour histograms (0 row = unknown)
    color: np.ndarray  # (n_person, 3) uint8 mean shirt BGR
    field: FieldObservation | None = None  # pitch keypoints (keyframes only)
    lines: tuple | None = None  # packed LineMask (keyframes only)
    jersey: dict | None = None  # {person detection index: (shirt number, confidence)} (keyframes only)
    shot: str = "unknown"  # set by cameras.solve_cameras: "wide", "closeup" or "no_pitch"
    ball_sv: np.ndarray | None = None  # (n_ball, 2) colour of each ball candidate (saturation, brightness)

    @property
    def keyframe(self) -> bool:
        return self.field is not None


@dataclass
class Analysis:
    video: str
    fps: float
    width: int
    height: int
    n_video_frames: int
    config: dict
    frames: list[FrameData] = field(default_factory=list)

    @property
    def stride(self) -> int:
        return int(self.config.get("stride_used") or self.config.get("stride") or 1)

    @property
    def proc_fps(self) -> float:
        """Analysed frames per second (from the real timestamps: robust to variable frame rate)."""
        if len(self.frames) > 2:
            dt = np.median(np.diff([f.t for f in self.frames]))
            if dt > 0:
                return float(1.0 / dt)
        return self.fps / self.stride

    def save(self, path: str | Path) -> None:
        with gzip.open(path, "wb", compresslevel=3) as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | Path) -> "Analysis":
        with gzip.open(path, "rb") as f:
            return pickle.load(f)


def _person_features(frame: np.ndarray, det: Detections, grass_bgr: np.ndarray):
    p = det.persons()
    desc = np.zeros((len(p), DESC_DIM), np.uint8)
    color = np.zeros((len(p), 3), np.uint8)
    for i, b in enumerate(p.boxes):
        d = jersey_descriptor(frame, b, grass_bgr)
        if d is not None:
            desc[i] = np.clip(np.round(d * 255), 0, 255).astype(np.uint8)
        x1, y1, x2, y2 = b.astype(int)
        h, w = y2 - y1, x2 - x1
        crop = frame[max(0, y1 + int(0.15 * h)):max(0, y1 + int(0.5 * h)), max(0, x1 + int(0.2 * w)):max(0, x2 - int(0.2 * w))]
        if crop.size:
            px = crop.reshape(-1, 3)
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
            shirt = ~((hsv[:, 0] > 30) & (hsv[:, 0] < 90) & (hsv[:, 1] > 60))  # drop grass pixels
            color[i] = np.median(px[shirt] if shirt.sum() >= 5 else px, 0).astype(np.uint8)
    return desc, color


ANALYSIS_VERSION = 4  # bump when the first (neural network) pass changes what it stores


def auto_stride(fps: float, target: float = 25.0) -> int:
    return max(1, int(round(fps / target)))


def probe_fps(path: str | Path, n: int = 90) -> float:
    """Real frame rate from the timestamps of the first frames. The header can lie (trimmed or
    re-muxed files, screen recordings: a 59.94 fps clip announced as 52.2 fps)."""
    cap = cv2.VideoCapture(str(path))
    header = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ts = []
    for _ in range(n):
        if not cap.grab():
            break
        ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    cap.release()
    dt = np.diff(ts)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) >= 5:
        fps = 1.0 / float(np.median(dt))
        if 5.0 <= fps <= 240.0:
            return fps
    return header


def iter_frames(path: str, start_s: float, stride: int, max_frames: int | None, with_time: bool = False,
                fps: float | None = None):
    """Yields (frame index, frame) — or (index, seconds, frame) with ``with_time``. The time is
    the container timestamp, correct for variable-frame-rate files (phone / screen recordings)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se puede abrir el vídeo: {path}")
    fps = fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
    start = int(round(start_s * fps))
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    idx = start
    n = 0
    last_t = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if not np.isfinite(t) or (last_t is not None and t <= last_t) or (t == 0 and idx > 0):
            t = idx / fps if last_t is None else max(idx / fps, last_t + 1e-3)
        last_t = t
        if (idx - start) % stride == 0:
            yield (idx, t, fr) if with_time else (idx, fr)
            n += 1
            if max_frames is not None and n >= max_frames:
                break
        idx += 1
    cap.release()


def analyze(video: str | Path, cfg: Config | None = None, progress: bool = True,
            detector: PlayerDetector | None = None, field_detector: FieldDetector | None = None) -> Analysis:
    cfg = cfg or Config()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se puede abrir el vídeo: {video}")
    header_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    fps = probe_fps(video)  # n_total (header) is only used for the progress bar
    stride = cfg.stride or auto_stride(fps)
    max_frames = None
    if cfg.max_seconds is not None:
        max_frames = int(np.ceil(cfg.max_seconds * fps / stride))
    n_expected = (max_frames or int(np.ceil(max(0, n_total - cfg.start_s * fps) / stride)))

    detector = detector or PlayerDetector(cfg.det_model, cfg.device, cfg.det_imgsz, cfg.det_conf,
                                          ball_conf=cfg.ball_conf)
    field_detector = field_detector or FieldDetector(cfg.device)
    jersey = JerseyReader(max_crops=cfg.jersey_crops) if cfg.jersey_ocr else None
    reg = Registrar(W, H)
    out = Analysis(str(video), fps, W, H, n_total, cfg.to_dict())
    out.config["stride_used"] = stride
    out.config["analysis_version"] = ANALYSIS_VERSION
    out.config["header_fps"] = header_fps
    out.config["jersey_ocr_available"] = bool(jersey and jersey.available)

    bar = None
    if progress:
        try:
            from tqdm.auto import tqdm
            bar = tqdm(total=n_expected, unit="frame", desc="Analizando vídeo")
        except ImportError:
            bar = None
    t0 = time.time()
    prev_sig = None
    since_key = 10 ** 9
    buf = []

    def flush(buf):
        nonlocal prev_sig, since_key
        frames = [fr for _, _, fr in buf]
        dets = detector(frames)
        recs, is_key = [], []
        for (idx, t, fr), det in zip(buf, dets):
            sig = frame_signature(fr)
            sd = 1.0 if prev_sig is None else signature_distance(prev_sig, sig)
            prev_sig = sig
            if sd > cfg.cut_threshold:
                reg.reset()
            r = reg.step(fr, det.persons().boxes)
            key = since_key + 1 >= cfg.keyframe_every or sd > cfg.cut_threshold or not r.ok
            since_key = 0 if key else since_key + 1
            g = grass_reference(fr)
            desc, color = _person_features(fr, det, g)
            recs.append(FrameData(idx, t, sd, grass_fraction(fr), r, det, desc, color))
            recs[-1].ball_sv = ball_colour(fr, det.balls().boxes)
            is_key.append(key)
        keys = [i for i, k in enumerate(is_key) if k]
        if keys:
            obs = field_detector([frames[i] for i in keys])
            for i, o in zip(keys, obs):
                recs[i].field = o
                p = dets[i].persons()
                recs[i].lines = LineMask.from_frame(frames[i], p.boxes).pack()
                if jersey is not None and jersey.available and len(o.keypoints) >= 4:  # only on pitch views
                    recs[i].jersey = jersey.read(frames[i], p.boxes, p.scores)
        out.frames.extend(recs)
        if bar is not None:
            bar.update(len(recs))

    for idx, t, fr in iter_frames(video, cfg.start_s, stride, max_frames, with_time=True, fps=fps):
        buf.append((idx, t, fr))
        if len(buf) >= cfg.batch:
            flush(buf)
            buf = []
    if buf:
        flush(buf)
    if bar is not None:
        bar.close()
    out.config["analysis_seconds"] = time.time() - t0
    return out


__all__ = ["Analysis", "FrameData", "analyze", "PERSON", "BALL"]
