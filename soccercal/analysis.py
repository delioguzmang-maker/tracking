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
from .detect import DESC_DIM, BALL, PERSON, Detections, PlayerDetector, grass_reference, jersey_descriptor
from .field_model import FieldDetector, FieldObservation
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
    def proc_fps(self) -> float:
        return self.fps / self.config.get("stride", 1)

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
            color[i] = np.median(crop.reshape(-1, 3), 0).astype(np.uint8)
    return desc, color


def iter_frames(path: str, start_s: float, stride: int, max_frames: int | None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se puede abrir el vídeo: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start = int(round(start_s * fps))
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    idx = start
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if (idx - start) % stride == 0:
            yield idx, fr
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
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    max_frames = None
    if cfg.max_seconds is not None:
        max_frames = int(np.ceil(cfg.max_seconds * fps / cfg.stride))
    n_expected = (max_frames or int(np.ceil(max(0, n_total - cfg.start_s * fps) / cfg.stride)))

    detector = detector or PlayerDetector(cfg.det_model, cfg.device, cfg.det_imgsz, cfg.det_conf)
    field_detector = field_detector or FieldDetector(cfg.device)
    reg = Registrar(W, H)
    out = Analysis(str(video), fps, W, H, n_total, cfg.to_dict())

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
        frames = [fr for _, fr in buf]
        dets = detector(frames)
        recs, is_key = [], []
        for (idx, fr), det in zip(buf, dets):
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
            recs.append(FrameData(idx, idx / fps, sd, grass_fraction(fr), r, det, desc, color))
            is_key.append(key)
        keys = [i for i, k in enumerate(is_key) if k]
        if keys:
            obs = field_detector([frames[i] for i in keys])
            for i, o in zip(keys, obs):
                recs[i].field = o
                recs[i].lines = LineMask.from_frame(frames[i], dets[i].persons().boxes).pack()
        out.frames.extend(recs)
        if bar is not None:
            bar.update(len(recs))

    for idx, fr in iter_frames(video, cfg.start_s, cfg.stride, max_frames):
        buf.append((idx, fr))
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
