"""Person and ball detection (Ultralytics YOLO, COCO classes) + appearance descriptors."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .device import pick_device

PERSON, BALL = 0, 32  # COCO ids


@dataclass
class Detections:
    boxes: np.ndarray  # (N,4) xyxy pixels
    scores: np.ndarray  # (N,)
    classes: np.ndarray  # (N,) 0 person, 32 ball

    def persons(self) -> "Detections":
        m = self.classes == PERSON
        return Detections(self.boxes[m], self.scores[m], self.classes[m])

    def balls(self) -> "Detections":
        m = self.classes == BALL
        return Detections(self.boxes[m], self.scores[m], self.classes[m])

    def __len__(self):
        return len(self.scores)

    @staticmethod
    def empty() -> "Detections":
        return Detections(np.zeros((0, 4)), np.zeros(0), np.zeros(0, int))


PERSON_NAMES = {"person", "player", "goalkeeper", "referee"}
BALL_NAMES = {"sports ball", "ball", "football"}


class PlayerDetector:
    """YOLO on the full frame. ``model`` can be any Ultralytics checkpoint: COCO models
    (``person`` / ``sports ball``) or a football-specific one whose class names include
    ``player`` / ``goalkeeper`` / ``referee`` / ``ball`` (much better at the ball)."""

    def __init__(self, model: str = "yolo11m.pt", device: str | None = None, imgsz: int = 1280,
                 conf: float = 0.10, person_class: int = PERSON, ball_class: int | None = BALL, half: bool | None = None,
                 ball_conf: float = 0.03):
        from ultralytics import YOLO

        self.device = pick_device(device)
        self.model = YOLO(model)
        names = {int(k): str(v).lower() for k, v in getattr(self.model, "names", {}).items()}
        pers = [k for k, v in names.items() if v in PERSON_NAMES]
        balls = [k for k, v in names.items() if v in BALL_NAMES]
        self.person_classes = pers or [person_class]
        self.ball_classes = balls or ([ball_class] if ball_class is not None else [])
        self.imgsz = imgsz
        self.conf = conf
        self.ball_conf = ball_conf
        self.person_class = person_class
        self.ball_class = ball_class
        self.half = (self.device == "cuda") if half is None else half

    def _precision_kw(self) -> dict:
        if not self.half:
            return {}
        try:  # Ultralytics >= 8.4 renamed ``half`` to ``quantize``
            from ultralytics.cfg import DEFAULT_CFG_DICT

            if "quantize" in DEFAULT_CFG_DICT:
                return {"quantize": 16}
        except ImportError:
            pass
        return {"half": True}

    def __call__(self, frames_bgr: list[np.ndarray]) -> list[Detections]:
        classes = self.person_classes + self.ball_classes
        lo = min(self.conf, self.ball_conf) if self.ball_classes else self.conf
        res = self.model.predict(frames_bgr, imgsz=self.imgsz, conf=lo, classes=classes, device=self.device,
                                 verbose=False, max_det=150, **self._precision_kw())
        out = []
        for r in res:
            b = r.boxes
            cls = b.cls.cpu().numpy().astype(int)
            sc = b.conf.cpu().numpy().astype(np.float32)
            cls = np.where(np.isin(cls, self.person_classes), PERSON, BALL)
            keep = np.where(cls == PERSON, sc >= self.conf, sc >= self.ball_conf)
            out.append(Detections(b.xyxy.cpu().numpy().astype(np.float32)[keep], sc[keep], cls[keep]))
        return out


# ------------------------------------------------------------ appearance
N_CHROMA, N_BRIGHT = 6, 3
DESC_DIM = N_CHROMA * N_CHROMA * N_BRIGHT


def grass_reference(frame_bgr: np.ndarray) -> np.ndarray:
    """Median BGR of grass pixels in the frame (the photometric reference)."""
    small = cv2.resize(frame_bgr, (320, 180), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    g = (hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 50) & (hsv[..., 2] > 30)
    if g.sum() < 100:
        return np.array([60.0, 120.0, 60.0])
    return np.median(small[g].astype(np.float64), 0)


def jersey_descriptor(frame_bgr: np.ndarray, box: np.ndarray, grass_bgr: np.ndarray) -> np.ndarray | None:
    """Colour histogram of the shirt, normalised by the grass colour.

    Shirt = upper body crop (15-50 % of the box height, central 60 %). Pixels that look
    like grass are dropped. Colours are divided by the grass colour (a white-balance /
    shade correction), then binned by log-chromaticity (log r/g, log b/g) and by brightness
    relative to the grass. Returns a Hellinger-embedded histogram (unit L2 norm) or None.
    """
    x1, y1, x2, y2 = box
    h, w = y2 - y1, x2 - x1
    if h < 12 or w < 4:
        return None
    H, W = frame_bgr.shape[:2]
    cx1, cx2 = int(max(0, x1 + 0.2 * w)), int(min(W, x2 - 0.2 * w))
    cy1, cy2 = int(max(0, y1 + 0.15 * h)), int(min(H, y1 + 0.5 * h))
    if cx2 - cx1 < 2 or cy2 - cy1 < 3:
        return None
    crop = frame_bgr[cy1:cy2, cx1:cx2].reshape(-1, 3).astype(np.float64)
    hsv = cv2.cvtColor(frame_bgr[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2HSV).reshape(-1, 3)
    grassy = (hsv[:, 0] > 30) & (hsv[:, 0] < 90) & (hsv[:, 1] > 60) & (hsv[:, 2] > 30)
    px = crop[~grassy]
    if len(px) < 8:
        return None
    n = (px + 1.0) / (grass_bgr + 1.0)  # b, g, r ratios to grass
    lb, lg, lr = np.log(n[:, 0]), np.log(n[:, 1]), np.log(n[:, 2])
    c1 = np.clip((lr - lg + 2.0) / 4.0, 0, 0.999)  # red vs green
    c2 = np.clip((lb - lg + 2.0) / 4.0, 0, 0.999)  # blue vs green
    br = np.clip((np.log(n.mean(1)) + 2.0) / 4.0, 0, 0.999)  # brightness vs grass
    idx = ((c1 * N_CHROMA).astype(int) * N_CHROMA + (c2 * N_CHROMA).astype(int)) * N_BRIGHT + (br * N_BRIGHT).astype(int)
    hist = np.bincount(idx, minlength=DESC_DIM).astype(np.float64)
    hist = np.sqrt(hist / hist.sum())
    return hist.astype(np.float32)


def ball_colour(frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """(k, 3) uint8 per ball candidate: median saturation, bright-pixel value, and the share of
    grass (x255) in the upper half of a ring around it. A ball on the pitch has grass above
    it (or a player's legs); a white dot on an advertising board or in the crowd does not."""
    out = np.zeros((len(boxes), 3), np.uint8)
    H, W = frame.shape[:2]
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cx, cy, r = (x1 + x2) / 2, (y1 + y2) / 2, max(1.5, 0.3 * min(x2 - x1, y2 - y1))
        a, b = int(max(0, cx - r)), int(min(W, cx + r + 1))
        c, d = int(max(0, cy - r)), int(min(H, cy + r + 1))
        if b > a and d > c:
            hsv = cv2.cvtColor(frame[c:d, a:b], cv2.COLOR_BGR2HSV).reshape(-1, 3)
            out[i, :2] = (np.median(hsv[:, 1]), np.percentile(hsv[:, 2], 90))
        s = max(x2 - x1, y2 - y1)
        r_in, r_out = 0.7 * s, 1.6 * s + 3
        a, b = int(max(0, cx - r_out)), int(min(W, cx + r_out + 1))
        c, d = int(max(0, cy - r_out)), int(min(H, cy + 1))
        if b > a and d > c:
            yy, xx = np.mgrid[c:d, a:b]
            rr = np.hypot(xx - cx, yy - cy)
            m = (rr >= r_in) & (rr <= r_out)
            hsv = cv2.cvtColor(frame[c:d, a:b], cv2.COLOR_BGR2HSV)
            g = (hsv[..., 0] > 28) & (hsv[..., 0] < 95) & (hsv[..., 1] > 35) & (hsv[..., 2] > 25)
            out[i, 2] = int(round(255 * g[m].mean())) if m.any() else 0
    return out
