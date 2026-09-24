"""Shirt numbers (dorsales) with a small bundled OCR model, voted per identity.

On a wide broadcast shot a number is only readable when the player is near the
camera and shows his back, so any single frame reads few numbers and sometimes
misreads (17 -> 12). Reading a few of the largest players on every keyframe and
voting over everything an identity accumulated is what makes it usable:
identities get their real shirt number, and two fragments of the same player
(same team, same number, never on screen at the same time) are merged.

Uses ``rapidocr_onnxruntime`` (PP-OCRv4 models shipped inside the pip wheel, no
download). If it is not installed, numbers are simply not read.
"""
from __future__ import annotations

from collections import Counter

import cv2
import numpy as np


class JerseyReader:
    def __init__(self, min_height_frac: float = 0.08, max_crops: int = 3, min_score: float = 0.6):
        self.min_height_frac = min_height_frac
        self.max_crops = max_crops
        self.min_score = min_score
        self.ocr = None
        try:
            from rapidocr_onnxruntime import RapidOCR

            self.ocr = RapidOCR()
        except Exception:  # noqa: BLE001 - optional dependency
            self.ocr = None

    @property
    def available(self) -> bool:
        return self.ocr is not None

    @staticmethod
    def torso(frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
        x1, y1, x2, y2 = box
        h, w = y2 - y1, x2 - x1
        H, W = frame.shape[:2]
        cy1, cy2 = int(max(0, y1 + 0.12 * h)), int(min(H, y1 + 0.55 * h))
        cx1, cx2 = int(max(0, x1 + 0.05 * w)), int(min(W, x2 - 0.05 * w))
        if cy2 - cy1 < 8 or cx2 - cx1 < 6:
            return None
        crop = frame[cy1:cy2, cx1:cx2]
        scale = max(1.0, 120.0 / crop.shape[0])  # the text detector needs ~>30 px digits
        return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    def read(self, frame: np.ndarray, boxes: np.ndarray, scores: np.ndarray) -> dict[int, tuple[int, float]]:
        """{person detection index: (number, confidence)} for the few most readable players."""
        if self.ocr is None or len(boxes) == 0:
            return {}
        H = frame.shape[0]
        h = boxes[:, 3] - boxes[:, 1]
        # large, confidently detected, not cut by the image border
        ok = (h >= self.min_height_frac * H) & (scores >= 0.5) & (boxes[:, 3] < H - 2) & (boxes[:, 1] > 1)
        idx = np.nonzero(ok)[0]
        idx = idx[np.argsort(-h[idx])][: self.max_crops]
        out = {}
        for j in idx:
            crop = self.torso(frame, boxes[j])
            if crop is None:
                continue
            try:
                res, _ = self.ocr(crop, use_det=True, use_cls=False, use_rec=True)
            except Exception:  # noqa: BLE001 - OCR must never break the pipeline
                continue
            best = None
            for r in res or []:
                txt, sc = str(r[1]), float(r[2])
                digits = "".join(c for c in txt if c.isdigit())
                # a shirt number is 1-2 digits and the text is (almost) only digits
                if not digits or len(digits) > 2 or len(digits) < len(txt.strip()) - 1 or sc < self.min_score:
                    continue
                n = int(digits)
                if 1 <= n <= 99 and (best is None or sc > best[1]):
                    best = (n, sc)
            if best is not None:
                out[int(j)] = best
        return out


def vote_number(reads: list[tuple[int, float]], min_votes: int = 2, min_share: float = 0.6) -> tuple[int | None, int]:
    """Shirt number of one identity from all its readings -> (number or None, votes)."""
    if not reads:
        return None, 0
    c = Counter()
    for n, s in reads:
        c[n] += s
    n, w = c.most_common(1)[0]
    votes = sum(1 for m, _ in reads if m == n)
    need = min_votes + (1 if n < 10 else 0)  # one digit is often half of a two-digit number
    if votes >= need and w / sum(c.values()) >= min_share:
        return n, votes
    return None, votes
