"""Shirt numbers (dorsales) with a small bundled OCR model, voted per identity.

On a wide broadcast shot a number is only readable when the player is near the
camera and shows his back, so any single frame reads few numbers and sometimes
misreads (17 -> 12). Reading a few of the largest players on every keyframe and
voting over everything an identity accumulated is what makes it usable:
identities get their real shirt number, and two fragments of the same player
(same team, same number, never on screen at the same time) are merged.

Uses ``rapidocr_onnxruntime`` (PP-OCRv4 models shipped inside the pip wheel, no
download). If it is not installed, numbers are simply not read.

Measured on the Bayern-PSG 720p clip (60 views of each of 10 players whose number is
visible): the text detector alone read 3 players' numbers; digit blobs + detector
read 6, with no wrong number. With every view of the clip (``refine.read_numbers``)
more players pass the vote.
"""
from __future__ import annotations

from collections import Counter

import cv2
import numpy as np


class JerseyReader:
    """Reads the number on a player's back with two complementary methods:

    1. *digit blobs*: printed numbers are uniform light (or dark) shapes that contrast with
       the shirt. Inside the back area, pixels much brighter / darker than the shirt's median
       are grouped into 1-2 digit-sized blobs; each blob group is cropped and read by the
       text recogniser alone. This finds lone digits that a text detector ignores.
    2. *text detector + recogniser* on the sharpened, upscaled back area (with lowered
       detection thresholds): better on crisp two-digit numbers.

    The more confident reading wins; agreement of both raises the confidence."""

    def __init__(self, min_height_frac: float = 0.08, max_crops: int = 3, min_score: float = 0.7,
                 det_min_score: float = 0.5):
        self.min_height_frac = min_height_frac
        self.max_crops = max_crops
        self.min_score = min_score
        self.det_min_score = det_min_score
        self.ocr = self.ocr_det = None
        try:
            from rapidocr_onnxruntime import RapidOCR

            self.ocr = RapidOCR()
            self.ocr_det = RapidOCR(det_thresh=0.2, det_box_thresh=0.3, det_unclip_ratio=2.0)
        except Exception:  # noqa: BLE001 - optional dependency
            self.ocr = self.ocr_det = None

    @property
    def available(self) -> bool:
        return self.ocr is not None

    # ------------------------------------------------------------ crops
    @staticmethod
    def torso(frame: np.ndarray, box: np.ndarray, target: int = 220) -> np.ndarray | None:
        x1, y1, x2, y2 = box
        h, w = y2 - y1, x2 - x1
        H, W = frame.shape[:2]
        cy1, cy2 = int(max(0, y1 + 0.12 * h)), int(min(H, y1 + 0.55 * h))
        cx1, cx2 = int(max(0, x1 + 0.05 * w)), int(min(W, x2 - 0.05 * w))
        if cy2 - cy1 < 8 or cx2 - cx1 < 6:
            return None
        crop = frame[cy1:cy2, cx1:cx2]
        scale = max(1.0, target / crop.shape[0])
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return cv2.addWeighted(crop, 1.6, cv2.GaussianBlur(crop, (0, 0), 2.0), -0.6, 0)  # unsharp mask

    @staticmethod
    def digit_windows(frame: np.ndarray, box: np.ndarray, up: int = 4) -> list[np.ndarray]:
        """Crops around groups of 1-2 digit-sized blobs that contrast with the shirt."""
        x1, y1, x2, y2 = box
        h, w = y2 - y1, x2 - x1
        H, W = frame.shape[:2]
        c = frame[int(max(0, y1 + 0.12 * h)):int(min(H, y1 + 0.62 * h)), int(max(0, x1 + 0.02 * w)):int(min(W, x2 - 0.02 * w))]
        if c.shape[0] < 10 or c.shape[1] < 6:
            return []
        c = cv2.resize(c, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        L = cv2.cvtColor(c, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
        shirt = ~((hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 60))  # not grass
        if shirt.sum() < 50:
            return []
        med = np.median(L[shirt])
        Hc, Wc = L.shape
        out = []
        for pol in (1, -1):  # light numbers on a dark / coloured shirt, dark numbers on a light shirt
            m = (pol * (L - med) > 25) & shirt
            if pol == 1:
                m &= hsv[..., 1] < 110  # printed numbers are white / light: low saturation
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
            comps = [tuple(st[k][:4]) for k in range(1, n)
                     if 0.18 * Hc < st[k][3] < 0.75 * Hc and st[k][2] < 0.6 * Wc and st[k][4] > 0.02 * st[k][3] ** 2
                     and 0.1 * Wc < st[k][0] + st[k][2] / 2 < 0.9 * Wc]
            comps.sort(key=lambda r: -r[3])
            for base in comps[:3]:
                grp = [r for r in comps if abs(r[1] - base[1]) < 0.3 * base[3] and abs(r[3] - base[3]) < 0.35 * base[3]
                       and abs((r[0] + r[2] / 2) - (base[0] + base[2] / 2)) < 1.4 * base[3]]
                xa, ya = min(r[0] for r in grp), min(r[1] for r in grp)
                xb, yb = max(r[0] + r[2] for r in grp), max(r[1] + r[3] for r in grp)
                pad = int(0.25 * (yb - ya))
                win = c[max(0, ya - pad):yb + pad, max(0, xa - pad):xb + pad]
                if win.size:
                    out.append(win)
        return out

    @staticmethod
    def _number(txt: str, sc: float) -> tuple[int, float] | None:
        digits = "".join(ch for ch in txt if ch.isdigit())
        # a shirt number is 1-2 digits and the text is (almost) only digits
        if not digits or len(digits) > 2 or len(digits) < len(txt.strip()) - 1:
            return None
        n = int(digits)
        return (n, sc) if 1 <= n <= 99 else None

    # ------------------------------------------------------------ reading
    def read_one(self, frame: np.ndarray, box: np.ndarray, use_det: bool = True) -> tuple[int, float] | None:
        """(number, confidence) for one player box, or None. ``use_det=False`` skips the slower
        text-detector method."""
        if self.ocr is None:
            return None
        cands = []
        try:
            best = None
            for win in self.digit_windows(frame, box):
                res, _ = self.ocr(win, use_det=False, use_cls=False, use_rec=True)
                for r in res or []:
                    p = self._number(str(r[0]), float(r[1]))
                    if p and (best is None or p[1] > best[1]):
                        best = p
            if best and best[1] >= self.min_score:
                cands.append(best)
            crop = self.torso(frame, box) if use_det else None
            if crop is not None:
                best = None
                res, _ = self.ocr_det(crop, use_det=True, use_cls=False, use_rec=True)
                for r in res or []:
                    p = self._number(str(r[1]), float(r[2]))
                    if p and (best is None or p[1] > best[1]):
                        best = p
                if best and best[1] >= self.det_min_score:
                    cands.append(best)
        except Exception:  # noqa: BLE001 - OCR must never break the pipeline
            return None
        if not cands:
            return None
        if len(cands) == 2 and cands[0][0] == cands[1][0]:
            return cands[0][0], min(1.0, max(cands[0][1], cands[1][1]) + 0.1)
        return max(cands, key=lambda c: c[1])

    def read(self, frame: np.ndarray, boxes: np.ndarray, scores: np.ndarray) -> dict[int, tuple[int, float]]:
        """{person detection index: (number, confidence)} for the few largest players (first pass)."""
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
            r = self.read_one(frame, boxes[j])
            if r is not None:
                out[int(j)] = r
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
