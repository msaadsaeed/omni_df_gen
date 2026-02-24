"""
data/face_detector.py
─────────────────────
Thread-local face detector with automatic fallback.

Priority:
  1. RetinaFace via insightface  (GPU-accelerated, most accurate)
  2. OpenCV Haar cascade         (zero-dep fallback, always available)

Detector instances are cached in thread-local storage so each DataLoader
worker initialises the detector exactly once and reuses it for every
subsequent batch — avoiding repeated model-load overhead.

All detection runs at CPU (ctx_id=-1) inside worker processes because
forked workers do NOT inherit the parent's CUDA context.
"""

from __future__ import annotations

import io
import logging
import sys
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
FACE_SIZE   = 224      # output spatial resolution (H × W)
FACE_MARGIN = 0.25     # fractional padding around the detected bounding box

_tls = threading.local()   # thread-local detector cache


# ── Geometry helpers ─────────────────────────────────────────────────────────

def _bbox_to_square_crop(
    x1: int, y1: int, x2: int, y2: int,
    H: int, W: int,
    margin: float = FACE_MARGIN,
) -> Tuple[int, int, int, int]:
    """Expand a face bbox into a square crop with margin, clipped to frame."""
    cx   = (x1 + x2) / 2.0
    cy   = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + margin)
    half = side / 2.0
    lx   = int(max(0, cx - half))
    ly   = int(max(0, cy - half))
    rx   = int(min(W, cx + half))
    ry   = int(min(H, cy + half))
    return lx, ly, rx, ry


def _centre_crop(frame_bgr: np.ndarray) -> np.ndarray:
    """Fallback: largest centred square crop of the frame."""
    H, W   = frame_bgr.shape[:2]
    side   = min(H, W)
    lx     = (W - side) // 2
    ly     = (H - side) // 2
    return frame_bgr[ly: ly + side, lx: lx + side]


# ── Detector classes ─────────────────────────────────────────────────────────

class _RetinaFaceDetector:
    """
    insightface RetinaFace — CPU mode (ctx_id=-1), safe inside forked workers.
    ONNX-Runtime EP-selection warnings are suppressed (expected & harmless).
    """

    def __init__(self) -> None:
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            from insightface.app import FaceAnalysis
            self._app = FaceAnalysis(
                name="buffalo_sc",
                allowed_modules=["detection"],
            )
            self._app.prepare(ctx_id=-1, det_size=(320, 320))
        finally:
            sys.stderr = old_stderr

    def detect(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        faces = self._app.get(frame_bgr)
        if not faces:
            return None
        best = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        x1, y1, x2, y2 = best.bbox.astype(int)
        return x1, y1, x2, y2


class _HaarDetector:
    """OpenCV Haar cascade — zero extra dependencies, always available."""

    def __init__(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._clf = cv2.CascadeClassifier(cascade_path)

    def detect(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._clf.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        return x, y, x + w, y + h


# ── Public API ────────────────────────────────────────────────────────────────

def get_detector() -> "_RetinaFaceDetector | _HaarDetector":
    """
    Return a worker-local face detector, created once per worker process.
    Always prefer RetinaFace; fall back to Haar if insightface is absent.
    """
    if not hasattr(_tls, "detector"):
        try:
            _tls.detector = _RetinaFaceDetector()
            logger.info("Face detector: RetinaFace (insightface, CPU mode)")
        except Exception as exc:
            logger.warning(
                f"insightface unavailable ({exc}); falling back to Haar cascade."
            )
            _tls.detector = _HaarDetector()
    return _tls.detector


def detect_and_crop_face(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Detect the largest face in a BGR uint8 frame.
    Returns a square crop resized to (FACE_SIZE, FACE_SIZE, 1) grayscale uint8.
    Falls back to centre crop when no face is detected.
    """
    H, W      = frame_bgr.shape[:2]
    detector  = get_detector()
    bbox      = detector.detect(frame_bgr)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = _bbox_to_square_crop(x1, y1, x2, y2, H, W)
        crop = frame_bgr[y1:y2, x1:x2]
    else:
        crop = _centre_crop(frame_bgr)

    if crop.size == 0:          # guard against degenerate boxes
        crop = _centre_crop(frame_bgr)

    resized = cv2.resize(crop, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LINEAR)
    gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)   # (H, W)
    return gray[:, :, np.newaxis]                          # (H, W, 1)
