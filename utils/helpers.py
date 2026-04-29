import ctypes
import os
import time
from pathlib import Path

import cv2 as cv
import numpy as np

################################################################################################
# To avoud getting a view mode larger than your screen :)
def get_screen_size() -> tuple[int, int]:
    try:
        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return 1280, 720


def fit_for_display(frame, max_w: int, max_h: int):
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        return frame

    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 1.0:
        return frame

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv.resize(frame, (new_w, new_h), interpolation=cv.INTER_AREA)
################################################################################################

def build_output_path(source: str, output_dir: str, is_image: bool, model: str = "") -> str:
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(source).stem if source != "0" else "webcam"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    ext = ".jpg" if is_image else ".mp4"
    return str(Path(output_dir) / f"{stem}_{model}_{stamp}{ext}")


def create_runner(model: str):
    if model == "mp":
        from runners.mp_runner import MPRunner  # lazy — avoids circular import
        return MPRunner("weights/pose_landmarker_full.task", "weights/face_landmarker.task")
    elif model == "rf":
        from runners.rf_runner import RFRunner
        return RFRunner()
    elif model == "cn_op":
        from runners.cn_op_runner import CNOPRunner
        return CNOPRunner()
    elif model == "yv8pose":
        from runners.yv8pose_runner import YV8PoseRunner
        return YV8PoseRunner()
    else:
        raise ValueError(f"Unknown model: {model!r}")

################################################################################################
# Pose / geometry utilities shared across runners and helpers

def get_angle(a, b, c) -> float:
    """Angle in degrees at vertex B given points A, B, C as (x, y) tuples."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def get_point(landmarks, idx: int, w: int, h: int) -> tuple[int, int]:
    """Return pixel (x, y) for a landmark index."""
    lm = landmarks[idx]
    return (int(lm.x * w), int(lm.y * h))


def is_visible(landmarks, *indices: int, threshold: float = 0.5) -> bool:
    """True only if every listed landmark exceeds the visibility threshold."""
    return all(landmarks[i].visibility > threshold for i in indices)


class _LMPoint:
    """Lightweight landmark proxy — matches the MediaPipe NormalizedLandmark interface."""
    __slots__ = ('x', 'y', 'z', 'visibility')

    def __init__(self, x: float, y: float, vis: float = 1.0):
        self.x = x
        self.y = y
        self.z = 0.0
        self.visibility = vis


# Maps YOLOv8 17-point indices → MediaPipe 33-point indices
_YV8_TO_MP: dict[int, int] = {
    0: 0,    # nose
    5: 11,   # L shoulder
    6: 12,   # R shoulder
    7: 13,   # L elbow
    8: 14,   # R elbow
    9: 15,   # L wrist
    10: 16,  # R wrist
    11: 23,  # L hip
    12: 24,  # R hip
    13: 25,  # L knee
    14: 26,  # R knee
    15: 27,  # L ankle
    16: 28,  # R ankle
}


def yv8_to_mp_landmarks(keypoints_row, w: int, h: int) -> list:
    """
    Convert a single YOLOv8 keypoints row (17, 3) to a 33-slot _LMPoint list
    whose indices align with MediaPipe's landmark numbering so form-check
    functions work identically for both backends.
    """
    dummy = _LMPoint(0.0, 0.0, 0.0)
    lms: list = [dummy] * 33
    for yv_idx, mp_idx in _YV8_TO_MP.items():
        x, y, conf = keypoints_row[yv_idx]
        lms[mp_idx] = _LMPoint(x / w, y / h, float(conf))
    return lms