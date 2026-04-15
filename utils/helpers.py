import ctypes
import os
import time
from pathlib import Path

import cv2 as cv

from runners.mp_runner import MPRunner

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