import time
from pathlib import Path

import cv2 as cv

from utils.args import get_args
from utils.helpers import build_output_path, create_runner, fit_for_display, get_screen_size

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

############################################################################################

def process_image(source: str, runner, save: bool, view: bool, output_dir: str, model: str):
    frame = cv.imread(source)
    if frame is None:
        raise FileNotFoundError("Could not read image")

    timestamp_ms = int(time.time() * 1000)
    pose_res, face_res = runner.get_landmarks(frame, timestamp_ms)
    output = runner.draw_on_frame(frame.copy(), pose_res, face_res)

    save_path = None
    if save:
        save_path = build_output_path(source, output_dir, is_image=True, model=model)
        cv.imwrite(save_path, output)
        print(f"Saved image to: {save_path}")

    if view:
        sw, sh = get_screen_size()
        preview = fit_for_display(output, int(sw * 0.9), int(sh * 0.9))
        cv.imshow("Inference", preview)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return save_path

############################################################################################

def process_video(source: str, runner, save: bool, view: bool, output_dir: str, model: str):
    
    # WebCam
    src = 0 if source == "0" else source
    
    # Same Execution for rest of the phase
    cap = cv.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")

    writer = None
    save_path = build_output_path(source, output_dir, is_image=False, model=model) if save else None
    sw, sh = get_screen_size()
    max_w, max_h = int(sw * 0.9), int(sh * 0.9)

    try:
        if save:
            fps = cap.get(cv.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0
            width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv.VideoWriter_fourcc(*"mp4v")
            writer = cv.VideoWriter(save_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            #To provide accurate stamps for MP Runner 
            timestamp_ms = int(cap.get(cv.CAP_PROP_POS_MSEC))
            pose_res, face_res = runner.get_landmarks(frame, timestamp_ms)
            output = runner.draw_on_frame(frame.copy(), pose_res, face_res)

            if writer is not None:
                writer.write(output)

            if view:
                preview = fit_for_display(output, max_w, max_h)
                cv.imshow("Inference", preview)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if view:
            cv.destroyAllWindows()

    if save:
        print(f"Saved video to: {save_path}")
    return save_path

############################################################################################

def main():
    args = get_args()
    source_ext = Path(args.source).suffix.lower()
    is_image = source_ext in IMAGE_EXTS

    runner = create_runner(args.model)
    try:
        if is_image:
            process_image(args.source, runner, args.save, args.view, args.output_dir, args.model)
        else:
            process_video(args.source, runner, args.save, args.view, args.output_dir, args.model)
    finally:
        if hasattr(runner, "close"):
            runner.close()
        elif hasattr(runner, "pose_landmarker"):
            runner.pose_landmarker.close()
        if hasattr(runner, "face_landmarker"):
            runner.face_landmarker.close()

############################################################################################

if __name__ == "__main__":
    main()