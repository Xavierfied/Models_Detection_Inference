import cv2 as cv
import mediapipe as mp
import torch
# from pathlib import Path

from runners_helpers.form_check import assess_form, check_deadlift_form, check_squat_form, draw_feedback
from runners_helpers.lstm_classifier import ACTIVITY_CLASSES, detect_activity, load_classifier
from utils.helpers import build_output_path, get_angle, get_point

WEIGHTS_PATH = "weights/activity_lstm.pt"
SEQ_LEN      = 30
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'


def run(source, config) -> None:
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    classifier = load_classifier(WEIGHTS_PATH, DEVICE)
    if classifier:
        print(f"LSTM loaded from {WEIGHTS_PATH}  [device: {DEVICE}]")
    else:
        print("LSTM weights not found — using rule-based fallback")

    is_webcam = str(source) == '0'
    src       = 0 if is_webcam else source
    cap       = cv.VideoCapture(src)
    if not cap.isOpened():
        print(f"Error: could not open source '{source}'")
        return

    fps        = cap.get(cv.CAP_PROP_FPS) or 30.0
    frame_wait = 1 if is_webcam else max(1, int(1000 / fps))  # respect video speed

    writer = None
    if getattr(config, 'save', False):
        fw  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        fh  = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        out = build_output_path(str(source), 'results', False, 'lift')
        writer = cv.VideoWriter(out, cv.VideoWriter_fourcc(*'mp4v'), fps, (fw, fh))
        print(f"Saving to {out}")

    buffer = []   # sliding window — 30 frames × 66 features

    with mp_pose.Pose(min_detection_confidence=0.6,
                      min_tracking_confidence=0.6) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w    = frame.shape[:2]
            results = pose.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

            activity = 'unknown'
            feedback = []

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                mp_draw.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2),
                )

                # Build sliding window
                frame_vec = [v for lmk in lm for v in (lmk.x, lmk.y)]
                buffer.append(frame_vec)
                if len(buffer) > SEQ_LEN:
                    buffer.pop(0)

                # Classify
                if classifier and len(buffer) == SEQ_LEN:
                    with torch.no_grad():
                        x        = torch.tensor([buffer], dtype=torch.float32).to(DEVICE)
                        activity = ACTIVITY_CLASSES[classifier(x).argmax(1).item()]
                else:
                    activity = detect_activity(lm, w, h)

                # Form check only when lifting
                if activity == 'lifting':
                    knee_angle = get_angle(
                        get_point(lm, 23, w, h),
                        get_point(lm, 25, w, h),
                        get_point(lm, 27, w, h),
                    )
                    if knee_angle < 120:
                        feedback  = check_squat_form(lm, w, h)
                        lift_type = 'SQUAT'
                    else:
                        feedback  = check_deadlift_form(lm, w, h)
                        lift_type = 'DEADLIFT'

                    # lift type label
                    cv.putText(frame, lift_type, (w - 160, 30),
                               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

                    # verdict banner
                    optimal      = assess_form(feedback)
                    verdict_text = 'OPTIMAL FORM' if optimal else 'NEEDS WORK'
                    verdict_col  = (0, 200, 80) if optimal else (0, 80, 220)
                    banner_w     = 220
                    cv.rectangle(frame, (w - banner_w, 45), (w, 85), (0, 0, 0), -1)
                    cv.rectangle(frame, (w - banner_w, 45), (w, 85), verdict_col, 2)
                    cv.putText(frame, verdict_text, (w - banner_w + 8, 73),
                               cv.FONT_HERSHEY_SIMPLEX, 0.75, verdict_col, 2)

            draw_feedback(frame, activity, feedback)

            if writer:
                writer.write(frame)

            cv.imshow('LiftOptimize', frame)
            if cv.waitKey(frame_wait) & 0xFF == ord('q'):
                break

    cap.release()
    if writer:
        writer.release()
    cv.destroyAllWindows()
