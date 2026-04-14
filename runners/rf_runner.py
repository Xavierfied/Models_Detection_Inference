import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import cv2 as cv


class RFRunner:
    def __init__(self):
        from retinaface import RetinaFace # Cant write above due to dependency issue 

        self.retinaface = RetinaFace
        try:
            self.model = RetinaFace.build_model()
        except Exception as exc:
            raise RuntimeError(
                "RF backend could not initialize RetinaFace. "
                "This workspace currently has a TensorFlow/Keras compatibility issue "
                "with the installed RetinaFace package."
            ) from exc

    def get_landmarks(self, frame, timestamp_ms):
        return None, detect_faces(frame, self.retinaface, self.model)

    def draw_on_frame(self, frame, pose_res, face_res):
        if isinstance(face_res, dict):
            for key in face_res:
                draw_face_full(frame, face_res[key])
        return frame

    def close(self):
        return None


def draw_face_full(frame, data):
    x1, y1, x2, y2 = data["facial_area"]
    score = data["score"]
    lm = data["landmarks"]
    rx, ry = lm["right_eye"]
    lx, ly = lm["left_eye"]
    nx, ny = lm["nose"]

    # Box with rounded corners (manual via arcs)
    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

    # Score badge
    cv.rectangle(frame, (x1, y1 - 22), (x1 + 90, y1), (0, 200, 255), -1)
    cv.putText(frame, f"Face {score:.2f}",
                (x1 + 3, y1 - 6),
                cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # Eye line (shows head tilt)
    cv.line(frame, (int(rx), int(ry)), (int(lx), int(ly)),
             (255, 255, 0), 2)

    # Nose dot
    cv.circle(frame, (int(nx), int(ny)), 6, (255, 100, 0), -1)

    # Eye dots
    for (ex, ey) in [(rx, ry), (lx, ly)]:
        cv.circle(frame, (int(ex), int(ey)), 6, (0, 255, 180), -1)
        cv.circle(frame, (int(ex), int(ey)), 6, (255, 255, 255), 1)

    # Mouth line
    mx1, my1 = lm["mouth_right"]
    mx2, my2 = lm["mouth_left"]
    cv.line(frame, (int(mx1), int(my1)), (int(mx2), int(my2)),
             (0, 100, 255), 2)


def detect_faces(frame, retinaface, model):
    # RetinaFace expects RGB input
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    faces = retinaface.detect_faces(rgb_frame, model=model, threshold=0.8)

    return faces