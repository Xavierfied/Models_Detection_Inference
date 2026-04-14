from controlnet_aux import OpenposeDetector
import cv2 as cv
import torch


class CNOPRunner:
    def __init__(self):
        self._last_pose = [] # to frame skip
        self._frame_idx = 0  # to frame skip


        try:
            self.model = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        except Exception as exc:
            raise RuntimeError(
                "CN-OP backend could not initialize OpenposeDetector. "
                "Check that controlnet_aux is installed."
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        for attr in ("body_estimation", "hand_estimation", "face_estimation"):
            sub = getattr(self.model, attr, None)
            if sub is not None and hasattr(sub, "to"):
                sub.to(device)


#####################################################################
    # def get_landmarks(self, frame, timestamp_ms):
    #     # BGR → RGB (detect_poses expects RGB input)
    #     rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    #     # Downscale to longest-side 640 for faster inference
    #     h, w = rgb.shape[:2]
    #     if max(h, w) > 640:
    #         scale = 640 / max(h, w)
    #         rgb = cv.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))),
    #                         interpolation=cv.INTER_AREA)

        # pose_res = self.model.detect_poses(rgb, include_face=False, include_hand=False)
        # return pose_res, None

    # Checking frame skipping
    def get_landmarks(self, frame, timestamp_ms):
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if max(h, w) > 640:
            scale = 640 / max(h, w)
            rgb = cv.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))),
                            interpolation=cv.INTER_AREA)

        if self._frame_idx % 3 == 0:   # only infer every 3rd frame
            self._last_pose = self.model.detect_poses(rgb, include_face=False, include_hand=False)
        self._frame_idx += 1

        return self._last_pose, None   # reuse on skipped frames
#####################################################################

    def close(self):
        pass
#####################################################################
    def draw_on_frame(self, frame, pose_res, _face_res=None):
        if not pose_res:
            return frame

        H, W = frame.shape[:2]

        def draw_keypoints(kps, color):
            if kps is None:
                return
            for kp in kps:
                if kp is None or kp.x < 0 or kp.y < 0:
                    continue
                cx, cy = int(kp.x * W), int(kp.y * H)
                cv.circle(frame, (cx, cy), 2, color, -1)

        for pose in pose_res:
            draw_keypoints(pose.body.keypoints, (255, 255, 0))
            draw_keypoints(pose.left_hand,      (0, 255, 255))
            draw_keypoints(pose.right_hand,     (0, 255, 255))
            draw_keypoints(pose.face,           (0, 180, 255))

        return frame
