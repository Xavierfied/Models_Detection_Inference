import cv2 as cv
import mediapipe as mp

class MPRunner:
    def __init__(self, pose_weights, face_weights):
        # Initialize Landmarkers
        base_pose = mp.tasks.BaseOptions(model_asset_path=pose_weights)
        base_face = mp.tasks.BaseOptions(model_asset_path=face_weights)
        
        self.pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
            mp.tasks.vision.PoseLandmarkerOptions(
                base_options=base_pose, 
                running_mode=mp.tasks.vision.RunningMode.VIDEO)
        )
        self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base_face, 
                running_mode=mp.tasks.vision.RunningMode.VIDEO)
        )
#########################################################
    # Getting Coords
    def get_landmarks(self, frame, timestamp_ms):
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        pose_res = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        face_res = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        return pose_res, face_res
#########################################################
    # Drawing
    def draw_on_frame(self, frame, pose_res, face_res):
        # Draw Pose
        if pose_res.pose_landmarks:
            for landmarks in pose_res.pose_landmarks:
                for lm in landmarks:
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
#########################################################
        # Draw Face
        if face_res.face_landmarks:
            for landmarks in face_res.face_landmarks:
                for lm in landmarks:
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv.circle(frame, (cx, cy), 1, (255, 0, 0), -1)
        return frame