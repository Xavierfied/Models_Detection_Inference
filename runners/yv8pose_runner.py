from ultralytics import YOLO
import cv2 as cv
import numpy as np


#  Pairs of (start_point_index, end_point_index)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # Facial features
    (5, 6),                              # Shoulders
    (5, 7), (7, 9),                      # Left arm
    (6, 8), (8, 10),                     # Right arm
    (5, 11), (6, 12), (11, 12),          # Torso
    (11, 13), (13, 15),                  # Left leg
    (12, 14), (14, 16)                   # Right leg
]

MODEL_PATH = "weights/yolov8m-pose.pt"
####################################################################################

class YV8PoseRunner:
    def __init__(self, model_path=MODEL_PATH):
        self.model = YOLO(model_path)
    

    def get_landmarks(self, frame, timestamp_ms=None):
        results = self.model(frame)
        return results[0].keypoints, None
    
    ################################################################################
    def draw_keypoints_on_frame(self, frame, pose_res, face_res=None):

        if pose_res is not None:

            all_kpts = pose_res.data.cpu().numpy()  # (num_people, num_keypoints, 4)

            for person in all_kpts:
                for keypoint in person:

                    x, y, conf = keypoint

                    if conf > 0.3:  # confidence threshold
                        cv.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

            # Draw skeleton connections
            for start_idx, end_idx in SKELETON_CONNECTIONS:
                start_point = all_kpts[:, start_idx]
                end_point = all_kpts[:, end_idx]

                # Check if both points are visible
                if np.all(start_point[:, 2] > 0.3) and np.all(end_point[:, 2] > 0.3):
                    for (x1, y1, _), (x2, y2, _) in zip(start_point, end_point):
                        cv.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        return frame
    
    ########################################################################################

    def draw_pose_positions(self, frame, pose_res, face_res=None):
        
        if pose_res is not None:
            all_kpts = pose_res.data.cpu().numpy()  # (num_people, num_keypoints, 4)

            for person in all_kpts:
                val_kpts = person[person[:, 2] > 0.3]  

                # Skip if there arent many visible kpts of a person
                if len(val_kpts) < 5:
                    continue
                    
                # Calculate BBs
                x_min, y_min = np.min(val_kpts[:, 0]), np.min(val_kpts[:, 1])
                x_max, y_max = np.max(val_kpts[:, 0]), np.max(val_kpts[:, 1])

                width, height = x_max - x_min, y_max - y_min

                # Check sitting/standing 

                # Standing ppl are 2x taller than they are wide, sitting ppl are closer to 1:1
                aspect_ratio = height / width if width > 0 else 0

                # Implementing via identifying joints (Hip & Knee) and aspect ratio as a fallback
                status = "Standing"

                if (person[11][2] > 0.3 or person[12][2] > 0.3) and (person[13][2] > 0.3 or person[14][2] > 0.3):
                    # person[11] = left hip, person[12] = right hip, person[13] = left knee, person[14] = right knee and [2] is confidence
                    hip_y = person[11][1] if person[11][2] > 0.3 else person[12][1]
                    knee_y = person[13][1] if person[13][2] > 0.3 else person[14][1]

                    # If the vertical distance between hip and knee is small, they are sitting
                    # Issue: Elevated Camera Angle
                    if abs(hip_y - knee_y) < (height * 0.15):

                        status = "Sitting"

                else:
                    # cant see legs
                    if aspect_ratio < 1.2:
                        status = "Sitting"

                # 4. Draw Box and Text
                color = (255, 255, 0) if status == "Standing" else (0, 165, 255) # Cyan vs Orange
                cv.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
                cv.putText(frame, f"{status}", (int(x_min), int(y_min) - 10), 
                cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame
    
    ######################################################################################
    def draw_on_frame(self, frame, pose_res, face_res=None):
        """Draw keypoints, skeleton, and pose position (sitting/standing) on frame."""
        if pose_res is None:
            return frame
        
        frame = self.draw_keypoints_on_frame(frame, pose_res, face_res)
        frame = self.draw_pose_positions(frame, pose_res, face_res)
        return frame
    
    #####################################################################################

    def close(self):
        return None
##########################################################################################

# yv = YV8PoseRunner(MODEL_PATH)
# # kp = yv.get_landmarks("samples\mix.jpg")

# # ka = yv.draw_on_frame(cv.imread("samples\mix.jpg"), kp)
# # cv.imshow("Pose", ka)
# # cv.waitKey(0)
# # cv.destroyAllWindows()

# cap = cv.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     kp = yv.get_landmarks(frame)
#     frame = yv.draw_on_frame(frame, kp)
#     frame = yv.draw_pose_positions(frame, kp)
#     cv.imshow("Pose", frame)

#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv.destroyAllWindows()
