# collect_from_videos.py
import cv2, mediapipe as mp, numpy as np, json
from pathlib import Path

DATA_DIR = Path('training_data')          # your folder above
SEQ_LEN  = 30
LABELS   = ['standing', 'walking', 'lifting']

mp_pose = mp.solutions.pose
all_sequences, all_labels = [], []

for label in LABELS:
    folder = DATA_DIR / label
    if not folder.exists():
        print(f"Skipping {label} — folder not found")
        continue

    video_files = list(folder.glob('*.mp4')) + list(folder.glob('*.avi'))
    print(f"{label}: processing {len(video_files)} videos")

    for video_path in video_files:
        cap = cv2.VideoCapture(str(video_path))
        buffer = []

        with mp_pose.Pose(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if not res.pose_landmarks:
                    buffer.clear()   # reset — broken sequence is useless
                    continue

                frame_vec = []
                for lm in res.pose_landmarks.landmark:
                    frame_vec += [lm.x, lm.y]   # 66 numbers per frame

                buffer.append(frame_vec)

                if len(buffer) == SEQ_LEN:
                    all_sequences.append(buffer.copy())
                    all_labels.append(LABELS.index(label))
                    buffer.pop(0)    # slide by 1 frame

        cap.release()

print(f"\nTotal sequences: {len(all_sequences)}")
for i, name in enumerate(LABELS):
    count = all_labels.count(i)
    print(f"  {name}: {count}")

np.save('sequences.npy', np.array(all_sequences, dtype=np.float32))
np.save('labels.npy',    np.array(all_labels,    dtype=np.int64))
print("Saved sequences.npy and labels.npy")
