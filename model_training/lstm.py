"""
Run from the project root:
    python model_training/lstm.py

Reads videos from training_data/{standing,walking,lifting}/
Extracts MediaPipe landmarks, trains ActivityLSTM, saves to weights/activity_lstm.pt
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path when called as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import mediapipe as mp
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model_training.config import (
    BATCH,
    DATA_DIR,
    DEVICE,
    EPOCHS,
    LR,
    POSE_DETECTION_CONFIDENCE,
    POSE_TRACKING_CONFIDENCE,
    SAVE_PATH,
    SEQ_LEN,
)
from runners_helpers.lstm_classifier import ACTIVITY_CLASSES, ActivityLSTM


def extract_sequences(label: str) -> list[list]:
    folder = DATA_DIR / label
    videos = list(folder.glob('*.mp4')) + list(folder.glob('*.avi'))
    if not videos:
        print(f"  [!] No videos found in {folder}")
        return []

    mp_pose = mp.solutions.pose
    sequences = []

    for vid in videos:
        cap = cv2.VideoCapture(str(vid))
        buffer = []

        with mp_pose.Pose(min_detection_confidence=POSE_DETECTION_CONFIDENCE,
                  min_tracking_confidence=POSE_TRACKING_CONFIDENCE) as pose:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if not res.pose_landmarks:
                    buffer.clear()   # broken sequence — reset
                    continue

                frame_vec = []
                for lm in res.pose_landmarks.landmark:
                    frame_vec += [lm.x, lm.y]   # 66 values per frame

                buffer.append(frame_vec)

                if len(buffer) == SEQ_LEN:
                    sequences.append(buffer.copy())
                    buffer.pop(0)

        cap.release()
        print(f"    {vid.name}: {len(sequences)} sequences so far")

    return sequences


def build_dataset():
    all_X, all_y = [], []

    for label in ACTIVITY_CLASSES:
        print(f"Processing: {label}")
        seqs = extract_sequences(label)
        idx  = ACTIVITY_CLASSES.index(label)
        all_X.extend(seqs)
        all_y.extend([idx] * len(seqs))
        print(f"  → {len(seqs)} sequences\n")

    X = torch.tensor(all_X, dtype=torch.float32)
    y = torch.tensor(all_y, dtype=torch.long)
    return X, y


def train():
    print(f"Device: {DEVICE}\n")

    X, y = build_dataset()
    print(f"Dataset: {len(y)} total sequences")
    for i, name in enumerate(ACTIVITY_CLASSES):
        print(f"  {name}: {(y == i).sum().item()}")

    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH, shuffle=True)

    model    = ActivityLSTM().to(DEVICE)
    opt      = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn  = nn.CrossEntropyLoss()

    print(f"\nTraining for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss   = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            correct    += (logits.argmax(1) == yb).sum().item()
            total      += len(yb)

        acc = correct / total * 100
        print(f"Epoch {epoch+1:02d}/{EPOCHS}  "
              f"loss={total_loss/len(loader):.4f}  acc={acc:.1f}%")

    SAVE_PATH.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\nSaved → {SAVE_PATH}")


if __name__ == '__main__':
    train()
