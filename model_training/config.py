from pathlib import Path

import torch

DATA_DIR = Path('training_data')  # Folder with the labeled training videos.
SAVE_PATH = Path('weights/activity_lstm.pt')  # Where the trained model is saved.
SEQ_LEN = 30  # Number of frames used for each sequence.
EPOCHS = 30  # Number of training epochs.
BATCH = 32  # Batch size used by the DataLoader.
LR = 1e-3  # Learning rate for the Adam optimizer.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU when available.
POSE_DETECTION_CONFIDENCE = 0.5  # Minimum confidence for pose detection.
POSE_TRACKING_CONFIDENCE = 0.5  # Minimum confidence for pose tracking.
