import torch
import torch.nn as nn

from utils.helpers import get_angle, get_point, is_visible

ACTIVITY_CLASSES = ['standing', 'walking', 'lifting', 'bicep', 'pushup', 'pullup']


class ActivityLSTM(nn.Module):
    """
    LSTM classifier for activity recognition.
    Input:  (batch, 30, 66) — 30 frames × 33 landmarks × (x, y)
    Output: (batch, N)      — logits for each class in ACTIVITY_CLASSES
    """

    def __init__(self, input_size: int = 66, hidden_size: int = 128,
                 num_layers: int = 2, num_classes: int = len(ACTIVITY_CLASSES)):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def load_classifier(weights_path: str, device: str = 'cpu'):
    """Return a loaded ActivityLSTM on `device`, or None if weights are unavailable."""
    model = ActivityLSTM(num_classes=len(ACTIVITY_CLASSES))
    try:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model
    except Exception:
        return None


def detect_activity(landmarks, w: int, h: int) -> str:
    """
    Rule-based activity classifier.
    Returns one of: 'standing', 'walking', 'lifting', 'unknown'
    """
    lm = landmarks
    if not is_visible(lm, 11, 12, 23, 24, 25, 26, 27, 28):
        return 'unknown'

    left_hip    = get_point(lm, 23, w, h)
    right_hip   = get_point(lm, 24, w, h)
    left_knee   = get_point(lm, 25, w, h)
    right_knee  = get_point(lm, 26, w, h)
    left_ankle  = get_point(lm, 27, w, h)
    right_ankle = get_point(lm, 28, w, h)

    left_knee_angle  = get_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = get_angle(right_hip, right_knee, right_ankle)
    avg_knee_angle   = (left_knee_angle + right_knee_angle) / 2
    knee_asymmetry   = abs(left_knee_angle - right_knee_angle)

    # Measure how far hips have dropped toward knee level.
    # y increases downward; when squatting, mid_knee_y - mid_hip_y shrinks.
    mid_hip_y   = (left_hip[1]   + right_hip[1])   / 2
    mid_knee_y  = (left_knee[1]  + right_knee[1])  / 2
    mid_ankle_y = (left_ankle[1] + right_ankle[1]) / 2
    upper_leg_px = max(mid_knee_y - mid_hip_y,   1)   # femur projection
    lower_leg_px = max(mid_ankle_y - mid_knee_y, 1)   # tibia projection
    # 0 = hips well above knees (standing), →1 = hips at knee level (deep squat)
    squat_depth = 1.0 - min(upper_leg_px / lower_leg_px, 1.0)

    # Strong knee bend or hips visibly dropped → lifting (squat / deadlift)
    if avg_knee_angle < 130 or (avg_knee_angle < 155 and squat_depth > 0.35):
        return 'lifting'

    # One knee noticeably more bent than the other → walking
    if knee_asymmetry > 25 and avg_knee_angle < 165:
        return 'walking'

    return 'standing'
