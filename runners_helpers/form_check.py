import cv2 as cv

from utils.helpers import get_angle, get_point, is_visible

COLORS = {
    'standing': (200, 200, 200),
    'walking':  (255, 200, 0),
    'lifting':  (0, 200, 255),
    'unknown':  (100, 100, 100),
    'good':     (0, 220, 100),
    'warn':     (0, 140, 255),
}


def check_squat_form(landmarks, w: int, h: int) -> list[str]:
    lm = landmarks
    feedback = []

    if not is_visible(lm, 11, 12, 23, 24, 25, 26, 27, 28):
        return ['Cannot assess — key joints not visible']

    left_shoulder  = get_point(lm, 11, w, h)
    right_shoulder = get_point(lm, 12, w, h)
    left_hip       = get_point(lm, 23, w, h)
    right_hip      = get_point(lm, 24, w, h)
    left_knee      = get_point(lm, 25, w, h)
    right_knee     = get_point(lm, 26, w, h)
    left_ankle     = get_point(lm, 27, w, h)
    right_ankle    = get_point(lm, 28, w, h)

    left_knee_angle  = get_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = get_angle(right_hip, right_knee, right_ankle)
    avg_knee_angle   = (left_knee_angle + right_knee_angle) / 2

    if avg_knee_angle > 110:
        feedback.append(f'Go deeper — knee angle {avg_knee_angle:.0f}° (target <100°)')
    else:
        feedback.append(f'Good depth — knee angle {avg_knee_angle:.0f}°')

    mid_hip_x = (left_hip[0] + right_hip[0]) / 2
    if left_knee[0] > left_hip[0]:
        feedback.append('Left knee caving inward')
    if right_knee[0] < right_hip[0]:
        feedback.append('Right knee caving inward')

    mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) // 2,
                    (left_shoulder[1] + right_shoulder[1]) // 2)
    mid_hip      = ((left_hip[0] + right_hip[0]) // 2,
                    (left_hip[1] + right_hip[1]) // 2)
    mid_knee     = ((left_knee[0] + right_knee[0]) // 2,
                    (left_knee[1] + right_knee[1]) // 2)

    torso_angle = get_angle(mid_shoulder, mid_hip, mid_knee)
    if torso_angle < 50:
        feedback.append('Torso too far forward — brace and stay upright')

    foot_spread    = abs(left_ankle[0] - right_ankle[0])
    shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
    if foot_spread < shoulder_width * 0.7:
        feedback.append('Feet too narrow — go shoulder-width or wider')

    return feedback


def check_deadlift_form(landmarks, w: int, h: int) -> list[str]:
    lm = landmarks
    feedback = []

    if not is_visible(lm, 11, 12, 23, 24, 25, 26, 15, 16):
        return ['Cannot assess — key joints not visible']

    left_shoulder  = get_point(lm, 11, w, h)
    right_shoulder = get_point(lm, 12, w, h)
    left_hip       = get_point(lm, 23, w, h)
    right_hip      = get_point(lm, 24, w, h)
    left_knee      = get_point(lm, 25, w, h)
    right_knee     = get_point(lm, 26, w, h)
    left_wrist     = get_point(lm, 15, w, h)
    right_wrist    = get_point(lm, 16, w, h)

    mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) // 2,
                    (left_shoulder[1] + right_shoulder[1]) // 2)
    mid_hip      = ((left_hip[0] + right_hip[0]) // 2,
                    (left_hip[1] + right_hip[1]) // 2)
    mid_knee     = ((left_knee[0] + right_knee[0]) // 2,
                    (left_knee[1] + right_knee[1]) // 2)
    mid_wrist    = ((left_wrist[0] + right_wrist[0]) // 2,
                    (left_wrist[1] + right_wrist[1]) // 2)

    back_angle = get_angle(mid_shoulder, mid_hip, mid_knee)
    if back_angle < 110:
        feedback.append(f'Back rounding — keep chest up, angle {back_angle:.0f}°')
    else:
        feedback.append(f'Back looks flat — angle {back_angle:.0f}°')

    wrist_hip_dist = abs(mid_wrist[0] - mid_hip[0])
    if wrist_hip_dist > w * 0.1:
        feedback.append('Bar drifting away from body — keep it close')

    left_ankle  = get_point(lm, 27, w, h)
    right_ankle = get_point(lm, 28, w, h)
    mid_ankle_x = (left_ankle[0] + right_ankle[0]) / 2
    if mid_hip[0] < mid_ankle_x - 30:
        feedback.append('Good hip hinge — hips back')
    elif mid_hip[0] > mid_ankle_x + 30:
        feedback.append('Hips too far forward — push them back more')

    vertical_offset = abs(mid_shoulder[0] - mid_hip[0])
    if vertical_offset < 40:
        feedback.append('Good lockout position')
    else:
        feedback.append('Not fully locked out — squeeze glutes and stand tall')

    return feedback


_GOOD_KWS = ('good', 'flat', 'hinge')


def assess_form(feedback: list[str]) -> bool:
    """True if every feedback line is a positive signal (no warnings)."""
    if not feedback or feedback[0].startswith('Cannot'):
        return False
    return all(any(kw in line.lower() for kw in _GOOD_KWS) for line in feedback)


def draw_feedback(frame, activity: str, feedback: list[str]) -> None:
    color = COLORS.get(activity, (200, 200, 200))

    cv.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), 1)
    cv.putText(frame, f'Activity: {activity.upper()}', (10, 28),
                cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 4)

    if feedback:
        y = 70
        for line in feedback:
            is_good = any(kw in line.lower() for kw in ['good', 'flat', 'hinge'])
            fc = COLORS['good'] if is_good else COLORS['warn']
            cv.putText(frame, line, (10, y),
                        cv.FONT_HERSHEY_SIMPLEX, 0.55, fc, 4)
            y += 24
