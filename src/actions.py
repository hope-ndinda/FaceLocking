import numpy as np

def detect_move(kps, prev_nose_x, thresh=8):
    nose_x = float(kps[2][0])
    if prev_nose_x is None:
        return None, nose_x

    dx = nose_x - prev_nose_x
    if dx > thresh:
        return ("MOVE_RIGHT", f"dx={dx:.1f}"), nose_x
    if dx < -thresh:
        return ("MOVE_LEFT", f"dx={dx:.1f}"), nose_x
    return None, nose_x


def eye_aspect_ratio(kps):
    # simple vertical vs horizontal eye distance
    le, re = kps[0], kps[1]
    return abs(le[1] - re[1]) / (abs(le[0] - re[0]) + 1e-6)


def detect_blink(kps, thresh=0.15):
    ear = eye_aspect_ratio(kps)
    if ear < thresh:
        return ("BLINK", f"EAR={ear:.2f}")
    return None


def detect_smile(kps, thresh=1.8):
    lm, rm = kps[3], kps[4]
    mouth_w = abs(rm[0] - lm[0])
    mouth_h = abs(rm[1] - kps[2][1]) + 1e-6
    ratio = mouth_w / mouth_h
    if ratio > thresh:
        return ("SMILE", f"ratio={ratio:.2f}")
    return None
