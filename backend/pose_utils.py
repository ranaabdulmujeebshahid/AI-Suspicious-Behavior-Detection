import numpy as np

FEATURE_COLS = (
    [f"lm_{i}_{axis}" for i in range(33) for axis in ("x", "y", "z", "vis")]
    + ["ang_l_elbow", "ang_r_elbow", "ang_l_shoulder", "ang_r_shoulder", "ang_l_knee", "ang_r_knee", "ang_torso"]
)

def _angle(a, b, c):
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cos_val = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.arccos(cos_val) / np.pi)

def extract_pose_features(landmarks):
    cx = (landmarks[23].x + landmarks[24].x) / 2.0
    cy = (landmarks[23].y + landmarks[24].y) / 2.0
    cz = (landmarks[23].z + landmarks[24].z) / 2.0

    sx = (landmarks[11].x + landmarks[12].x) / 2.0
    sy = (landmarks[11].y + landmarks[12].y) / 2.0
    scale = max(float(np.hypot(sx - cx, sy - cy)), 1e-4)

    feats = []
    for lm in landmarks:
        feats.extend([
            (lm.x - cx) / scale,
            (lm.y - cy) / scale,
            (lm.z - cz) / scale,
            float(lm.visibility or 0.0),
        ])

    angles = [
        _angle(landmarks[11], landmarks[13], landmarks[15]),
        _angle(landmarks[12], landmarks[14], landmarks[16]),
        _angle(landmarks[23], landmarks[11], landmarks[13]),
        _angle(landmarks[24], landmarks[12], landmarks[14]),
        _angle(landmarks[23], landmarks[25], landmarks[27]),
        _angle(landmarks[24], landmarks[26], landmarks[28]),
        float(np.arctan2(sx - cx, cy - sy) / np.pi),
    ]
    return feats + angles
