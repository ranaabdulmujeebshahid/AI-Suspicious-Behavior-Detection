import os
import glob
import random
import cv2
import joblib
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from pose_utils import FEATURE_COLS, extract_pose_features

BASE_DIR = "/content/drive/MyDrive/AI_Final_Project/backend"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CLEAN_CSV = os.path.join(DATASET_DIR, "pose_features_cctv_clean.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "behavior_rf_model.pkl")

yolo = YOLO(os.path.join(MODELS_DIR, "yolov8n.pt"))
base_options = mp_python.BaseOptions(model_asset_path=os.path.join(MODELS_DIR, "pose_landmarker_lite.task"))
landmarker = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_poses=1)
)

CLASS_SOURCES = {
    "Normal": [
        os.path.join(DATASET_DIR, "UCF_Crime/Normal/**/*.mp4"),
    ],
    "Fighting_Assault": [
        os.path.join(DATASET_DIR, "UCF_Crime/Anomaly/**/Fighting/*.mp4"),
        os.path.join(DATASET_DIR, "UCF_Crime/Anomaly/**/Assault/*.mp4"),
        os.path.join(DATASET_DIR, "UCF_Crime/Anomaly/**/Abuse/*.mp4"),
    ],
    "Robbery_Theft": [
        os.path.join(DATASET_DIR, "UCF_Crime/Anomaly/**/Robbery/*.mp4"),
        os.path.join(DATASET_DIR, "UCF_Crime/Anomaly/**/Burglary/*.mp4"),
        os.path.join(DATASET_DIR, "UCF_Crime/Anomaly/**/Stealing/*.mp4"),
        os.path.join(DATASET_DIR, "UCF_Crime/Anomaly/**/Shoplifting/*.mp4"),
    ],
    "Critical_Threat": [
        os.path.join(DATASET_DIR, "UCF_Crime/Anomaly/**/Shooting/*.mp4"),
        os.path.join(DATASET_DIR, "UCF_Crime/Anomaly/**/Arson/*.mp4"),
        os.path.join(DATASET_DIR, "UCF_Crime/Anomaly/**/Explosion/*.mp4"),
        os.path.join(DATASET_DIR, "UCF_Crime/Anomaly/**/Vandalism/*.mp4"),
    ],
}

TARGET_SAMPLES = 900
random.seed(42)
rows = []

for label, patterns in CLASS_SOURCES.items():
    videos = []
    for p in patterns:
        videos.extend(glob.glob(p, recursive=True))
    videos = sorted(set(videos))
    random.shuffle(videos)
    collected = 0
    print(f"[{label}] Extracting up to {TARGET_SAMPLES} CCTV samples from {len(videos)} videos...")

    for vpath in videos:
        if collected >= TARGET_SAMPLES:
            break
        cap = cv2.VideoCapture(vpath)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total_frames < 30:
            cap.release()
            continue

        start_f = 0 if label == "Normal" else int(total_frames * 0.2)
        end_f = total_frames if label == "Normal" else int(total_frames * 0.8)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

        frame_idx = start_f
        per_video = 0
        while cap.isOpened() and frame_idx <= end_f and collected < TARGET_SAMPLES and per_video < 25:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 12 == 0:
                results = yolo.predict(frame, classes=[0], conf=0.45, verbose=False)
                h, w, _ = frame.shape
                for box in results[0].boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if (x2 - x1) < 30 or (y2 - y1) < 50:
                        continue
                    crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
                    pose_res = landmarker.detect(mp_img)
                    if pose_res.pose_landmarks:
                        feats = extract_pose_features(pose_res.pose_landmarks[0])
                        rows.append(feats + [label])
                        collected += 1
                        per_video += 1
                        if collected >= TARGET_SAMPLES or per_video >= 25:
                            break
            frame_idx += 1
        cap.release()
    print(f"[{label}] Collected {collected} samples.")

df = pd.DataFrame(rows, columns=FEATURE_COLS + ["label"]).dropna()
df.to_csv(CLEAN_CSV, index=False)

X = df[FEATURE_COLS]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"\nCCTV Normalized Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred))

joblib.dump({"model": clf, "feature_columns": FEATURE_COLS, "classes": list(clf.classes_)}, MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
