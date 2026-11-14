import os
import glob
import cv2
import joblib
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = "/content/drive/MyDrive/AI_Final_Project/backend"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RAW_CSV = os.path.join(DATASET_DIR, "pose_features.csv")
BALANCED_CSV = os.path.join(DATASET_DIR, "pose_features_balanced.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "behavior_rf_model.pkl")

yolo = YOLO(os.path.join(MODELS_DIR, "yolov8n.pt"))
base_options = mp_python.BaseOptions(model_asset_path=os.path.join(MODELS_DIR, "pose_landmarker_lite.task"))
landmarker = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_poses=1)
)

df_existing = pd.read_csv(RAW_CSV).dropna()
feature_cols = [c for c in df_existing.columns if c != "label"]

TARGET_PER_CLASS = 1000
EXTRA_SOURCES = {
    "Normal": [
        os.path.join(DATASET_DIR, "UCF101/UCF-101/WalkingWithDog/*.avi"),
        os.path.join(DATASET_DIR, "UCF101/UCF-101/TaiChi/*.avi"),
        os.path.join(DATASET_DIR, "UCF101/UCF-101/Typing/*.avi"),
        os.path.join(DATASET_DIR, "UCF101/UCF-101/Biking/*.avi"),
        os.path.join(DATASET_DIR, "UCF101/UCF-101/MoppingFloor/*.avi"),
        os.path.join(DATASET_DIR, "UCF101/UCF-101/WritingOnBoard/*.avi"),
        os.path.join(DATASET_DIR, "UCF101/UCF-101/PlayingPiano/*.avi"),
        os.path.join(DATASET_DIR, "UCF101/UCF-101/BrushingTeeth/*.avi"),
        os.path.join(DATASET_DIR, "UCF101/UCF-101/Bowling/*.avi"),
    ],
    "Fighting_Assault": [
        os.path.join(DATASET_DIR, "UCF101/UCF-101/Punch/*.avi"),
        os.path.join(DATASET_DIR, "UCF101/UCF-101/SumoWrestling/*.avi"),
        os.path.join(DATASET_DIR, "UCF101/UCF-101/BoxingPunchingBag/*.avi"),
        os.path.join(DATASET_DIR, "UCF101/UCF-101/Fencing/*.avi"),
        os.path.join(DATASET_DIR, "UCF_Crime/Anomaly/**/Fighting/*.mp4"),
        os.path.join(DATASET_DIR, "UCF_Crime/Anomaly/**/Assault/*.mp4"),
    ],
    "Critical_Threat": [
        os.path.join(DATASET_DIR, "UCF_Crime/Anomaly/**/Shooting/*.mp4"),
        os.path.join(DATASET_DIR, "UCF_Crime/Anomaly/**/Arson/*.mp4"),
        os.path.join(DATASET_DIR, "UCF_Crime/Anomaly/**/Explosion/*.mp4"),
        os.path.join(DATASET_DIR, "UCF_Crime/Anomaly/**/Vandalism/*.mp4"),
    ],
}

balanced_parts = []
for label in ["Normal", "Fighting_Assault", "Robbery_Theft", "Critical_Threat"]:
    sub = df_existing[df_existing["label"] == label]
    if len(sub) >= TARGET_PER_CLASS:
        balanced_parts.append(sub.sample(n=TARGET_PER_CLASS, random_state=42))
        print(f"[{label}] Downsampled to {TARGET_PER_CLASS} samples.")
        continue

    needed = TARGET_PER_CLASS - len(sub)
    print(f"[{label}] Have {len(sub)}, collecting {needed} more samples...")
    new_rows = []
    videos = []
    for p in EXTRA_SOURCES.get(label, []):
        videos.extend(glob.glob(p, recursive=True))

    for vpath in sorted(set(videos)):
        if len(new_rows) >= needed:
            break
        cap = cv2.VideoCapture(vpath)
        frame_idx = 0
        while cap.isOpened() and len(new_rows) < needed:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 10 == 0:
                results = yolo.predict(frame, classes=[0], conf=0.4, verbose=False)
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
                        feats = []
                        for lm in pose_res.pose_landmarks[0]:
                            feats.extend([lm.x, lm.y, lm.z, float(lm.visibility or 0.0)])
                        new_rows.append(feats + [label])
                        if len(new_rows) >= needed:
                            break
            frame_idx += 1
        cap.release()

    df_new = pd.DataFrame(new_rows, columns=feature_cols + ["label"])
    combined = pd.concat([sub, df_new], ignore_index=True)
    balanced_parts.append(combined)
    print(f"[{label}] Total balanced samples: {len(combined)}")

df_balanced = pd.concat(balanced_parts, ignore_index=True)
df_balanced.to_csv(BALANCED_CSV, index=False)

X = df_balanced[feature_cols]
y = df_balanced["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"\nBalanced Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=clf.classes_))

joblib.dump({"model": clf, "feature_columns": feature_cols, "classes": list(clf.classes_)}, MODEL_PATH)
print(f"\nUpdated model saved to {MODEL_PATH}")
