import os
import glob
import urllib.request
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

BASE_DIR = "/content/drive/MyDrive/AI_Final_Project/backend"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
TASK_MODEL_PATH = os.path.join(MODELS_DIR, "pose_landmarker_lite.task")
OUTPUT_CSV = os.path.join(DATASET_DIR, "pose_features.csv")

os.makedirs(MODELS_DIR, exist_ok=True)

if not os.path.exists(TASK_MODEL_PATH):
    print("Downloading MediaPipe PoseLandmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    urllib.request.urlretrieve(url, TASK_MODEL_PATH)

yolo = YOLO(os.path.join(MODELS_DIR, "yolov8n.pt"))

base_options = mp_python.BaseOptions(model_asset_path=TASK_MODEL_PATH)
pose_options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1
)
landmarker = vision.PoseLandmarker.create_from_options(pose_options)

CLASS_SOURCES = {
    "Normal": [
        os.path.join(DATASET_DIR, "UCF_Crime/Normal/**/*.mp4"),
        os.path.join(DATASET_DIR, "UCF101/UCF-101/WalkingWithDog/*.avi"),
        os.path.join(DATASET_DIR, "UCF101/UCF-101/TaiChi/*.avi"),
        os.path.join(DATASET_DIR, "UCF101/UCF-101/Typing/*.avi"),
        os.path.join(DATASET_DIR, "UCF101/UCF-101/Biking/*.avi"),
    ],
    "Fighting_Assault": [
        os.path.join(DATASET_DIR, "UCF_Crime/Anomaly/**/Fighting/*.mp4"),
        os.path.join(DATASET_DIR, "UCF_Crime/Anomaly/**/Assault/*.mp4"),
        os.path.join(DATASET_DIR, "UCF_Crime/Anomaly/**/Abuse/*.mp4"),
        os.path.join(DATASET_DIR, "UCF101/UCF-101/Punch/*.avi"),
        os.path.join(DATASET_DIR, "UCF101/UCF-101/SumoWrestling/*.avi"),
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

MAX_VIDEOS_PER_CLASS = 60
FRAME_STRIDE = 15

columns = [f"lm_{i}_{axis}" for i in range(33) for axis in ("x", "y", "z", "vis")] + ["label"]
rows = []

for label, patterns in CLASS_SOURCES.items():
    video_files = []
    for p in patterns:
        video_files.extend(glob.glob(p, recursive=True))
    video_files = sorted(set(video_files))[:MAX_VIDEOS_PER_CLASS]
    print(f"[{label}] Processing {len(video_files)} videos...")

    for idx, vpath in enumerate(video_files, 1):
        cap = cv2.VideoCapture(vpath)
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % FRAME_STRIDE == 0:
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
                        rows.append(feats + [label])
            frame_idx += 1
        cap.release()
        if idx % 15 == 0:
            print(f"  {label}: {idx}/{len(video_files)} videos | Samples collected: {len(rows)}")

df = pd.DataFrame(rows, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(df)} skeleton pose samples to {OUTPUT_CSV}")
print(df["label"].value_counts())
