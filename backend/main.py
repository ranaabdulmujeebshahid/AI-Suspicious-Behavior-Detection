import os
import tempfile
import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
YOLO_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")
TASK_MODEL_PATH = os.path.join(MODELS_DIR, "pose_landmarker_lite.task")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "behavior_rf_model.pkl")

app = FastAPI(title="AI Suspicious Behavior Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

yolo = YOLO(YOLO_PATH)

base_options = mp_python.BaseOptions(model_asset_path=TASK_MODEL_PATH)
pose_options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1
)
landmarker = vision.PoseLandmarker.create_from_options(pose_options)

bundle = joblib.load(RF_MODEL_PATH)
clf = bundle["model"]
feature_columns = bundle["feature_columns"]
classes = bundle["classes"]


def process_bgr_frame(frame: np.ndarray, conf_threshold: float = 0.60):
    h, w, _ = frame.shape
    results = yolo.predict(frame, classes=[0], conf=0.4, verbose=False)
    detections = []

    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if (x2 - x1) < 30 or (y2 - y1) < 50:
            continue

        crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
        pose_res = landmarker.detect(mp_img)

        if not pose_res.pose_landmarks:
            continue

        feats = []
        skeleton = []
        for lm in pose_res.pose_landmarks[0]:
            feats.extend([lm.x, lm.y, lm.z, float(lm.visibility or 0.0)])
            skeleton.append({"x": round(lm.x, 4), "y": round(lm.y, 4)})

        X_df = pd.DataFrame([feats], columns=feature_columns)
        probs = clf.predict_proba(X_df)[0]
        best_idx = int(np.argmax(probs))
        label = str(classes[best_idx])
        confidence = float(probs[best_idx])
        is_suspicious = (label != "Normal") and (confidence >= conf_threshold)

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "behavior": label,
            "confidence": round(confidence, 4),
            "alert": is_suspicious,
            "skeleton": skeleton
        })

    return detections


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "privacy_mode": "skeleton_only_no_facial_recognition",
        "classes": classes
    }


@app.post("/analyze-frame")
async def analyze_frame(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    detections = process_bgr_frame(frame)
    return {
        "persons_detected": len(detections),
        "alert_triggered": any(d["alert"] for d in detections),
        "detections": detections
    }


@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...), frame_stride: int = 15):
    suffix = os.path.splitext(file.filename or "upload.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_idx = 0
        timeline = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_stride == 0:
                detections = process_bgr_frame(frame)
                alerts = [d for d in detections if d["alert"]]
                if alerts:
                    timeline.append({
                        "frame": frame_idx,
                        "timestamp_sec": round(frame_idx / fps, 2),
                        "alerts": alerts
                    })
            frame_idx += 1
        cap.release()

        return {
            "filename": file.filename,
            "total_frames_processed": frame_idx,
            "suspicious_events_count": len(timeline),
            "timeline": timeline
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
