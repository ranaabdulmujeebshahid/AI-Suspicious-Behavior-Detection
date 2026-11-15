import os
import sys
import json
import base64
import asyncio
import tempfile
import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pose_utils import extract_pose_features

MODELS_DIR = os.path.join(BASE_DIR, "models")
YOLO_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")
TASK_MODEL_PATH = os.path.join(MODELS_DIR, "pose_landmarker_lite.task")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "behavior_rf_model.pkl")

SEVERITY_META = {
    "Shooting": {"severity": "CRITICAL", "desc": "Active firearm posture detected"},
    "Explosion": {"severity": "CRITICAL", "desc": "Blast reaction / sudden drop detected"},
    "Arson": {"severity": "CRITICAL", "desc": "Suspicious fire-setting activity"},
    "RoadAccidents": {"severity": "CRITICAL", "desc": "Collision / pedestrian impact posture"},
    "Critical_Threat": {"severity": "CRITICAL", "desc": "High-severity threat posture"},
    "Fighting": {"severity": "HIGH", "desc": "Physical altercation in progress"},
    "Assault": {"severity": "HIGH", "desc": "Violent physical strike detected"},
    "Abuse": {"severity": "HIGH", "desc": "Aggressive physical abuse posture"},
    "Robbery": {"severity": "HIGH", "desc": "Forceful confrontation / robbery"},
    "Arrest": {"severity": "HIGH", "desc": "Restraint / struggle posture"},
    "Fighting_Assault": {"severity": "HIGH", "desc": "Physical assault / fighting detected"},
    "Burglary": {"severity": "MEDIUM", "desc": "Forced entry / trespassing posture"},
    "Shoplifting": {"severity": "MEDIUM", "desc": "Concealment / retail theft posture"},
    "Stealing": {"severity": "MEDIUM", "desc": "Unauthorized item snatching"},
    "Vandalism": {"severity": "MEDIUM", "desc": "Property destruction posture"},
    "Robbery_Theft": {"severity": "MEDIUM", "desc": "Theft / unauthorized taking posture"},
    "Normal": {"severity": "SAFE", "desc": "Routine movement"},
}

app = FastAPI(title="AI Suspicious Behavior Detection API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

yolo = YOLO(YOLO_PATH)
base_options = mp_python.BaseOptions(model_asset_path=TASK_MODEL_PATH)
landmarker = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_poses=1)
)

bundle = joblib.load(RF_MODEL_PATH)
clf = bundle["model"]
feature_columns = bundle["feature_columns"]
classes = bundle["classes"]


def process_bgr_frame(frame: np.ndarray, suspicious_threshold: float = 0.62):
    h, w, _ = frame.shape
    results = yolo.predict(frame, classes=[0], conf=0.45, verbose=False)
    detections = []

    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if (x2 - x1) < 24 or (y2 - y1) < 40:
            continue

        crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
        pose_res = landmarker.detect(mp_img)

        if not pose_res.pose_landmarks:
            continue

        landmarks = pose_res.pose_landmarks[0]
        feats = extract_pose_features(landmarks)
        skeleton = [{"x": round(lm.x, 4), "y": round(lm.y, 4)} for lm in landmarks]

        X_df = pd.DataFrame([feats], columns=feature_columns)
        probs = clf.predict_proba(X_df)[0]
        prob_map = dict(zip(classes, probs))

        normal_prob = float(prob_map.get("Normal", 0.0))
        suspicious_prob = 1.0 - normal_prob
        sus_classes = [c for c in classes if c != "Normal"]
        best_sus_label = max(sus_classes, key=lambda c: prob_map[c]) if sus_classes else "Critical_Threat"

        if suspicious_prob >= suspicious_threshold:
            label = best_sus_label
            confidence = suspicious_prob
            is_suspicious = True
        else:
            label = "Normal"
            confidence = normal_prob
            is_suspicious = False

        meta = SEVERITY_META.get(label, {"severity": "HIGH" if is_suspicious else "SAFE", "desc": label})
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "behavior": label,
            "severity": meta["severity"],
            "description": meta["desc"],
            "confidence": round(confidence, 4),
            "alert": is_suspicious,
            "skeleton": skeleton
        })

    return detections


def decode_base64_image(data_url: str):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    img_bytes = base64.b64decode(data_url)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


@app.get("/")
def root():
    return {"status": "online", "version": "2.0.0", "classes": classes}


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


@app.websocket("/ws/surveillance")
async def surveillance_ws(websocket: WebSocket):
    await websocket.accept()
    stream_task = None

    async def run_cctv_stream(stream_url: str):
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            await websocket.send_json({"type": "error", "message": "Unable to open CCTV stream URL. Verify link is public and online."})
            return
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    await websocket.send_json({"type": "error", "message": "CCTV stream ended or disconnected."})
                    break
                h, w = frame.shape[:2]
                if w > 640:
                    scale = 640.0 / w
                    frame = cv2.resize(frame, (640, int(h * scale)))
                detections = process_bgr_frame(frame)
                _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                b64 = base64.b64encode(buf).decode("utf-8")
                await websocket.send_json({
                    "type": "stream_frame",
                    "image": f"data:image/jpeg;base64,{b64}",
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                    "detections": detections
                })
                await asyncio.sleep(0.22)
        except asyncio.CancelledError:
            pass
        finally:
            cap.release()

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            action = msg.get("action")

            if action == "frame":
                frame = decode_base64_image(msg.get("image", ""))
                if frame is not None:
                    detections = process_bgr_frame(frame)
                    await websocket.send_json({
                        "type": "detections",
                        "timestamp": msg.get("timestamp", 0),
                        "width": frame.shape[1],
                        "height": frame.shape[0],
                        "detections": detections
                    })

            elif action == "start_stream":
                if stream_task and not stream_task.done():
                    stream_task.cancel()
                url = msg.get("url", "").strip()
                if url:
                    stream_task = asyncio.create_task(run_cctv_stream(url))

            elif action == "stop_stream":
                if stream_task and not stream_task.done():
                    stream_task.cancel()
                    stream_task = None

    except WebSocketDisconnect:
        if stream_task and not stream_task.done():
            stream_task.cancel()
