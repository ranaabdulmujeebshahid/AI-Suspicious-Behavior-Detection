# AI-Powered Suspicious Behavior Detection System

A privacy-preserving CCTV surveillance system that detects suspicious human activity using **skeleton-based pose estimation** rather than facial recognition or raw biometric tracking.

## Architecture & Pipeline
1. **Person Detection (`YOLOv8n`)**: Localizes human bounding boxes in CCTV frames.
2. **Privacy-Safe Pose Extraction (`MediaPipe PoseLandmarker`)**: Extracts 33 body landmarks (`x, y, z, visibility`), normalizes them relative to hip center and torso scale, and computes 7 joint angles—discarding facial identity.
3. **Behavior Classifier (`Scikit-Learn Ensemble`)**: Classifies normalized pose vectors into 4 categories:
   - `Normal` (everyday walking, standing, shopping)
   - `Fighting_Assault` (Fighting, Assault, Abuse)
   - `Robbery_Theft` (Robbery, Burglary, Stealing, Shoplifting)
   - `Critical_Threat` (Shooting, Arson, Explosion, Vandalism)
4. **Inference API (`FastAPI`)**: Exposes `/health`, `/analyze-frame`, and `/analyze-video`.
5. **Security Dashboard (`React + Vite`)**: Displays real-time skeleton overlays on images and clickable incident timestamps on CCTV videos.

## Quick Start (Google Colab)
```bash
cd /content/drive/MyDrive/AI_Final_Project
bash start_app.sh
