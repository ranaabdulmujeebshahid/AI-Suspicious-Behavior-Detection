#!/bin/bash
set -e

cd /content/drive/MyDrive/AI_Final_Project

echo "[1/5] Checking dependencies..."
python3 -c "import fastapi, uvicorn, websockets, mediapipe, ultralytics, sklearn, joblib" 2>/dev/null || python3 -m pip install -q -r backend/requirements.txt websockets

if [ ! -f /usr/local/bin/cloudflared ]; then
  wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
  chmod +x /usr/local/bin/cloudflared
fi

pkill -9 -f "uvicorn backend.main:app" 2>/dev/null || true
pkill -9 -f "vite" 2>/dev/null || true
pkill -9 -f "cloudflared" 2>/dev/null || true
rm -f /tmp/cf_api.log /tmp/cf_web.log /tmp/fastapi.log /tmp/vite.log

echo "[2/5] Starting FastAPI Backend..."
nohup python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 > /tmp/fastapi.log 2>&1 &
for i in {1..20}; do
  if curl -s http://127.0.0.1:8000/health >/dev/null; then break; fi
  sleep 1
done

echo "[3/5] Starting Backend Tunnel..."
nohup cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8000 > /tmp/cf_api.log 2>&1 &
API_URL=""
for i in {1..20}; do
  API_URL=$(grep -ao 'https://[a-z0-9-]*\.trycloudflare\.com' /tmp/cf_api.log | head -n 1 || true)
  if [ -n "$API_URL" ]; then break; fi
  sleep 1
done

echo "[4/5] Injecting Backend URL ($API_URL) into React..."
rm -rf /tmp/frontend && cp -r frontend /tmp/frontend
echo "VITE_API_URL=$API_URL" > /tmp/frontend/.env
cd /tmp/frontend && npm install --silent

echo "[5/5] Starting SOC Dashboard & Public Tunnel..."
nohup npm run dev > /tmp/vite.log 2>&1 &
nohup cloudflared tunnel --no-autoupdate --url http://127.0.0.1:5173 > /tmp/cf_web.log 2>&1 &

WEB_URL=""
for i in {1..20}; do
  WEB_URL=$(grep -ao 'https://[a-z0-9-]*\.trycloudflare\.com' /tmp/cf_web.log | head -n 1 || true)
  if [ -n "$WEB_URL" ]; then break; fi
  sleep 1
done

sleep 5
echo ""
echo "=========================================================="
echo "1. Backend API  : $API_URL"
echo "2. HUD Dashboard: $WEB_URL"
echo "=========================================================="
