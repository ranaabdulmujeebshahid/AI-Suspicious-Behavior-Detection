set -e

cd /content/drive/MyDrive/AI_Final_Project

echo "[1/4] Checking Python dependencies..."
python3 -c "import fastapi, uvicorn, mediapipe, ultralytics, sklearn, joblib" 2>/dev/null || python3 -m pip install -q -r backend/requirements.txt

if [ ! -f /usr/local/bin/cloudflared ]; then
  echo "[2/4] Installing cloudflared..."
  wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
  chmod +x /usr/local/bin/cloudflared
fi

echo "[3/4] Preparing frontend..."
pkill -f "uvicorn backend.main:app" || true
pkill -f "vite" || true
pkill -f "cloudflared" || true

rm -rf /tmp/frontend && cp -r frontend /tmp/frontend
cd /tmp/frontend && npm install --silent

echo "[4/4] Starting FastAPI, React, and HTTPS tunnels..."
cd /content/drive/MyDrive/AI_Final_Project
nohup python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 > /tmp/fastapi.log 2>&1 &

cd /tmp/frontend
nohup npm run dev > /tmp/vite.log 2>&1 &

nohup cloudflared tunnel --url http://127.0.0.1:8000 > /tmp/cf_api.log 2>&1 &
nohup cloudflared tunnel --url http://127.0.0.1:5173 > /tmp/cf_web.log 2>&1 &

sleep 10
echo ""
echo "=========================================================="
echo "1. FastAPI Backend URL : $(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /tmp/cf_api.log | head -n 1)"
echo "2. React Dashboard URL : $(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /tmp/cf_web.log | head -n 1)"
echo "=========================================================="
