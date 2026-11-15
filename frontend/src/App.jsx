import React, { useState, useRef, useEffect } from "react";

const API_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");
const WS_URL = API_URL.replace(/^http/, "ws") + "/ws/surveillance";

const POSE_CONNECTIONS = [
  [11, 12], [11, 23], [12, 24], [23, 24],
  [11, 13], [13, 15], [12, 14], [14, 16],
  [23, 25], [25, 27], [24, 26], [26, 28],
  [0, 11], [0, 12]
];

export default function App() {
  const [backendOnline, setBackendOnline] = useState(false);
  const [mode, setMode] = useState("video"); // "video" | "stream" | "frame"
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [alarmActive, setAlarmActive] = useState(false);
  const [toasts, setToasts] = useState([]);
  const [incidents, setIncidents] = useState([]);
  const [personsCount, setPersonsCount] = useState(0);
  const [threatLevel, setThreatLevel] = useState("SAFE");

  // Media states
  const [mediaUrl, setMediaUrl] = useState("");
  const [streamUrl, setStreamUrl] = useState("");
  const [streamActive, setStreamActive] = useState(false);
  const [streamFrameSrc, setStreamFrameSrc] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [statusText, setStatusText] = useState("Ready");

  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const wsRef = useRef(null);
  const loopTimerRef = useRef(null);
  const streakRef = useRef(0);
  const lastAlarmTimeRef = useRef({});
  const audioCtxRef = useRef(null);

  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then((r) => setBackendOnline(r.ok))
      .catch(() => setBackendOnline(false));
    return () => stopRealtimeEngine();
  }, []);

  const unlockAudio = () => {
    if (!audioCtxRef.current) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (Ctx) audioCtxRef.current = new Ctx();
    }
    if (audioCtxRef.current && audioCtxRef.current.state === "suspended") {
      audioCtxRef.current.resume();
    }
  };

  const playSiren = (severity) => {
    if (!soundEnabled) return;
    unlockAudio();
    const ctx = audioCtxRef.current;
    if (!ctx) return;

    const now = ctx.currentTime;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = severity === "CRITICAL" ? "sawtooth" : "triangle";

    if (severity === "CRITICAL") {
      osc.frequency.setValueAtTime(880, now);
      osc.frequency.linearRampToValueAtTime(1320, now + 0.18);
      osc.frequency.linearRampToValueAtTime(880, now + 0.36);
      gain.gain.setValueAtTime(0.18, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.42);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(now);
      osc.stop(now + 0.43);
    } else {
      osc.frequency.setValueAtTime(740, now);
      osc.frequency.setValueAtTime(980, now + 0.14);
      gain.gain.setValueAtTime(0.12, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.3);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(now);
      osc.stop(now + 0.31);
    }
  };

  const triggerRuntimeAlarm = (alertDet, timestampSec, snapshotDataUrl) => {
    const now = Date.now();
    const lastFired = lastAlarmTimeRef.current[alertDet.behavior] || 0;
    if (now - lastFired < 3200) return;
    lastAlarmTimeRef.current[alertDet.behavior] = now;

    setAlarmActive(true);
    setThreatLevel(alertDet.severity || "HIGH");
    playSiren(alertDet.severity || "HIGH");

    const id = `${now}-${Math.random().toString(36).slice(2, 6)}`;
    const toastObj = {
      id,
      behavior: alertDet.behavior,
      severity: alertDet.severity || "HIGH",
      description: alertDet.description || "Suspicious posture detected",
      confidence: alertDet.confidence,
      timestamp: timestampSec,
    };

    setToasts((prev) => [toastObj, ...prev].slice(0, 3));
    setIncidents((prev) => [{ ...toastObj, thumb: snapshotDataUrl, clock: new Date().toLocaleTimeString() }, ...prev].slice(0, 30));

    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4500);

    setTimeout(() => {
      setAlarmActive(false);
    }, 2600);
  };

  const drawHudDetections = (detections, srcW, srcH, clearFirst = true) => {
    const canvas = overlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;

    canvas.width = rect.width;
    canvas.height = rect.height;
    const ctx = canvas.getContext("2d");
    if (clearFirst) ctx.clearRect(0, 0, canvas.width, canvas.height);

    const scaleX = canvas.width / srcW;
    const scaleY = canvas.height / srcH;

    detections.forEach((det) => {
      const [rx1, ry1, rx2, ry2] = det.bbox;
      const x1 = rx1 * scaleX;
      const y1 = ry1 * scaleY;
      const bw = (rx2 - rx1) * scaleX;
      const bh = (ry2 - ry1) * scaleY;
      const color = det.alert ? (det.severity === "MEDIUM" ? "#f59e0b" : "#ef4444") : "#22c55e";

      ctx.strokeStyle = color;
      ctx.lineWidth = det.alert ? 3 : 2;
      ctx.strokeRect(x1, y1, bw, bh);

      const labelText = `${det.behavior} ${(det.confidence * 100).toFixed(0)}%`;
      ctx.font = "bold 12px Inter, sans-serif";
      const textW = ctx.measureText(labelText).width + 12;
      ctx.fillStyle = color;
      ctx.fillRect(x1, Math.max(0, y1 - 22), Math.max(textW, 95), 22);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(labelText, x1 + 6, Math.max(15, y1 - 6));

      if (det.skeleton && det.skeleton.length === 33) {
        ctx.strokeStyle = det.alert ? "#fca5a5" : "#38bdf8";
        ctx.lineWidth = 2;
        POSE_CONNECTIONS.forEach(([i, j]) => {
          const p1 = det.skeleton[i];
          const p2 = det.skeleton[j];
          ctx.beginPath();
          ctx.moveTo(x1 + p1.x * bw, y1 + p1.y * bh);
          ctx.lineTo(x1 + p2.x * bw, y1 + p2.y * bh);
          ctx.stroke();
        });

        ctx.fillStyle = "#facc15";
        det.skeleton.forEach((pt) => {
          ctx.beginPath();
          ctx.arc(x1 + pt.x * bw, y1 + pt.y * bh, 2.5, 0, Math.PI * 2);
          ctx.fill();
        });
      }
    });
  };

  const evaluateDetections = (detections, timestampSec, snapshotDataUrl) => {
    setPersonsCount(detections.length);
    const alerts = detections.filter((d) => d.alert);
    if (alerts.length > 0) {
      streakRef.current += 1;
      if (streakRef.current >= 2) {
        const topAlert = alerts.sort((a, b) => b.confidence - a.confidence)[0];
        triggerRuntimeAlarm(topAlert, timestampSec, snapshotDataUrl);
      }
    } else {
      streakRef.current = 0;
      if (!alarmActive) setThreatLevel("SAFE");
    }
  };

  const stopRealtimeEngine = () => {
    if (loopTimerRef.current) {
      clearInterval(loopTimerRef.current);
      loopTimerRef.current = null;
    }
    if (wsRef.current) {
      try {
        wsRef.current.send(JSON.stringify({ action: "stop_stream" }));
        wsRef.current.close();
      } catch {}
      wsRef.current = null;
    }
    setStreamActive(false);
    streakRef.current = 0;
  };

  // Start WebSocket when video starts playing
  const handleVideoPlay = () => {
    unlockAudio();
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;
    let busy = false;

    ws.onopen = () => {
      setStatusText("Live HUD Tracking Active");
      loopTimerRef.current = setInterval(() => {
        const vid = videoRef.current;
        if (!vid || vid.paused || vid.ended || busy || ws.readyState !== WebSocket.OPEN) return;

        const targetW = Math.min(640, vid.videoWidth || 640);
        const scale = targetW / (vid.videoWidth || 640);
        const targetH = Math.round((vid.videoHeight || 360) * scale);

        const cap = captureCanvasRef.current;
        cap.width = targetW;
        cap.height = targetH;
        cap.getContext("2d").drawImage(vid, 0, 0, targetW, targetH);
        const dataUrl = cap.toDataURL("image/jpeg", 0.7);
        cap.dataset.lastFrame = dataUrl;

        busy = true;
        ws.send(JSON.stringify({ action: "frame", image: dataUrl, timestamp: Number(vid.currentTime.toFixed(2)) }));
      }, 240);
    };

    ws.onmessage = (ev) => {
      busy = false;
      const msg = JSON.parse(ev.data);
      if (msg.type === "detections") {
        drawHudDetections(msg.detections, msg.width, msg.height);
        const snap = captureCanvasRef.current?.dataset?.lastFrame || "";
        evaluateDetections(msg.detections, msg.timestamp, snap);
      }
    };

    ws.onclose = () => {
      if (loopTimerRef.current) clearInterval(loopTimerRef.current);
      setStatusText("Playback Paused");
    };
  };

  const handleVideoPauseOrEnd = () => {
    if (loopTimerRef.current) {
      clearInterval(loopTimerRef.current);
      loopTimerRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setStatusText("Paused");
  };

  // Start Live Public CCTV URL Stream
  const toggleLiveStream = () => {
    unlockAudio();
    if (streamActive) {
      stopRealtimeEngine();
      setStatusText("Stream Stopped");
      return;
    }
    if (!streamUrl.trim()) return;

    setStatusText("Connecting to CCTV Feed...");
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setStreamActive(true);
      setStatusText("LIVE CCTV FEED");
      ws.send(JSON.stringify({ action: "start_stream", url: streamUrl.trim() }));
    };

    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "stream_frame") {
        setStreamFrameSrc(msg.image);
        drawHudDetections(msg.detections, msg.width, msg.height);
        evaluateDetections(msg.detections, null, msg.image);
      } else if (msg.type === "error") {
        setStatusText(msg.message);
        stopRealtimeEngine();
      }
    };

    ws.onerror = () => {
      setStatusText("WebSocket connection error.");
      stopRealtimeEngine();
    };
  };

  // Analyze Single Image Snapshot
  const handleAnalyzeImage = async () => {
    if (!selectedFile) return;
    unlockAudio();
    setStatusText("Scanning Frame...");
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res = await fetch(`${API_URL}/analyze-frame`, { method: "POST", body: formData });
      const data = await res.json();
      const img = new Image();
      img.onload = () => {
        drawHudDetections(data.detections || [], img.width, img.height);
        const alerts = (data.detections || []).filter((d) => d.alert);
        setPersonsCount(data.persons_detected || 0);
        if (alerts.length > 0) {
          triggerRuntimeAlarm(alerts[0], null, mediaUrl);
        } else {
          setThreatLevel("SAFE");
        }
        setStatusText(`Scanned (${data.persons_detected} person${data.persons_detected === 1 ? "" : "s"})`);
      };
      img.src = mediaUrl;
    } catch {
      setStatusText("Failed to analyze image");
    }
  };

  const switchMode = (newMode) => {
    stopRealtimeEngine();
    setMode(newMode);
    setMediaUrl("");
    setSelectedFile(null);
    setStreamFrameSrc("");
    setPersonsCount(0);
    setThreatLevel("SAFE");
    setStatusText("Ready");
    const canvas = overlayRef.current;
    if (canvas) canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  };

  const handleFilePick = (e) => {
    stopRealtimeEngine();
    const f = e.target.files[0];
    if (!f) return;
    setSelectedFile(f);
    setMediaUrl(URL.createObjectURL(f));
    setStatusText(mode === "video" ? "Press Play for Live Detection" : "Ready to Scan");
  };

  return (
    <div className={`app-shell ${alarmActive ? "alarm-active" : ""}`}>
      {/* Emergency Police/SOC Lightbar */}
      <div className={`siren-bar ${alarmActive ? "strobe" : ""}`}>
        <div className="siren-led red" />
        <div className="siren-led blue" />
        <div className="siren-led amber" />
        <div className="siren-led amber" />
        <div className="siren-led blue" />
        <div className="siren-led red" />
      </div>

      {/* Fancy Slide-in Event Popups */}
      <div className="toast-stack">
        {toasts.map((t) => (
          <div key={t.id} className={`alarm-toast ${t.severity}`}>
            <div className="toast-top">
              <span className={`sev-tag ${t.severity}`}>{t.severity} ALERT</span>
              <button className="toast-close" onClick={() => setToasts((prev) => prev.filter((x) => x.id !== t.id))}>
                ×
              </button>
            </div>
            <div style={{ fontWeight: 800, fontSize: "15px", marginBottom: "2px" }}>{t.behavior}</div>
            <div style={{ fontSize: "12px", color: "#cbd5e1" }}>
              {t.description} • <strong>{(t.confidence * 100).toFixed(1)}%</strong>
              {t.timestamp !== null ? ` @ ${t.timestamp}s` : ""}
            </div>
          </div>
        ))}
      </div>

      <div className="container">
        <header className="soc-header">
          <div className="brand">
            <span className={`radar-dot ${alarmActive ? "alert" : ""}`} />
            <div>
              <h2 style={{ fontSize: "18px", fontWeight: 800 }}>AEGIS SURVEILLANCE HUD</h2>
              <p style={{ fontSize: "12px", color: "var(--muted)" }}>Real-Time Skeleton Behavior & Threat Detection</p>
            </div>
          </div>

          <div className="header-pills">
            <span className="pill privacy">SKELETON-ONLY PRIVACY</span>
            <span className={`pill ${backendOnline ? "online" : "offline"}`}>
              {backendOnline ? "● AI ENGINE ONLINE" : "● ENGINE OFFLINE"}
            </span>
            <button
              className={`siren-btn ${soundEnabled ? "unmuted" : ""}`}
              onClick={() => {
                unlockAudio();
                setSoundEnabled(!soundEnabled);
              }}
            >
              {soundEnabled ? "🔊 SIREN ARMED" : "🔇 SIREN MUTED"}
            </button>
          </div>
        </header>

        <div className="grid-main">
          {/* Left Surveillance Monitor */}
          <div className="panel">
            <div className="mode-tabs">
              <button className={`tab-btn ${mode === "video" ? "active" : ""}`} onClick={() => switchMode("video")}>
                ▶ Live Video Playback
              </button>
              <button className={`tab-btn ${mode === "stream" ? "active" : ""}`} onClick={() => switchMode("stream")}>
                📡 Public CCTV Feed URL
              </button>
              <button className={`tab-btn ${mode === "frame" ? "active" : ""}`} onClick={() => switchMode("frame")}>
                🖼 Frame Snapshot
              </button>
            </div>

            {mode === "video" && (
              <div className="input-bar">
                <input type="file" accept="video/*" onChange={handleFilePick} />
              </div>
            )}

            {mode === "stream" && (
              <div className="input-bar">
                <input
                  type="text"
                  placeholder="Paste public CCTV URL (.m3u8 HLS, MJPEG, or RTSP)..."
                  value={streamUrl}
                  onChange={(e) => setStreamUrl(e.target.value)}
                />
                <button
                  className={`action-btn ${streamActive ? "danger" : ""}`}
                  onClick={toggleLiveStream}
                  disabled={!streamUrl.trim()}
                >
                  {streamActive ? "Stop Feed" : "Connect Live Feed"}
                </button>
              </div>
            )}

            {mode === "frame" && (
              <div className="input-bar">
                <input type="file" accept="image/*" onChange={handleFilePick} />
                <button className="action-btn" onClick={handleAnalyzeImage} disabled={!selectedFile}>
                  Scan Frame
                </button>
              </div>
            )}

            <div className="hud-stage">
              <span className="hud-badge">{statusText}</span>

              {mode === "video" && mediaUrl && (
                <video
                  ref={videoRef}
                  src={mediaUrl}
                  controls
                  onPlay={handleVideoPlay}
                  onPause={handleVideoPauseOrEnd}
                  onEnded={handleVideoPauseOrEnd}
                />
              )}

              {mode === "stream" && streamFrameSrc && (
                <img className="stream-img" src={streamFrameSrc} alt="Live CCTV Stream" />
              )}

              {mode === "frame" && mediaUrl && (
                <img className="stream-img" src={mediaUrl} alt="CCTV Snapshot" />
              )}

              {!mediaUrl && !streamFrameSrc && (
                <p style={{ color: "var(--muted)", fontSize: "13px" }}>
                  {mode === "video" && "Select a CCTV clip and hit Play for live real-time skeleton tracking & alarms."}
                  {mode === "stream" && "Enter a live public CCTV stream URL above to begin real-time monitoring."}
                  {mode === "frame" && "Upload a CCTV image to inspect detected skeletons."}
                </p>
              )}

              <canvas ref={overlayRef} className="hud-overlay" />
            </div>
            <canvas ref={captureCanvasRef} style={{ display: "none" }} />
          </div>

          {/* Right Live Telemetry & Incident Log */}
          <div className="panel">
            <div className="metrics-row">
              <div className="metric-box">
                <div className="metric-label">Threat Status</div>
                <div
                  className="metric-val"
                  style={{
                    color:
                      threatLevel === "CRITICAL"
                        ? "var(--critical)"
                        : threatLevel === "HIGH"
                        ? "var(--high)"
                        : threatLevel === "MEDIUM"
                        ? "var(--medium)"
                        : "var(--safe)",
                  }}
                >
                  {threatLevel}
                </div>
              </div>
              <div className="metric-box">
                <div className="metric-label">Tracked Subjects</div>
                <div className="metric-val">{personsCount}</div>
              </div>
              <div className="metric-box">
                <div className="metric-label">Alarms Logged</div>
                <div className="metric-val">{incidents.length}</div>
              </div>
            </div>

            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" }}>
              <h3 style={{ fontSize: "14px", fontWeight: 700 }}>RUNTIME INCIDENT LOG</h3>
              {incidents.length > 0 && (
                <button className="tab-btn" style={{ padding: "4px 10px", fontSize: "11px" }} onClick={() => setIncidents([])}>
                  Clear Log
                </button>
              )}
            </div>

            <div className="incident-list">
              {incidents.length === 0 && (
                <p style={{ color: "var(--muted)", fontSize: "13px" }}>
                  No runtime alarms triggered yet. Play a video or connect a live feed to monitor.
                </p>
              )}

              {incidents.map((inc) => (
                <div
                  key={inc.id}
                  className={`incident-card ${inc.severity}`}
                  onClick={() => {
                    if (mode === "video" && videoRef.current && inc.timestamp !== null) {
                      videoRef.current.currentTime = inc.timestamp;
                    }
                  }}
                >
                  {inc.thumb && <img className="incident-thumb" src={inc.thumb} alt="Incident snapshot" />}
                  <div style={{ flex: 1 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <strong style={{ fontSize: "13px" }}>{inc.behavior}</strong>
                      <span className={`sev-tag ${inc.severity}`}>{inc.severity}</span>
                    </div>
                    <div style={{ fontSize: "12px", color: "var(--muted)", marginTop: "2px" }}>
                      Confidence: {(inc.confidence * 100).toFixed(1)}% • {inc.clock}
                      {inc.timestamp !== null ? ` (${inc.timestamp}s)` : ""}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
