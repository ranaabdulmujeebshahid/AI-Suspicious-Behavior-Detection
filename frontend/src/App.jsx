import React, { useState, useRef } from "react";

const POSE_CONNECTIONS = [
  [11, 12], [11, 23], [12, 24], [23, 24],
  [11, 13], [13, 15], [12, 14], [14, 16],
  [23, 25], [25, 27], [24, 26], [26, 28],
  [0, 11], [0, 12]
];

export default function App() {
  const [apiUrl, setApiUrl] = useState("http://127.0.0.1:8000");
  const [mode, setMode] = useState("frame");
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const canvasRef = useRef(null);

  const drawFrameWithSkeleton = (imageUrl, detections) => {
    const img = new Image();
    img.onload = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);

      detections.forEach((det) => {
        const [x1, y1, x2, y2] = det.bbox;
        const bw = x2 - x1;
        const bh = y2 - y1;
        const color = det.alert ? "#ef4444" : "#22c55e";

        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, bw, bh);

        ctx.fillStyle = color;
        ctx.fillRect(x1, Math.max(0, y1 - 24), Math.max(170, bw), 24);
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 13px sans-serif";
        ctx.fillText(
          `${det.behavior} (${(det.confidence * 100).toFixed(1)}%)`,
          x1 + 6,
          Math.max(16, y1 - 7)
        );

        if (det.skeleton && det.skeleton.length === 33) {
          ctx.strokeStyle = "#38bdf8";
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
            ctx.arc(x1 + pt.x * bw, y1 + pt.y * bh, 3, 0, Math.PI * 2);
            ctx.fill();
          });
        }
      });
    };
    img.src = imageUrl;
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      const endpoint = mode === "frame" ? "/analyze-frame" : "/analyze-video?frame_stride=15";
      const res = await fetch(`${apiUrl}${endpoint}`, { method: "POST", body: formData });
      if (!res.ok) throw new Error(`Server responded with ${res.status}`);
      const data = await res.json();
      setResult(data);

      if (mode === "frame") {
        drawFrameWithSkeleton(URL.createObjectURL(file), data.detections || []);
      }
    } catch (err) {
      setError(err.message || "Failed to connect to FastAPI server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <header>
        <div>
          <h2>AI Suspicious Behavior Detection</h2>
          <p style={{ color: "var(--muted)", fontSize: "14px" }}>
            Privacy-Preserving CCTV Skeleton Pose Surveillance
          </p>
        </div>
        <span className="badge">Privacy Mode: Skeleton-Only (No Facial Recognition)</span>
      </header>

      <div className="controls">
        <div className="card">
          <h3>1. FastAPI Backend URL</h3>
          <input type="text" value={apiUrl} onChange={(e) => setApiUrl(e.target.value)} />
        </div>

        <div className="card">
          <h3>2. Select Media & Analyze</h3>
          <div className="tabs">
            <button
              className={mode === "frame" ? "" : "secondary"}
              onClick={() => { setMode("frame"); setResult(null); }}
            >
              CCTV Frame (Image)
            </button>
            <button
              className={mode === "video" ? "" : "secondary"}
              onClick={() => { setMode("video"); setResult(null); }}
            >
              CCTV Clip (Video)
            </button>
          </div>
          <input
            type="file"
            accept={mode === "frame" ? "image/*" : "video/*"}
            onChange={(e) => setFile(e.target.files[0] || null)}
          />
          <button onClick={handleAnalyze} disabled={!file || loading}>
            {loading ? "Analyzing Pose Skeletons..." : "Run Behavior Detection"}
          </button>
        </div>
      </div>

      {error && <div className="alert-item danger">{error}</div>}

      <div className="workspace">
        <div className="card">
          <h3>Skeleton Overlay Preview</h3>
          {mode === "frame" ? (
            <canvas ref={canvasRef} />
          ) : (
            <p style={{ color: "var(--muted)", fontSize: "14px" }}>
              Video timeline alerts appear on the right with exact timestamps and confidence scores.
            </p>
          )}
        </div>

        <div className="card">
          <h3>Detection Report</h3>
          {!result && <p style={{ color: "var(--muted)", fontSize: "14px" }}>Upload a frame or video to view results.</p>}

          {result && mode === "frame" && (
            <div>
              <p style={{ marginBottom: "10px" }}>Persons Detected: <strong>{result.persons_detected}</strong></p>
              {result.detections.map((d, i) => (
                <div key={i} className={`alert-item ${d.alert ? "danger" : ""}`}>
                  <strong>{d.behavior}</strong> — Confidence: {(d.confidence * 100).toFixed(1)}%
                  <div>Status: {d.alert ? "SUSPICIOUS ALERT" : "Normal Activity"}</div>
                </div>
              ))}
            </div>
          )}

          {result && mode === "video" && (
            <div>
              <p style={{ marginBottom: "10px" }}>
                Frames Processed: <strong>{result.total_frames_processed}</strong> | Suspicious Events:{" "}
                <strong>{result.suspicious_events_count}</strong>
              </p>
              {result.timeline.length === 0 && (
                <div className="alert-item">No suspicious behavior detected.</div>
              )}
              {result.timeline.map((ev, idx) => (
                <div key={idx} className="alert-item danger">
                  <strong>Timestamp: {ev.timestamp_sec}s</strong> (Frame {ev.frame})
                  {ev.alerts.map((a, j) => (
                    <div key={j}>{a.behavior} — {(a.confidence * 100).toFixed(1)}%</div>
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
