import React, { useEffect, useRef, useState } from "react";

export default function ObjectDetectionStream() {
  const videoRef = useRef(null);
  const canvasRef = useRef(document.createElement("canvas"));
  const wsRef = useRef(null);
  const streamIntervalRef = useRef(null);
  const lastFrameTime = useRef(Date.now());

  const [connected, setConnected] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [fps, setFps] = useState(0);
  const [targetFound, setTargetFound] = useState(false);
  const [target, setTarget] = useState(""); // Start empty
  const [detectionLog, setDetectionLog] = useState([]);
  const [availableClasses, setAvailableClasses] = useState([]);

  // WebSocket setup
  useEffect(() => {
    const ws = new WebSocket("wss://webcam.aavashlamichhane.com.np/ws/stream");
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      ws.send(JSON.stringify({ type: "get_classes" }));
    };

    ws.onclose = () => setConnected(false);
    ws.onerror = (err) => console.error("WebSocket error:", err);

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "detection_result") {
          setTargetFound(data.target_detected);
          const now = new Date().toLocaleTimeString();
          setDetectionLog((prev) => [
            { time: now, detected: data },
            ...prev.slice(0, 9),
          ]);
        } else if (data.type === "available_classes") {
          setAvailableClasses(data.classes || []);
        }
      } catch (err) {
        console.warn("WS message parse error:", err);
      }
    };

    return () => ws.close();
  }, []);

  const startStream = async () => {
    if (streaming) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "environment" },
      });
      videoRef.current.srcObject = stream;
      setStreaming(true);
      setFrameCount(0);
      const ctx = canvasRef.current.getContext("2d");

      streamIntervalRef.current = setInterval(() => {
        const video = videoRef.current;
        if (!video || !wsRef.current || wsRef.current.readyState !== 1) return;

        canvasRef.current.width = video.videoWidth;
        canvasRef.current.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);

        canvasRef.current.toBlob(
          (blob) => {
            const reader = new FileReader();
            reader.onloadend = () => {
              wsRef.current.send(reader.result);
              const now = Date.now();
              const newFps = Math.round(1000 / (now - lastFrameTime.current));
              lastFrameTime.current = now;
              setFps(newFps);
              setFrameCount((count) => count + 1);
            };
            reader.readAsDataURL(blob);
          },
          "image/jpeg",
          0.8
        );
      }, 100);
    } catch (err) {
      console.error("Camera access error:", err);
    }
  };

  const stopStream = () => {
    setStreaming(false);
    if (streamIntervalRef.current) {
      clearInterval(streamIntervalRef.current);
      streamIntervalRef.current = null;
    }

    const tracks = videoRef.current?.srcObject?.getTracks();
    tracks?.forEach((track) => track.stop());
    videoRef.current.srcObject = null;
  };

  const sendTarget = (newTarget) => {
    if (wsRef.current?.readyState === 1 && newTarget) {
      wsRef.current.send(
        JSON.stringify({ type: "set_target", target: newTarget })
      );
    }
  };

  const startVoiceRecognition = () => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      alert("Speech Recognition is not supported in this browser.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript.toLowerCase().trim();
      const matched = availableClasses.find((cls) =>
        transcript.includes(cls.toLowerCase())
      );

      if (matched) {
        setTarget(matched);
        sendTarget(matched);
        alert(`🎯 Target set to "${matched}"`);
      } else {
        alert(`❌ "${transcript}" is not a recognized object.`);
      }
    };

    recognition.start();
  };

  return (
    <div className="p-4 max-w-screen-md mx-auto bg-white rounded-xl shadow space-y-4">
      <h1 className="text-xl font-bold text-center">🎯 Object Detection</h1>

      {/* Controls */}
      <div className="flex flex-col sm:flex-row gap-3 items-center justify-center">
        <button
          className="bg-blue-600 text-white px-4 py-2 rounded w-full sm:w-auto"
          onClick={startStream}
          disabled={streaming}
        >
          📹 Start Camera
        </button>
        <button
          className="bg-red-600 text-white px-4 py-2 rounded w-full sm:w-auto"
          onClick={stopStream}
          disabled={!streaming}
        >
          ⏹️ Stop Camera
        </button>
        <button
          className="bg-purple-600 text-white px-4 py-2 rounded w-full sm:w-auto"
          onClick={startVoiceRecognition}
        >
          🎤 Speak Target
        </button>
      </div>

      {/* Video */}
      <div className="text-center">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          className="w-full max-w-full border rounded-md"
        />
      </div>

      {/* Status */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm">
        <div className="bg-gray-50 p-3 rounded">
          <p>
            <strong>WebSocket:</strong>{" "}
            <span className={connected ? "text-green-600" : "text-red-600"}>
              {connected ? "Connected" : "Disconnected"}
            </span>
          </p>
          <p>
            <strong>Streaming:</strong> {streaming ? "Yes" : "No"}
          </p>
          <p>
            <strong>FPS:</strong> {fps}
          </p>
          <p>
            <strong>Frame Count:</strong> {frameCount}
          </p>
        </div>
        <div className="bg-gray-50 p-3 rounded">
          <p>
            <strong>Target:</strong>{" "}
            {target ? (
              <span className="text-blue-600">{target}</span>
            ) : (
              <span className="text-gray-500">Not set</span>
            )}
          </p>
          <p>
            <strong>Target Found:</strong>{" "}
            <span className={targetFound ? "text-green-600" : "text-red-600"}>
              {targetFound ? "✅" : "❌"}
            </span>
          </p>
        </div>
      </div>

      {/* Detection Log */}
      <div className="bg-gray-50 p-3 rounded max-h-44 overflow-y-auto text-xs font-mono">
        <h3 className="font-semibold mb-1">📝 Detection Log</h3>
        <ul className="space-y-1">
          {detectionLog.map((log, idx) => (
            <li key={idx}>
              [{log.time}] ➤ Target Detected:{" "}
              {log.detected.target_detected ? "✅" : "❌"} | Total:{" "}
              {log.detected.total_objects}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
