import React, { useEffect, useRef, useState } from "react";

const COCO_CLASSES = [
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "backpack",
  "umbrella",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "couch",
  "potted plant",
  "bed",
  "dining table",
  "toilet",
  "tv",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush",
];

const ObjectDetectionVoiceMobile = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(document.createElement("canvas"));
  const wsRef = useRef(null);
  const streamIntervalRef = useRef(null);
  const recognitionRef = useRef(null);

  const [streaming, setStreaming] = useState(false);
  const [target, setTarget] = useState("");
  const [targetFound, setTargetFound] = useState(false);
  const [detectedObjects, setDetectedObjects] = useState([]);
  const [listening, setListening] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState("Disconnected");

  // WebSocket setup
  useEffect(() => {
    const wsUrl = `${window.location.origin.replace(/^http/, "ws")}/ws/stream`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => setConnectionStatus("Connected");
    ws.onclose = () => {
      setConnectionStatus("Disconnected");
      setTimeout(() => {
        if (wsRef.current.readyState !== WebSocket.OPEN) {
          wsRef.current = new WebSocket(wsUrl);
        }
      }, 3000);
    };
    ws.onerror = () => setConnectionStatus("Error");
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "detection_result") {
          setDetectedObjects(data.detected_objects || []);
          setTargetFound(data.target_detected);
        }
      } catch {}
    };

    return () => ws.close();
  }, []);

  const startStream = async () => {
    if (streaming) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
      });
      videoRef.current.srcObject = stream;
      setStreaming(true);

      const ctx = canvasRef.current.getContext("2d");

      streamIntervalRef.current = setInterval(() => {
        if (!streaming || wsRef.current.readyState !== WebSocket.OPEN) return;
        const video = videoRef.current;
        if (!video || video.readyState !== 4) return;

        canvasRef.current.width = video.videoWidth;
        canvasRef.current.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);

        canvasRef.current.toBlob(
          (blob) => {
            if (!blob) return;
            const reader = new FileReader();
            reader.onloadend = () => {
              if (wsRef.current.readyState === WebSocket.OPEN) {
                wsRef.current.send(reader.result);
              }
            };
            reader.readAsDataURL(blob);
          },
          "image/jpeg",
          0.7
        );
      }, 150);
    } catch (err) {
      alert("Camera error: " + err.message);
    }
  };

  const stopStream = () => {
    setStreaming(false);
    if (streamIntervalRef.current) {
      clearInterval(streamIntervalRef.current);
      streamIntervalRef.current = null;
    }
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
  };

  const startListening = () => {
    if (
      !("webkitSpeechRecognition" in window) &&
      !("SpeechRecognition" in window)
    ) {
      alert("Speech Recognition API not supported in this browser.");
      return;
    }
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.continuous = false;

    recognition.onstart = () => setListening(true);
    recognition.onend = () => setListening(false);
    recognition.onerror = () => setListening(false);
    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript.toLowerCase().trim();
      const matched = COCO_CLASSES.find((cls) => transcript.includes(cls));
      if (matched) {
        setTarget(matched);
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(
            JSON.stringify({ type: "set_target", target: matched })
          );
          alert(`Target set to: ${matched}`);
        }
      } else {
        alert(`No valid object found in speech: "${transcript}"`);
      }
    };

    recognition.start();
    recognitionRef.current = recognition;
  };

  const stopListening = () => {
    recognitionRef.current?.stop();
  };

  return (
    <div
      style={{
        padding: 20,
        fontFamily: "Arial, sans-serif",
        backgroundColor: "#111",
        minHeight: "100vh",
        color: "#fff",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 20,
      }}
    >
      <h1 style={{ fontSize: 24, fontWeight: "bold", color: "#0ff" }}>
        🎥 Object Detection Voice Control
      </h1>

      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{
          width: "100%",
          maxWidth: 400,
          borderRadius: 15,
          border: "3px solid #0ff",
          aspectRatio: "4 / 3",
          backgroundColor: "#000",
        }}
      />

      <button
        onClick={streaming ? stopStream : startStream}
        style={{
          width: "100%",
          maxWidth: 400,
          padding: 18,
          fontSize: 20,
          fontWeight: "bold",
          color: "#111",
          backgroundColor: streaming ? "#f22" : "#0ff",
          border: "none",
          borderRadius: 12,
          cursor: "pointer",
          boxShadow: streaming ? "0 0 15px #f44" : "0 0 15px #0ff",
          transition: "background-color 0.3s ease",
        }}
        aria-label="Toggle camera"
      >
        {streaming ? "Stop Camera" : "Start Camera"}
      </button>

      <button
        onClick={listening ? stopListening : startListening}
        style={{
          width: "100%",
          maxWidth: 400,
          padding: 18,
          fontSize: 20,
          fontWeight: "bold",
          color: "#111",
          backgroundColor: listening ? "#f22" : "#0ff",
          border: "none",
          borderRadius: 12,
          cursor: "pointer",
          boxShadow: listening ? "0 0 15px #f44" : "0 0 15px #0ff",
          transition: "background-color 0.3s ease",
        }}
        aria-label="Toggle voice command"
      >
        {listening ? "Stop Listening" : "Start Voice Command"}
      </button>

      <div style={{ width: "100%", maxWidth: 400 }}>
        <div style={{ marginBottom: 10 }}>
          <strong>Connection Status: </strong>
          <span
            style={{
              color: connectionStatus === "Connected" ? "#0f0" : "#f44",
            }}
          >
            {connectionStatus}
          </span>
        </div>

        <div style={{ marginBottom: 10, fontSize: 18 }}>
          <strong>Current Target: </strong>
          {target ? (
            <span style={{ color: "#0ff", fontWeight: "bold" }}>{target}</span>
          ) : (
            <span style={{ color: "#666" }}>No target set</span>
          )}
        </div>

        <div style={{ marginBottom: 10, fontSize: 18 }}>
          <strong>Target Found: </strong>
          {targetFound ? (
            <span style={{ color: "#0f0", fontWeight: "bold" }}>Yes 🎯</span>
          ) : (
            <span style={{ color: "#f44", fontWeight: "bold" }}>No</span>
          )}
        </div>

        <div>
          <strong>Detected Objects:</strong>
          {detectedObjects.length === 0 ? (
            <p style={{ color: "#888", fontSize: 16, marginTop: 6 }}>
              No objects detected yet.
            </p>
          ) : (
            <ul
              style={{
                marginTop: 6,
                listStyleType: "disc",
                paddingLeft: 20,
                maxHeight: 140,
                overflowY: "auto",
                fontSize: 16,
                color: "#0ff",
              }}
            >
              {detectedObjects.map(({ class: cls, confidence }, i) => (
                <li
                  key={i}
                  style={{
                    fontWeight: cls === target ? "bold" : "normal",
                    color: cls === target ? "#0f0" : "#0ff",
                    marginBottom: 4,
                  }}
                >
                  {cls} — {(confidence * 100).toFixed(1)}%
                  {cls === target ? " (Target!)" : ""}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
};

export default ObjectDetectionVoiceMobile;
