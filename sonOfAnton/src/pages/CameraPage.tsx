import React, { useEffect, useRef, useState } from "react";

const COCO_CLASSES = [
  /* your list here, omitted for brevity */
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
    const wsUrl = "wss://webcam.aavashlamichhane.com.np/ws/stream";
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
        if (!video || video.readyState < 2) return; // Changed here

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
    <div className="p-5 bg-gray-900 min-h-screen flex flex-col items-center gap-5 text-white font-sans">
      <h1 className="text-2xl font-bold text-cyan-400">
        🎥 Object Detection Voice Control
      </h1>

      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-full max-w-md rounded-xl border-4 border-cyan-400 bg-black aspect-[4/3]"
      />

      <button
        onClick={streaming ? stopStream : startStream}
        aria-label="Toggle camera"
        className={`w-full max-w-md py-4 text-xl font-bold rounded-xl border-none cursor-pointer transition-shadow ${
          streaming
            ? "bg-red-600 text-gray-900 shadow-[0_0_15px_#f44]"
            : "bg-cyan-400 text-gray-900 shadow-[0_0_15px_#0ff]"
        }`}
      >
        {streaming ? "Stop Camera" : "Start Camera"}
      </button>

      <button
        onClick={listening ? stopListening : startListening}
        aria-label="Toggle voice command"
        className={`w-full max-w-md py-4 text-xl font-bold rounded-xl border-none cursor-pointer transition-shadow ${
          listening
            ? "bg-red-600 text-gray-900 shadow-[0_0_15px_#f44]"
            : "bg-cyan-400 text-gray-900 shadow-[0_0_15px_#0ff]"
        }`}
      >
        {listening ? "Stop Listening" : "Start Voice Command"}
      </button>

      <div className="w-full max-w-md">
        <div className="mb-2">
          <strong>Connection Status: </strong>
          <span
            className={
              connectionStatus === "Connected"
                ? "text-green-500"
                : "text-red-500"
            }
          >
            {connectionStatus}
          </span>
        </div>

        <div className="mb-2 text-lg">
          <strong>Current Target: </strong>
          {target ? (
            <span className="text-cyan-400 font-bold">{target}</span>
          ) : (
            <span className="text-gray-500">No target set</span>
          )}
        </div>

        <div className="mb-2 text-lg">
          <strong>Target Found: </strong>
          {targetFound ? (
            <span className="text-green-500 font-bold">Yes 🎯</span>
          ) : (
            <span className="text-red-500 font-bold">No</span>
          )}
        </div>

        <div>
          <strong>Detected Objects:</strong>
          {detectedObjects.length === 0 ? (
            <p className="text-gray-500 text-base mt-1">
              No objects detected yet.
            </p>
          ) : (
            <ul className="mt-1 list-disc pl-5 max-h-36 overflow-y-auto text-base text-cyan-400">
              {detectedObjects.map(({ class: cls, confidence }, i) => (
                <li
                  key={i}
                  className={`mb-1 ${
                    cls === target ? "font-bold text-green-500" : ""
                  }`}
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
