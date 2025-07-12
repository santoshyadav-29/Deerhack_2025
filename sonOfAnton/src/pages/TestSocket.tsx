import React, { useRef, useEffect, useState } from "react";

export default function HttpCameraSender() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [detectedObjects, setDetectedObjects] = useState([]);

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
      } catch (err) {
        console.error("Error accessing camera:", err);
      }
    };
    startCamera();
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      if (!videoRef.current || !canvasRef.current) return;

      const ctx = canvasRef.current.getContext("2d");
      ctx.drawImage(videoRef.current, 0, 0, 320, 240);

      canvasRef.current.toBlob(
        async (blob) => {
          if (!blob) return;

          const formData = new FormData();
          formData.append("file", blob, "frame.jpg");

          try {
            const res = await fetch("http://localhost:8000/detect/", {
              method: "POST",
              body: formData,
            });

            if (res.ok) {
              const data = await res.json();
              setDetectedObjects(data.objects);
            } else {
              console.error("Detection failed", res.statusText);
            }
          } catch (e) {
            console.error("Detection request error:", e);
          }
        },
        "image/jpeg",
        0.7
      );
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <h2>HTTP Camera Frame Detection</h2>
      <video ref={videoRef} width="320" height="240" />
      <canvas
        ref={canvasRef}
        width="320"
        height="240"
        style={{ display: "none" }}
      />
      <div>
        <h3>Detected Objects:</h3>
        <ul>
          {detectedObjects.length === 0 && <li>None detected</li>}
          {detectedObjects.map((obj, i) => (
            <li key={i}>{obj}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}
