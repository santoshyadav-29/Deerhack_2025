import { useEffect, useRef } from "react";

function CameraPage() {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    async function startCamera() {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter((d) => d.kind === "videoinput");

        // Try to find a non-DroidCam camera
        const preferredDevice =
          videoDevices.find(
            (device) =>
              !/droidcam/i.test(device.label) &&
              /camera|usb/i.test(device.label)
          ) || videoDevices[0]; // fallback to first if not found

        if (!preferredDevice) {
          alert("No camera device found.");
          return;
        }

        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            deviceId: preferredDevice.deviceId,
          },
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Camera error:", err);
        alert("Unable to access camera. Please check permissions.");
      }
    }

    startCamera();

    return () => {
      if (videoRef.current?.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach((t) => t.stop());
      }
    };
  }, []);

  return (
    <div className="flex justify-center items-center min-h-screen bg-black px-4">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="rounded-xl w-full max-w-sm shadow-lg"
      />
    </div>
  );
}

export default CameraPage;
