import { useEffect, useRef, useState } from "react";

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
];

function App() {
  const [isActive, setIsActive] = useState(false);
  const [transcript, setTranscript] = useState("");
  const recognitionRef = useRef(null);

  useEffect(() => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Your browser does not support Speech Recognition");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognitionRef.current = recognition;

    recognition.onresult = (event) => {
      const lastResult = event.results[event.results.length - 1][0].transcript
        .trim()
        .toLowerCase();
      console.log("Heard:", lastResult);
      setTranscript(lastResult);

      if (lastResult.includes("hello")) {
        setIsActive(true);
        speak("Hi, I am Anton. How can I help you?");
        return;
      }

      if (lastResult.includes("bye")) {
        setIsActive(false);
        speak("Goodbye!");
        return;
      }

      if (isActive) {
        const foundObject = COCO_CLASSES.find((obj) =>
          lastResult.includes(obj)
        );
        if (foundObject) {
          speak(`I heard you mention: ${foundObject}`);
        } else {
          speak("I didn't recognize any object from your sentence.");
        }
      }
    };

    recognition.onerror = (event) => {
      console.error("Speech recognition error", event.error);
    };

    recognition.start();

    return () => recognition.stop();
  }, [isActive]);

  const speak = (text) => {
    const synth = window.speechSynthesis;
    const utterance = new SpeechSynthesisUtterance(text);
    synth.speak(utterance);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white p-4">
      <div
        className={`w-40 h-40 rounded-full mb-8 transition-all duration-500 ${
          isActive
            ? "bg-gradient-to-tr from-green-400 to-blue-500 animate-pulse"
            : "bg-gray-700"
        }`}
      />

      <div className="bg-gray-800 p-4 rounded-xl w-full max-w-md shadow-lg text-center text-lg">
        {isActive ? (
          <p className="text-green-400">Anton is listening...</p>
        ) : (
          <p className="text-gray-400">Say "Hello Anton" to activate</p>
        )}
        <p className="mt-2 text-sm text-gray-300">{transcript}</p>
      </div>
    </div>
  );
}

export default App;
