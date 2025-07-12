import { useState, useEffect, useRef } from "react";

const cocoObjects = [
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
  // add rest of COCO if needed
];

function App() {
  const [isListening, setIsListening] = useState(false);
  const [started, setStarted] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [statusText, setStatusText] = useState("Say 'Hello' to start.");
  const [lang, setLang] = useState<"en" | "ne">("en");

  const listeningRef = useRef(isListening);
  listeningRef.current = isListening;
  const shouldRestartRef = useRef(true);

  useEffect(() => {
    if (!started) return;

    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setStatusText("Sorry, your browser doesn't support Web Speech API.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = lang === "en" ? "en-US" : "ne-NP";

    recognition.onresult = async (event) => {
      let finalTranscript = "";
      let interimTranscript = "";

      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        } else {
          interimTranscript += event.results[i][0].transcript;
        }
      }

      setTranscript(interimTranscript);
      const command = finalTranscript.trim().toLowerCase();

      // Start and Stop commands with both languages
      if (
        !listeningRef.current &&
        (command.includes("hello") || command.includes("नमस्ते"))
      ) {
        shouldRestartRef.current = true;
        setIsListening(true);
        const msg = lang === "en" ? "I'm listening..." : "म सुन्दैछु...";
        setStatusText(msg);
        speak(msg, lang);
        return;
      }

      if (
        listeningRef.current &&
        (command.includes("bye") || command.includes("विदा"))
      ) {
        shouldRestartRef.current = false;
        setIsListening(false);
        const msg =
          lang === "en"
            ? "Goodbye! Say 'Hello' to start again."
            : "विदा! फेरि सुरु गर्न 'Hello' भन्नुहोस्।";
        setStatusText(msg);
        speak(lang === "en" ? "Goodbye!" : "विदा!", lang);
        setTranscript("");
        return;
      }

      if (listeningRef.current && finalTranscript) {
        let spokenText = finalTranscript.toLowerCase();

        // Translate Nepali → English
        if (lang === "ne") {
          try {
            const res = await fetch("http://localhost:5000/api/translate", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ text: spokenText }),
            });
            const data = await res.json();
            spokenText = data?.translated?.toLowerCase() || spokenText;
          } catch (err) {
            console.error("Translation error:", err);
          }
        }

        // Tokenize text to words
        const tokenize = (text: string) =>
          text
            .toLowerCase()
            .split(/[\s,\.!?]+/)
            .filter(Boolean);

        const words = tokenize(spokenText);

        const detectedObjects = cocoObjects.filter((obj) =>
          words.includes(obj.toLowerCase())
        );

        let responseText = "";

        if (detectedObjects.length > 0) {
          try {
            const res = await fetch("http://localhost:5000/api/find", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ objects: detectedObjects, lang }),
            });
            const data = await res.json();
            responseText =
              data?.response ||
              (lang === "en"
                ? `I found ${detectedObjects.join(", ")}.`
                : `मैले ${detectedObjects.join(", ")} भेटेँ।`);
          } catch (error) {
            console.error("Find API error:", error);
            responseText =
              lang === "en" ? "Something went wrong." : "केही गलत भयो।";
          }
        } else {
          try {
            const res = await fetch("http://localhost:5000/api/no-object", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ input: spokenText, lang }),
            });
            const data = await res.json();
            responseText =
              data?.response ||
              (lang === "en"
                ? `I didn't detect any known object.`
                : `मैले चिनिएको कुनै वस्तु भेटिन।`);
          } catch (error) {
            console.error("No Object API error:", error);
            responseText =
              lang === "en" ? "Something went wrong." : "केही गलत भयो।";
          }
        }

        setStatusText(responseText);
        speak(responseText, lang);
        setTranscript("");
      }
    };

    recognition.onend = () => {
      if (shouldRestartRef.current) recognition.start();
    };

    recognition.onerror = (event) => {
      console.error("Speech recognition error:", event.error);
    };

    recognition.start();

    return () => {
      shouldRestartRef.current = false;
      recognition.stop();
    };
  }, [started, lang]);

  const speak = async (text: string, lang: "en" | "ne") => {
    try {
      const response = await fetch("http://localhost:5000/api/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, lang }),
      });

      if (!response.ok) throw new Error("Failed to fetch audio");

      const blob = await response.blob();
      const audioUrl = URL.createObjectURL(blob);
      const audio = new Audio(audioUrl);
      await audio.play();
    } catch (err) {
      console.error("Speech playback error:", err);
    }
  };

  if (!started) {
    return (
      <main className="bg-gray-900 text-white min-h-screen flex items-center justify-center p-6">
        <button
          onClick={() => setStarted(true)}
          className="bg-indigo-600 hover:bg-indigo-700 px-6 py-3 text-lg font-semibold rounded-xl shadow-lg"
        >
          Start Assistant
        </button>
      </main>
    );
  }

  return (
    <main className="bg-gray-900 text-white min-h-screen flex flex-col items-center justify-center p-6 font-sans select-none">
      <div className="absolute top-4 right-4">
        <button
          onClick={() => setLang(lang === "en" ? "ne" : "en")}
          className="bg-indigo-600 hover:bg-indigo-700 px-4 py-2 rounded-lg shadow text-sm font-medium"
        >
          {lang === "en" ? "Switch to नेपाली" : "Switch to English"}
        </button>
      </div>

      <div className="relative flex items-center justify-center mt-16">
        <div
          className={`transition-all duration-500 w-56 h-56 rounded-full bg-gradient-to-br from-purple-500 to-indigo-700 ${
            isListening ? "animate-pulse" : "shadow-lg"
          }`}
        ></div>
        <div className="absolute text-xl font-semibold tracking-wider">
          {isListening
            ? lang === "en"
              ? "Listening..."
              : "सुन्दैछु..."
            : lang === "en"
            ? "Idle"
            : "निस्क्रिय"}
        </div>
      </div>

      <div className="w-full max-w-lg text-center mt-8">
        <p className="text-2xl font-medium mb-2">{statusText}</p>
        <p className="text-gray-400 text-lg min-h-[28px]">
          {transcript && `"${transcript}"`}
        </p>
      </div>
    </main>
  );
}

export default App;
