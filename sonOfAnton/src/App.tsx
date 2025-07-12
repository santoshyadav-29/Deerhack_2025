import { useState, useEffect, useRef } from "react";

const commandResponses: {
  [key: string]: { text: string; lang?: string; translate?: boolean };
} = {
  "what is your name": { text: "My name is Anton." },
  "what can you do": {
    text: "I can listen to your commands and respond. Try asking what time it is.",
  },
  "how are you": {
    text: "I am a computer program, so I don't have feelings, but I appreciate you asking!",
  },
  "what time is it": {
    text: `The current time is ${new Date().toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    })}.`,
  },
  "speak in nepali": {
    text: "तपाईंलाई कस्तो छ?",
    lang: "ne",
  },
};

function App() {
  const [isListening, setIsListening] = useState(false);
  const [started, setStarted] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [statusText, setStatusText] = useState("Say 'Hello' to start.");

  const listeningRef = useRef(isListening);
  listeningRef.current = isListening;
  const shouldRestartRef = useRef(true);

  useEffect(() => {
    if (!started) return;

    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setStatusText("Sorry, your browser doesn't support the Web Speech API.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onresult = (event) => {
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

      if (!listeningRef.current && command.includes("hello")) {
        shouldRestartRef.current = true;
        setIsListening(true);
        setStatusText("I'm listening...");
        speak("I'm listening...");
        return;
      }

      if (listeningRef.current && command.includes("bye")) {
        shouldRestartRef.current = false;
        setIsListening(false);
        setStatusText("Goodbye! Say 'Hello' to start again.");
        speak("Goodbye!");
        setTranscript("");
        return;
      }

      if (listeningRef.current && finalTranscript) {
        const response = commandResponses[command];

        if (response) {
          setStatusText(response.text);
          speak(
            response.text,
            response.lang || "en",
            response.translate || false
          );
        } else {
          setStatusText(`I heard you say: ${finalTranscript}`);
          speak(finalTranscript);
        }

        setTranscript("");
      }
    };

    recognition.onend = () => {
      if (shouldRestartRef.current) {
        recognition.start();
      }
    };

    recognition.onerror = (event) => {
      console.error("Speech recognition error:", event.error);
    };

    recognition.start();

    return () => {
      shouldRestartRef.current = false;
      recognition.onend = null;
      recognition.stop();
    };
  }, [started]);

  useEffect(() => {
    if (started) {
      const guideTimeout = setTimeout(() => {
        speak("Welcome.");
      }, 1000);

      return () => clearTimeout(guideTimeout);
    }
  }, [started]);

  const speak = async (
    text: string,
    lang: string = "en",
    translate = false
  ) => {
    try {
      const response = await fetch("http://localhost:5000/api/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, lang, translate }),
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
      <div className="flex-grow"></div>

      <div className="relative flex items-center justify-center">
        <div
          className={`transition-all duration-500 w-56 h-56 sm:w-72 sm:h-72 rounded-full bg-gradient-to-br from-purple-500 to-indigo-700 ${
            isListening ? "animate-pulse" : "shadow-lg"
          }`}
        ></div>
        <div className="absolute text-xl font-semibold tracking-wider">
          {isListening ? "Listening..." : "Idle"}
        </div>
      </div>

      <div className="flex-grow"></div>

      <div className="w-full max-w-lg text-center h-28 flex flex-col justify-end">
        <p className="text-2xl font-medium mb-2">{statusText}</p>
        <p className="text-gray-400 text-lg min-h-[28px]">
          {transcript && `"${transcript}"`}
        </p>
      </div>
    </main>
  );
}

export default App;
