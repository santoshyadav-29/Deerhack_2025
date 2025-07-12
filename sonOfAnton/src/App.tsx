import { useState, useEffect, useRef } from "react";

// A mapping of commands to their voice responses.
const commandResponses: { [key: string]: string } = {
  "what is your name": "My name is Anton.",
  "what can you do":
    "I can listen to your commands and respond. Try asking me what time it is, or how I am.",
  "how are you":
    "I am a computer program, so I don't have feelings, but thank you for asking!",
  "what time is it": `The time is ${new Date().toLocaleTimeString()}.`,
};

function App() {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [statusText, setStatusText] = useState('Say "Hello Anton" to start.');

  // A ref is used to hold the latest `isListening` state.
  // This avoids issues with stale state in the `onresult` callback.
  const listeningRef = useRef(isListening);
  listeningRef.current = isListening;

  useEffect(() => {
    // Check if the browser supports the Web Speech API.
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setStatusText("Sorry, your browser doesn't support the Web Speech API.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true; // Keep listening even after the user stops speaking.
    recognition.interimResults = true; // Get results as the user is speaking.
    recognition.lang = "en-US";

    // Event handler for when speech is recognized.
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

      setTranscript(interimTranscript); // Display the live, interim transcript.
      const command = finalTranscript.trim().toLowerCase();

      // --- Activation ("Hello Anton") ---
      if (!listeningRef.current && command.includes("hello anton")) {
        setIsListening(true);
        setStatusText("I'm listening...");
        speak("I'm listening...");
        return;
      }

      // --- Deactivation ("Bye Anton") ---
      if (listeningRef.current && command.includes("bye")) {
        setIsListening(false);
        setStatusText('Goodbye! Say "Hello Anton" to start again.');
        speak("Goodbye!");
        setTranscript("");
        return;
      }

      // --- Command Handling ---
      if (listeningRef.current && finalTranscript) {
        const response =
          commandResponses[command] || `I heard you say: ${finalTranscript}`;
        setStatusText(response);
        speak(response);
        setTranscript(""); // Clear transcript after responding.
      }
    };

    // Event handler for when the recognition service ends.
    recognition.onend = () => {
      // The service can sometimes stop unexpectedly. Restart it to ensure continuous listening.
      recognition.start();
    };

    // Event handler for recognition errors.
    recognition.onerror = (event) => {
      console.error("Speech recognition error:", event.error);
    };

    // Start the recognition service.
    recognition.start();

    // Cleanup function to stop recognition when the component unmounts.
    return () => {
      recognition.onend = null; // Prevent restart on unmount
      recognition.stop();
    };
  }, []); // Empty dependency array ensures this effect runs only once.

  // Function to make the browser speak a given text.
  const speak = (text: string) => {
    window.speechSynthesis.cancel(); // Cancel any speech in progress.
    const utterance = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(utterance);
  };

  return (
    <main className="bg-gray-900 text-white min-h-screen flex flex-col items-center justify-center p-6 font-sans select-none">
      <div className="flex-grow"></div>

      {/* The main animated circle */}
      <div className="relative flex items-center justify-center">
        <div
          className={`
            transition-all duration-500
            w-56 h-56 sm:w-72 sm:h-72 
            rounded-full
            bg-gradient-to-br from-purple-500 to-indigo-700
            ${isListening ? "animate-pulseGlow" : "shadow-lg"}
          `}
        ></div>
        {/* Text inside the circle indicating status */}
        <div className="absolute text-xl font-semibold tracking-wider">
          {isListening ? "Listening..." : "Idle"}
        </div>
      </div>

      <div className="flex-grow"></div>

      {/* Container for status and transcript text at the bottom */}
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
