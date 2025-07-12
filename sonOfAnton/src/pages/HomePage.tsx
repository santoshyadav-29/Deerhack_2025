import { useState, useEffect, useRef } from "react";

const cocoObjects = [
  "person",
  "bottle",
  "cup",
  "cell phone",
  "book",
  "laptop",
  "chair",
  "couch",
  "mouse",
  "keyboard",
  "tv",
];

function HomePage() {
  const [isListening, setIsListening] = useState(false);
  const [started, setStarted] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [statusText, setStatusText] = useState("Say 'Hello' to start.");
  const [lang, setLang] = useState("en");
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [lastResponse, setLastResponse] = useState("");

  const listeningRef = useRef(isListening);
  listeningRef.current = isListening;
  const shouldRestartRef = useRef(true);

  useEffect(() => {
    if (!started) return;

    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setStatusText("Web Speech API is not supported.");
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
        const result = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += result;
        } else {
          interimTranscript += result;
        }
      }

      setTranscript(interimTranscript);
      const command = finalTranscript.trim().toLowerCase();

      if (
        !listeningRef.current &&
        (command.includes("hello") || command.includes("नमस्ते"))
      ) {
        shouldRestartRef.current = true;
        setIsListening(true);
        const msg = lang === "en" ? "I'm listening..." : "म सुन्दैछु...";
        setStatusText(msg);
        speak(msg);
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
        speak(msg);
        setTranscript("");
        return;
      }

      if (listeningRef.current && command) {
        let spokenText = command;

        if (lang === "ne") {
          try {
            const res = await fetch(
              "https://share.aavashlamichhane.com.np/api/translate",
              {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: spokenText }),
              }
            );
            const data = await res.json();
            spokenText = data?.translated?.toLowerCase() || spokenText;
          } catch (err) {
            console.error("Translation error:", err);
          }
        }

        if (spokenText.includes("repeat") || spokenText.includes("फेरि भन")) {
          speak(lastResponse);
          return;
        }

        const detectedObjects = cocoObjects.filter((obj) =>
          spokenText.includes(obj)
        );
        let responseText = "";

        if (detectedObjects.length > 0) {
          try {
            await fetch(
              `https://share.aavashlamichhane.com.np/api/change_target/${detectedObjects[0]}`,
              {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ objects: detectedObjects, lang }),
              }
            );

            await fetch("https://share.aavashlamichhane.com.np/api/start", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
            });

            const pollInstruction = () => {
              return new Promise((resolve, reject) => {
                const instructionInterval = setInterval(async () => {
                  try {
                    const res = await fetch(
                      "https://share.aavashlamichhane.com.np/api/instruction",
                      {
                        method: "GET",
                        headers: { "Content-Type": "application/json" },
                      }
                    );
                    const data = await res.json();

                    if (data?.natural_language) {
                      clearInterval(instructionInterval);

                      let spokenInstruction = data.natural_language;

                      if (lang === "ne") {
                        try {
                          const translationRes = await fetch(
                            "https://share.aavashlamichhane.com.np/api/translate",
                            {
                              method: "POST",
                              headers: { "Content-Type": "application/json" },
                              body: JSON.stringify({
                                text: spokenInstruction,
                              }),
                            }
                          );
                          const translationData = await translationRes.json();
                          spokenInstruction =
                            translationData?.translated || spokenInstruction;
                        } catch (err) {
                          console.error("Translation error:", err);
                        }
                      }

                      speak(spokenInstruction);
                      resolve(spokenInstruction);

                      // 🟢 Start polling speak-current
                      const speakCurrentInterval = setInterval(async () => {
                        try {
                          const response = await fetch(
                            "https://share.aavashlamichhane.com.np/api/speak_current",
                            {
                              method: "POST",
                              headers: {
                                "Content-Type": "application/json",
                              },
                              body: JSON.stringify({ lang }),
                            }
                          );
                          if (!response.ok) throw new Error("Speak failed");

                          const blob = await response.blob();
                          const audioUrl = URL.createObjectURL(blob);
                          const audio = new Audio(audioUrl);
                          audio.volume = 0.8;
                          audio.play();
                          audio.onended = () => URL.revokeObjectURL(audioUrl);
                        } catch (err) {
                          console.error("speak-current error:", err);
                          clearInterval(speakCurrentInterval);
                        }
                      }, 2000);

                      setTimeout(
                        () => clearInterval(speakCurrentInterval),
                        30000
                      );
                    }
                  } catch (err) {
                    clearInterval(instructionInterval);
                    reject(err);
                  }
                }, 2000);
              });
            };

            responseText = await pollInstruction();
          } catch (err) {
            console.error("Object handling error:", err);
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
            responseText = data?.response || `No known object detected.`;
          } catch (err) {
            responseText =
              lang === "en" ? "Something went wrong." : "केही गलत भयो।";
          }
        }

        setStatusText(responseText);
        setLastResponse(responseText);
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

  const speak = async (text) => {
    try {
      setIsSpeaking(true);
      const response = await fetch(
        "https://share.aavashlamichhane.com.np/api/speak",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, lang }),
        }
      );
      if (!response.ok) throw new Error("Failed to fetch audio");

      const blob = await response.blob();
      const audioUrl = URL.createObjectURL(blob);
      const audio = new Audio(audioUrl);
      audio.volume = 0.8;
      audio.play();
      audio.onended = () => {
        URL.revokeObjectURL(audioUrl);
        setIsSpeaking(false);
      };
    } catch (err) {
      console.error("Speech playback error:", err);
      setIsSpeaking(false);
    }
  };

  return (
    <main className="min-h-screen bg-gray-900 text-white flex flex-col items-center px-4 pt-10 pb-24">
      {!started ? (
        <button
          onClick={() => setStarted(true)}
          className="bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-3 text-lg font-semibold rounded-xl shadow-lg mt-20"
        >
          Start Assistant
        </button>
      ) : (
        <>
          <div className="flex justify-end w-full max-w-sm mb-6">
            <button
              onClick={() => setLang(lang === "en" ? "ne" : "en")}
              className="bg-indigo-600 hover:bg-indigo-700 px-4 py-2 rounded-md text-sm"
            >
              {lang === "en" ? "Switch to नेपाली" : "Switch to English"}
            </button>
          </div>

          <div className="relative w-48 h-48 sm:w-56 sm:h-56 rounded-full bg-gradient-to-br from-purple-500 to-indigo-700 flex items-center justify-center mb-6">
            <div className="absolute text-xl sm:text-2xl font-medium">
              {isListening
                ? lang === "en"
                  ? "Listening..."
                  : "सुन्दैछु..."
                : lang === "en"
                ? "Idle"
                : "निस्क्रिय"}
            </div>
            {isListening && (
              <div className="absolute inset-0 animate-ping rounded-full bg-indigo-500 opacity-20"></div>
            )}
          </div>

          <div className="text-center w-full max-w-md">
            <p className="text-xl font-semibold mb-2">{statusText}</p>
            <p className="text-gray-400 italic min-h-[24px] text-sm sm:text-base">
              {transcript && `"${transcript}"`}
            </p>
            {isSpeaking && (
              <p className="text-green-400 mt-2 animate-pulse">Speaking...</p>
            )}
          </div>
        </>
      )}
    </main>
  );
}

export default HomePage;
