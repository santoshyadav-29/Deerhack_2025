from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from gtts import gTTS
from io import BytesIO
from googletrans import Translator

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

translator = Translator()

@app.post("/api/speak")
async def speak(request: Request):
    data = await request.json()
    text = data.get("text", "")
    lang = data.get("lang", "en")

    if not text:
        return JSONResponse(content={"error": "No text provided"}, status_code=400)

    tts = gTTS(text=text, lang="ne" if lang == "ne" else "en", slow=False)
    audio_fp = BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return StreamingResponse(audio_fp, media_type="audio/mpeg")


@app.post("/api/translate")
async def translate_to_en(request: Request):
    data = await request.json()
    text = data.get("text", "")
    if not text:
        return JSONResponse({"translated": ""})
    translated = translator.translate(text, dest="en")
    return JSONResponse({"translated": translated.text})


@app.post("/api/find")
async def find_objects(request: Request):
    data = await request.json()
    objects = data.get("objects", [])
    lang = data.get("lang", "en")

    if not objects:
        return JSONResponse({"response": "No objects provided."})

    if lang == "ne":
        translated_objects = ", ".join(objects)
        return JSONResponse({"response": f"मैले {translated_objects} भेटेँ।"})
    else:
        return JSONResponse({"response": f"I found {', '.join(objects)}."})


@app.post("/api/no-object")
async def no_object(request: Request):
    data = await request.json()
    lang = data.get("lang", "en")

    if lang == "ne":
        return JSONResponse({"response": "मैले चिनिएको कुनै वस्तु भेटिन।"})
    else:
        return JSONResponse({"response": "I didn't detect any known object."})
