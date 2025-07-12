from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from gtts import gTTS
import uuid
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

class SpeakRequest(BaseModel):
    text: str
    lang: str = "en"
    translate: bool = False

@app.post("/api/speak")
async def speak(req: SpeakRequest):
    if not req.text.strip():
        return JSONResponse(status_code=400, content={"error": "Missing text"})

    try:
        filename = f"{uuid.uuid4().hex}.mp3"
        path = os.path.join(AUDIO_DIR, filename)

        tts = gTTS(text=req.text, lang=req.lang)
        tts.save(path)

        return FileResponse(path, media_type="audio/mpeg", filename=filename)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
