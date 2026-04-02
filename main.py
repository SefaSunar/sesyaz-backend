from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # Dosyayı geçici olarak kaydet
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Whisper ile transkript al
        with open(tmp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=audio_file,
                language="tr"
            )

        transcript = transcription.text

        # Özet ve notlar üret
        chat = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "Sen Türkçe ses transkriptlerini analiz eden bir asistansın. Verilen metni analiz et ve JSON formatında yanıt ver."
                },
                {
                    "role": "user",
                    "content": f"""Aşağıdaki Türkçe transkripti analiz et ve sadece JSON formatında yanıt ver:

Transkript: {transcript}

Şu formatta yanıt ver:
{{
  "summary": "2-3 cümlelik özet",
  "notes": ["madde 1", "madde 2", "madde 3", "madde 4", "madde 5"]
}}"""
                }
            ]
        )

        import json
        raw = chat.choices[0].message.content
        # JSON parse et
        start = raw.find("{")
        end = raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])

        return {
            "transcript": transcript,
            "summary": parsed.get("summary", ""),
            "notes": parsed.get("notes", [])
        }

    finally:
        os.unlink(tmp_path)