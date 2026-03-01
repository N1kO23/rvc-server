from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import asyncio
import numpy as np
import soundfile as sf
import io
import os

from rvc_infer import RVCModel

app = FastAPI()

CHARACTER_NAME = os.getenv("CHARACTER_NAME")

if not CHARACTER_NAME:
    raise RuntimeError("CHARACTER_NAME environment variable not set")

MODEL_PATH = f"/models/{CHARACTER_NAME}/model.pth"
INDEX_PATH = f"/models/{CHARACTER_NAME}/model.index"

print(f"[RVC] Loading model: {MODEL_PATH}")
print(f"[RVC] Loading index: {INDEX_PATH}")

rvc = RVCModel(MODEL_PATH, INDEX_PATH)

def warmup():
    dummy = np.zeros(32000, dtype=np.float32)
    rvc.convert(dummy)

print("[RVC] Begin warmup... This might take a while")
warmup()
print("[RVC] Warmup complete")

@app.post("/convert")
async def convert_audio(
    audio: UploadFile = File(...),
    pitch_shift: int = Form(0),
    index_rate: float = Form(0.75)
):
    input_bytes = await audio.read()
    audio_np, sr = sf.read(io.BytesIO(input_bytes))

    converted = rvc.convert(audio_np, pitch_shift, index_rate)

    output_buffer = io.BytesIO()
    sf.write(output_buffer, converted, sr, format="WAV")
    output_buffer.seek(0)

    return StreamingResponse(output_buffer, media_type="audio/wav")

CHUNK_SIZE = 32000  # 1 second at 32kHz
OVERLAP = 1024  # small overlap buffer

@app.post("/convert_stream")
async def convert_stream(
    audio: UploadFile = File(...),
    pitch_shift: int = Form(0),
    index_rate: float = Form(0.75)
):
    input_bytes = await audio.read()
    audio_np, sr = sf.read(io.BytesIO(input_bytes))

    async def generator():
        total = len(audio_np)
        prev_tail = np.zeros(OVERLAP)

        for start in range(0, total, CHUNK_SIZE):
            chunk = audio_np[start:start + CHUNK_SIZE]
            chunk = np.concatenate([prev_tail, chunk])
            converted = rvc.convert(chunk, pitch_shift, index_rate)
            prev_tail = chunk[-OVERLAP:]

            buffer = io.BytesIO()
            sf.write(buffer, converted, sr, format="WAV")
            buffer.seek(0)

            yield buffer.read()

            await asyncio.sleep(0)  # yield control

    return StreamingResponse(generator(), media_type="audio/wav")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "character": CHARACTER_NAME
    }
