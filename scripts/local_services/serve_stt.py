"""SenseVoice OpenAI-compatible STT server (:8766/v1/audio/transcriptions)."""
import io
import os
import sys
import tempfile

os.environ.setdefault("MODELSCOPE_CACHE", "F:/ai/cache/modelscope")
os.environ.setdefault("HF_HOME", "F:/ai/cache/huggingface")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import uvicorn

DEVICE = os.environ.get("SENSEVOICE_DEVICE", "cuda:0")
print(f"loading SenseVoice on {DEVICE} (cwd must be SenseVoice repo for remote_code)", flush=True)

model = AutoModel(
    model="iic/SenseVoiceSmall",
    trust_remote_code=True,
    remote_code="./model.py",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device=DEVICE,
    disable_update=True,
)
print("SenseVoice loaded", flush=True)

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok", "model": "SenseVoiceSmall", "device": DEVICE}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model_name: str = Form("SenseVoiceSmall"),
    language: str = Form("auto"),
    response_format: str = Form("json"),
):
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        res = model.generate(
            input=path,
            cache={},
            language=language if language in ("zh", "en", "yue", "ja", "ko", "auto") else "auto",
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        text = rich_transcription_postprocess(res[0]["text"])
        return {"text": text}
    finally:
        os.unlink(path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8766, log_level="warning")
