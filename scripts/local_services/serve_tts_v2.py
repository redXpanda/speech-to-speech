"""CosyVoice3 OpenAI-compatible TTS server, TRUE STREAMING (:8767/v1/audio/speech).

每个 CosyVoice3 stream chunk 立即以 PCM16 写出，STS 用 response.aiter_bytes() 增量消费，
实现 LLM 出字 -> 边合成边播放。
"""
import os
import sys

os.environ.setdefault("MODELSCOPE_CACHE", "F:/ai/cache/modelscope")
os.environ.setdefault("HF_HOME", "F:/ai/cache/huggingface")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, "F:/ai/aivoice/CosyVoice")
sys.path.append("F:/ai/aivoice/CosyVoice/third_party/Matcha-TTS")
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from cosyvoice.cli.cosyvoice import AutoModel
import uvicorn

MODEL_DIR = "F:/ai/aivoice/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
REF_WAV = r"F:\ai\aivoice\voice_data\_ref_02.wav"
REF_TEXT = (
    "You are a helpful assistant.<|endofprompt|>"
    "Okay, I had an idea I need to run this by you you got some someone in in in the chat here said"
)

print("loading CosyVoice3...", flush=True)
cosyvoice = AutoModel(model_dir=MODEL_DIR)
SR = cosyvoice.sample_rate
print(f"CosyVoice3 loaded, sample_rate={SR}", flush=True)

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok", "model": "Fun-CosyVoice3-0.5B-stream", "sample_rate": SR, "stream": True}


class SpeechRequest(BaseModel):
    model: str = "cosyvoice3"
    input: str
    voice: str = "ref02"
    response_format: str = "pcm"
    stream_format: str = "audio"
    stream: bool = False
    speed: float = 1.0


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    def gen():
        chunk_count = 0
        for item in cosyvoice.inference_zero_shot(
            req.input, REF_TEXT, REF_WAV, stream=True, speed=req.speed
        ):
            audio = item["tts_speech"]
            pcm = (audio.numpy() * 32767).astype("<i2").tobytes()
            chunk_count += 1
            print(f"[tts] chunk {chunk_count}: {audio.shape[1]} samples", flush=True)
            yield pcm

    return StreamingResponse(
        gen(),
        media_type="audio/pcm",
        headers={"X-Sample-Rate": str(SR), "X-Stream": "1"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8767, log_level="warning")
