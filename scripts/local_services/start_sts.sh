#!/bin/bash
set -e
cd /f/ai/aivoice/speech-to-speech
export PYTHONPATH='F:\ai\aivoice\speech-to-speech\src'
export SPEECH_TO_SPEECH_OUTPUT_RATE=32000
export TENCENT_API_KEY=$(sed -n 's/^[[:space:]]*TENCENT_API_KEY:[[:space:]]*//p' /c/Users/Admin/.dsh/.credentials.yaml | head -1 | tr -d '"')
export PYTHONIOENCODING=utf-8
export NLTK_DATA='F:\ai\cache\nltk_data'

echo "TENCENT_API_KEY len=${#TENCENT_API_KEY}"

P=/f/ai/aivoice/speech-to-speech/.venv/Scripts/python.exe
$P -m speech_to_speech.cli serve \
  --host 0.0.0.0 \
  --port 8765 \
  --stt openai \
  --openai_stt_base_url http://127.0.0.1:8766/v1 \
  --openai_stt_model sensevoice \
  --openai_stt_response_format json \
  --llm_backend responses-api \
  --model_name hy4-flash \
  --responses_api_base_url https://tokenhub.tencentmaas.com/v1 \
  --responses_api_api_key "$TENCENT_API_KEY" \
  --tts openai \
  --openai_tts_base_url http://127.0.0.1:8767/v1 \
  --openai_tts_model cosyvoice3 \
  --openai_tts_voice_override ref02 \
  --openai_tts_response_format pcm \
  --openai_tts_sample_rate 24000
