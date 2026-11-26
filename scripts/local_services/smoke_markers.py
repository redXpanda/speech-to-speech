"""验证三段式 prompt 能让 LLM 输出 CosyVoice 标记，并到达 TTS。"""
import asyncio
import base64
import json
import ssl

import websockets

URL = "wss://192.168.3.210:8443/v1/realtime"
CV = """你在进行语音对话，你的文字回复会被 CosyVoice3 直接合成为语音。请遵守：
1. 用自然口语，回答简短，一到三句。
2. 在开心时插入 [laughter]，无奈感慨时插入 [sigh]，长句停顿处插入 [breath]。
3. 标记原样输出，方括号不要省略，不要解释标记含义。"""
TPL = "这是一段语音对话，你来扮演{{char}}来与{{user}}对话，下面是角色描述：{{desc}}"
DESC = "char: 爱莉希雅\n描述: 一个温柔开心的伙伴，说话爱笑。"
merged = TPL.replace("{{char}}", "爱莉希雅").replace("{{user}}", "用户").replace("{{desc}}", DESC)
SYSTEM = CV + "\n\n" + merged

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


async def main():
    async with websockets.connect(URL, ssl=ctx, open_timeout=15) as ws:
        await ws.send(json.dumps({"type": "session.update", "session": {
            "type": "realtime",
            "instructions": SYSTEM,
            "output_modalities": ["audio"],
        }}))
        await ws.send(json.dumps({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "今天发生了一件特别好笑的事，你听了肯定开心"}],
        }}))
        await ws.send(json.dumps({"type": "response.create"}))
        audio = 0
        transcripts = []
        async def recv():
            nonlocal audio
            while True:
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
                t = m.get("type", "")
                if t == "response.output_audio.delta":
                    audio += len(base64.b64decode(m.get("delta", "")))
                elif t == "response.output_audio_transcript.delta":
                    transcripts.append(m.get("delta", ""))
                elif t == "response.done":
                    return
        await recv()
        text = "".join(transcripts)
        print("TRANSCRIPT:", text)
        print("AUDIO_BYTES:", audio)
        print("HAS_MARKERS:", any(k in text for k in ["[laughter]", "[sigh]", "[breath]"]))


if __name__ == "__main__":
    asyncio.run(main())
