"""经 HTTPS 反代的 Realtime WS 冒烟测试：建连 -> session.update -> 文本轮次 -> 收音频。"""
import asyncio
import base64
import json
import ssl
import sys

import websockets

URL = "wss://192.168.3.210:8443/v1/realtime"
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


async def main():
    async with websockets.connect(URL, ssl=ctx, open_timeout=15) as ws:
        print("connected", flush=True)
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": "用简短口语回答一句话。",
                "output_modalities": ["audio"],
            },
        }))
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "用一句话介绍你自己"}],
            },
        }))
        await ws.send(json.dumps({"type": "response.create"}))
        audio_bytes = 0
        text = ""
        async def recv():
            nonlocal audio_bytes, text
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=40))
                t = msg.get("type", "")
                if t not in ("response.output_text.delta",):
                    print("EVENT", t, flush=True)
                if t == "response.output_audio.delta":
                    audio_bytes += len(base64.b64decode(msg.get("delta", "")))
                elif t == "response.output_text.delta":
                    text += msg.get("delta", "")
                elif t in ("response.done", "error"):
                    print(t, json.dumps(msg, ensure_ascii=False)[:500], flush=True)
                    return
        try:
            await recv()
        except asyncio.TimeoutError:
            print("recv timeout", flush=True)
        print(f"audio_bytes={audio_bytes} text={text!r}", flush=True)
        assert audio_bytes > 0, "no audio received"


if __name__ == "__main__":
    asyncio.run(main())
