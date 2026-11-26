"""ASR 评测：在 wenjing20.tsv 上对比 SenseVoice / FunASR / Whisper 的效果与速度。

用法（在各后端对应 venv 下运行）：
  python asr_eval.py --backend sensevoice   # 需 cwd 在 SenseVoice 仓库（remote_code=./model.py）
  python asr_eval.py --backend funasr
  python asr_eval.py --backend whisper
"""
import argparse
import json
import os
import re
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

TSV = r"F:\ai\aivoice\docs\asr_eval\wenjing20.tsv"
OUT = r"F:\ai\aivoice\docs\asr_eval\result_{backend}.json"


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            wav, text = line.rstrip("\n").split("\t", 1)
            rows.append((wav, text))
    return rows


def clean(text):
    """去掉 SenseVoice 富文本里的情感/事件标签，便于和其他后端公平比 CER。"""
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"<\|[^|]*\|>", "", text)
    text = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF]", "", text)
    return "".join(text.split())


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref, hyp):
    ref = "".join(ref.split())
    hyp = clean(hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def build_backend(name):
    if name == "sensevoice":
        from funasr import AutoModel
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        model = AutoModel(
            model="iic/SenseVoiceSmall",
            trust_remote_code=True,
            remote_code="./model.py",
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="cuda:0",
        )

        def transcribe(wav):
            res = model.generate(
                input=wav, cache={}, language="zh", use_itn=True,
                batch_size_s=60, merge_vad=True, merge_length_s=15,
            )
            return rich_transcription_postprocess(res[0]["text"])

        return transcribe

    if name == "funasr":
        from funasr import AutoModel

        model = AutoModel(
            model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc",
            device="cuda:0",
        )

        def transcribe(wav):
            return model.generate(input=wav)[0]["text"]

        return transcribe

    if name == "whisper":
        import torch
        import whisper

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"whisper device={device}", flush=True)
        model = whisper.load_model("small", device=device)

        def transcribe(wav):
            return model.transcribe(wav, language="zh")["text"]

        return transcribe

    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["sensevoice", "funasr", "whisper"], required=True)
    args = ap.parse_args()

    rows = load_rows(TSV)
    transcribe = build_backend(args.backend)

    import torch
    results = []
    t_start = time.time()
    for wav, ref in rows:
        t0 = time.time()
        hyp = transcribe(wav)
        dt = time.time() - t0
        results.append({
            "wav": os.path.basename(wav),
            "ref": ref,
            "hyp": hyp,
            "cer": round(cer(ref, hyp), 4),
            "sec": round(dt, 3),
        })
        print(f"{os.path.basename(wav)} cer={results[-1]['cer']:.3f} {dt:.2f}s | {hyp[:30]}", flush=True)

    total = time.time() - t_start
    summary = {
        "backend": args.backend,
        "items": len(results),
        "mean_cer": round(sum(r["cer"] for r in results) / len(results), 4),
        "total_sec": round(total, 1),
        "mean_sec": round(total / len(results), 3),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2) if torch.cuda.is_available() else None,
        "detail": results,
    }
    out = OUT.format(backend=args.backend)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"SUMMARY mean_cer={summary['mean_cer']} mean_sec={summary['mean_sec']} vram={summary['peak_vram_gb']}GB -> {out}")


if __name__ == "__main__":
    main()
