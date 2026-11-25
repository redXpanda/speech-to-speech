"""Pipeline audio sample rates.

The input side is fixed at 16 kHz on purpose: Silero VAD and every STT backend
(VAD-gated Whisper, Paraformer, Parakeet) are trained for it, and resampling
earlier would only add latency.

The output side is configurable. It stays at 16 kHz by default, which is what
every built-in TTS backend and the packaged clients assume. Raise it only when
the TTS backend produces a wider band (for example a 48 kHz GPT-SoVITS
checkpoint) and the client negotiates the same rate through ``session.update``;
otherwise the extra samples carry no information.
"""

from __future__ import annotations

import os

INPUT_SAMPLE_RATE = 16000
"""Rate the VAD/STT chain and inbound client audio are converted to."""

_VALID_OUTPUT_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)
_DEFAULT_OUTPUT_RATE = 16000


def _resolve_output_rate() -> int:
    raw = os.environ.get("SPEECH_TO_SPEECH_OUTPUT_RATE", "").strip()
    if not raw:
        return _DEFAULT_OUTPUT_RATE
    try:
        rate = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"SPEECH_TO_SPEECH_OUTPUT_RATE must be an integer, got {raw!r}"
        ) from exc
    if rate not in _VALID_OUTPUT_RATES:
        raise ValueError(
            f"SPEECH_TO_SPEECH_OUTPUT_RATE must be one of {_VALID_OUTPUT_RATES}, got {rate}"
        )
    return rate


OUTPUT_SAMPLE_RATE = _resolve_output_rate()
"""Rate TTS audio is carried at between the TTS backend and the transport."""
