"""Text-only transport for the external Realtime conversation authority."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Literal

from uuid import uuid4


class ContextProtocolError(Exception):
    """The context service did not return a well-formed event stream."""


@dataclass(frozen=True)
class ContextTurn:
    turn_id: str
    revision: int
    response_id: str


@dataclass(frozen=True)
class ContextEvent:
    type: Literal["text", "completed", "cancelled", "failed"]
    text: str = ""


class ExternalContextClient:
    def __init__(self, send: Callable[[dict], Awaitable[None]]) -> None:
        self._send = send
        self._streams: dict[str, asyncio.Queue] = {}
        self._acks: dict[str, asyncio.Future] = {}

    def receive(self, frame: dict) -> None:
        kind = frame["type"]
        if kind in ("aivoice.context.text", "aivoice.context.terminal"):
            response_id = frame.get("response_id")
            if isinstance(response_id, str) and response_id in self._streams:
                self._streams[response_id].put_nowait(frame)
        elif kind == "aivoice.context.ack":
            control_id = frame.get("control_id")
            ack = self._acks.get(control_id) if isinstance(control_id, str) else None
            if ack is not None and not ack.done():
                ack.set_result(frame.get("ok") is True)

    async def stream(
        self,
        turn: ContextTurn,
        *,
        transcript: str,
        language: str | None,
        voice: str | None,
    ) -> AsyncIterator[ContextEvent]:
        if turn.response_id in self._streams:
            raise ContextProtocolError("Duplicate context response ID")
        queue = self._streams[turn.response_id] = asyncio.Queue()
        try:
            await self._send({"type": "aivoice.context.request", **asdict(turn),
                              "transcript": transcript, "language": language, "voice": voice})
            expected_seq = 0
            while True:
                event = await asyncio.wait_for(queue.get(), timeout=125)
                if event["type"] == "aivoice.context.text":
                    text, seq = event.get("text"), event.get("seq")
                    if type(seq) is not int or seq != expected_seq or not isinstance(text, str) or not text:
                        raise ContextProtocolError("Malformed or out-of-order text event")
                    expected_seq += 1
                    yield ContextEvent("text", text)
                else:
                    status = event.get("status")
                    if status not in ("completed", "cancelled", "failed"):
                        raise ContextProtocolError("Invalid context terminal status")
                    yield ContextEvent(status)
                    return
        finally:
            self._streams.pop(turn.response_id)

    async def _control(self, action: str, turn: ContextTurn, **fields) -> None:
        control_id = uuid4().hex
        ack = self._acks[control_id] = asyncio.get_running_loop().create_future()
        try:
            await self._send({"type": f"aivoice.context.{action}", **asdict(turn),
                              "control_id": control_id, **fields})
            if not await ack:
                raise ContextProtocolError(f"Context {action} was rejected")
        finally:
            self._acks.pop(control_id)

    async def mutate(self, turn: ContextTurn, *, role: str, op: str, text: str | None) -> None:
        await self._control("mutate", turn, role=role, op=op, text=text)

    async def cancel(self, turn: ContextTurn) -> None:
        await self._control("cancel", turn)

    async def receipt(self, turn: ContextTurn, *, text: str, played_ms: int, final: bool) -> None:
        await self._control("receipt", turn, text=text, played_ms=played_ms, final=final)
