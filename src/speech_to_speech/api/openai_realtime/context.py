"""Connection-local bridge from Realtime to the conversation authority."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import uuid4

from speech_to_speech.pipeline.external_context import ContextTurn, ExternalContextClient
from speech_to_speech.pipeline.messages import EndOfResponse, LLMResponseChunk
from speech_to_speech.pipeline.speculative_turns import TurnGateAction
from speech_to_speech.utils.utils import _generate_id

if TYPE_CHECKING:
    from speech_to_speech.api.openai_realtime.service import RealtimeService
    from speech_to_speech.pipeline.messages import GenerateResponseRequest

logger = logging.getLogger(__name__)


@dataclass
class Reply:
    response_id: str = field(default_factory=lambda: _generate_id("resp"))
    turn: ContextTurn | None = None
    generation: asyncio.Task | None = None
    item_id: str | None = None
    status: str | None = None
    receipt_sent: bool = False
    receipt_task: asyncio.Task | None = None
    receipt_done: asyncio.Event = field(default_factory=asyncio.Event)
    regenerating: bool = False


class RealtimeContext:
    def __init__(self, send: Callable[[dict], Awaitable[None]]) -> None:
        self.client = ExternalContextClient(send)
        self.replies: dict[str, Reply] = {}
        self.tasks: set[asyncio.Task] = set()
        self.active: Reply | None = None
        self.input: tuple[str, str, str | None] | None = None
        self.last_turn_id: str | None = None
        self.revision = -1
        self.confirmed_turn_id: str | None = None
        self.items = {}
        self.item_replies: dict[str, Reply] = {}
        self.mutation: asyncio.Task | None = None

    def remember(self, item, reply: Reply | None = None) -> None:
        self.items[item.id] = item
        if reply is not None:
            self.item_replies[item.id] = reply

    def spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    async def close(self) -> None:
        tasks = list(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def start(self, service: RealtimeService, request: GenerateResponseRequest,
              regenerate_response_id: str | None = None) -> Reply:
        previous = [reply.receipt_done for reply in self.replies.values()
                    if reply.item_id is not None and not reply.receipt_done.is_set()]
        reply = self.active = self.replies[request.response_key] = Reply(regenerating=regenerate_response_id is not None)
        while len(self.replies) > 32:
            self.replies.pop(next(iter(self.replies)))
        mutation = self.mutation if self.mutation is not None and not self.mutation.done() else None
        reply.generation = self.spawn(self._generate(service, request, reply, previous, self.input, mutation,
                                                     regenerate_response_id))
        return reply

    async def _generate(self, service, request, reply, previous, user_input, mutation, regenerate_response_id) -> None:
        queue = service.context_output_queue
        scope = service.context_cancel_scope
        tags = dict(response_key=request.response_key, cancel_generation=scope.generation,
                    turn_id=request.turn_id, turn_revision=request.turn_revision)
        status, error = "failed", None
        try:
            if mutation is not None:
                await asyncio.shield(mutation)
                if user_input and user_input[0] in self.items and self.input and user_input[0] == self.input[0]:
                    user_input = self.input
                elif user_input and user_input[0] not in self.items:
                    user_input = None
            for receipt in previous:
                await asyncio.wait_for(receipt.wait(), timeout=10)
            tracker = service.speculative_turns
            while tracker is not None:
                gate = tracker.gate(request.turn_id, request.turn_revision)
                if gate.action is TurnGateAction.DROP:
                    status = "cancelled"
                    break
                if gate.action is not TurnGateAction.HOLD:
                    break
                await asyncio.sleep(0.02)
            if status != "cancelled":
                item_id, transcript, language = user_input or (uuid4().hex, "", None)
                if regenerate_response_id is None:
                    if self.confirmed_turn_id == item_id:
                        item_id, transcript = uuid4().hex, ""
                    self.revision = max(request.turn_revision or 0, self.revision + 1 if item_id == self.last_turn_id else 0)
                    self.last_turn_id = item_id
                    turn_id = f"{request.runtime_config.trace_session_id}:{item_id}"
                    turn = reply.turn = ContextTurn(turn_id, self.revision, reply.response_id)
                    if item_id in self.items:
                        self.item_replies[item_id] = reply
                    options = {}
                else:
                    # The gateway resolves the original turn and assigns its next revision.
                    turn = ContextTurn(reply.response_id, 0, reply.response_id)
                    transcript = ""
                    options = {"regenerate_response_id": regenerate_response_id}
                voice = service.response._response_voice_id(request.runtime_config.session.audio.output.voice)
                selected = request.runtime_config.selected_language
                if selected not in (None, "auto"):
                    language = selected
                async with aclosing(self.client.stream(turn, transcript=transcript, language=language, voice=voice,
                                                       **options)) as stream:
                    async for event in stream:
                        if event.type == "ready":
                            reply.turn = event.turn
                        elif event.type == "text":
                            queue.put(LLMResponseChunk(text=event.text, language_code=language,
                                                       runtime_config=request.runtime_config, external_delta=True, **tags))
                        else:
                            status = event.type
                if status == "completed" and regenerate_response_id is not None and reply.turn is not None:
                    if item_id in self.items and reply.turn.turn_id == f"{request.runtime_config.trace_session_id}:{item_id}":
                        self.item_replies[item_id] = reply
                        self.last_turn_id, self.revision = item_id, reply.turn.revision
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Context generation failed (%s)", type(exc).__name__)
            error = "External context generation failed"
        if status == "failed":
            error = error or "External context generation failed"
        queue.put(EndOfResponse(external_delta=True, cleanup_only=status == "cancelled", error=error, **tags))

    def cancel(self, *, notify_gateway: bool = True) -> None:
        reply = self.active
        if reply is None or reply.status is not None:
            return
        if reply.generation is not None and not reply.generation.done():
            reply.generation.cancel()
        if notify_gateway and reply.turn is not None:
            self.spawn(self._cancel(reply.turn))

    async def _cancel(self, turn: ContextTurn) -> None:
        try:
            await asyncio.wait_for(self.client.cancel(turn), timeout=5)
        except Exception as exc:
            logger.warning("Context cancellation failed (%s)", type(exc).__name__)

    async def receipt(self, reply: Reply, text: str, played_ms: int) -> None:
        try:
            assert reply.turn is not None
            await asyncio.wait_for(self.client.receipt(reply.turn, text=text, played_ms=played_ms, final=True), timeout=5)
        except Exception as exc:
            reply.receipt_sent = False
            logger.warning("Context playback receipt failed (%s)", type(exc).__name__)
            return
        if text and self.input is not None:
            item_id = self.input[0]
            if reply.turn.turn_id.endswith(":" + item_id):
                self.confirmed_turn_id = item_id
        reply.receipt_done.set()
