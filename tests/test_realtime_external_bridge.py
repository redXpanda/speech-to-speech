import asyncio
from queue import Queue
from threading import Event
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock

from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
from speech_to_speech.api.openai_realtime.handlers.audio import PIPELINE_OUTPUT_RATE
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.api.openai_realtime.websocket_router import _dispatch_client_event
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.external_context import ContextEvent, ContextTurn
from speech_to_speech.api.openai_realtime.context import Reply
from openai.types.realtime.realtime_conversation_item_assistant_message import (
    Content, RealtimeConversationItemAssistantMessage,
)
from speech_to_speech.pipeline.events import (
    AssistantOutputEvent, PartialTranscriptionEvent, SpeechStartedEvent, SpeechStoppedEvent, TranscriptionCompletedEvent,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.pipeline.messages import EndOfResponse, TTSInput


class ExternalBridgeTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service = RealtimeService(text_prompt_queue=Queue(), default_instructions="legacy prompt")
        self.scope = self.service.context_cancel_scope = CancelScope()
        self.queue = self.service.context_output_queue = Queue()
        self.conn = self.service.register()
        self.st = self.service._state(self.conn)
        self.transport = SimpleNamespace(send_events=AsyncMock(), send_context=AsyncMock(), discard_pending_audio=lambda: 0)
        self.unit = SimpleNamespace(service=self.service, cancel_scope=self.scope,
                                    text_prompt_queue=self.service.text_prompt_queue,
                                    output_queue=Queue(), text_output_queue=Queue(), response_playing=Event())
        await self.dispatch({"type": "session.update", "session": {"type": "realtime"},
                             "aivoice_context": {"transport": "websocket"}})
        self.context = self.st.external_context
        self.addAsyncCleanup(self.context.close)
        self.assertIsNone(self.st.runtime_config.session.instructions)
        setup = self.transport.send_events.await_args.args[0][0].model_dump()
        self.assertTrue(setup["aivoice_context_ready"])

    async def dispatch(self, event):
        await _dispatch_client_event(self.unit, self.conn, event, self.transport)

    async def test_streams_before_model_end_and_waits_for_confirmed_playback(self):
        wire = asyncio.Queue()
        self.transport.send_context.side_effect = wire.put_nowait

        async def fragment(response_id, seq, text):
            await self.dispatch({"type": "aivoice.context.text", "response_id": response_id,
                                 "seq": seq, "text": text})

        async def complete(response_id):
            await self.dispatch({"type": "aivoice.context.terminal", "response_id": response_id,
                                 "status": "completed"})

        await self.dispatch({"type": "conversation.item.create", "item": {"type": "message", "role": "user",
                             "content": [{"type": "input_text", "text": "question"}]}})
        await self.dispatch({"type": "response.create"})
        request = await asyncio.wait_for(wire.get(), 1)
        first = self.context.active
        self.assertEqual((request["type"], request["response_id"], request["transcript"]),
                         ("aivoice.context.request", first.response_id, "question"))
        await fragment(first.response_id, 0, "**First** sentence. Second ")

        async def next_chunk():
            while self.queue.empty():
                await asyncio.sleep(0)
            return self.queue.get_nowait()

        processor = LMOutputProcessor(Event(), queue_in=Queue(), queue_out=Queue())
        output = list(processor.process(await asyncio.wait_for(next_chunk(), 1)))
        self.assertEqual([item.text for item in output if isinstance(item, TTSInput)], ["First sentence."])
        self.assertTrue(self.service.text_prompt_queue.empty())
        self.assertFalse(first.generation.done())
        self.service.dispatch_pipeline_event(self.conn, output[0])
        _, item_id, _, _ = self.service.begin_audio_output(self.conn, self.st.current_response_key, PIPELINE_OUTPUT_RATE)
        await fragment(first.response_id, 1, "sentence.")
        await complete(first.response_id)
        await first.generation
        while not self.queue.empty():
            for item in processor.process(self.queue.get_nowait()):
                if isinstance(item, TTSInput):
                    self.assertEqual(item.text, "Second sentence.")
                    self.service.begin_audio_output(self.conn, self.st.current_response_key, PIPELINE_OUTPUT_RATE)
                elif not isinstance(item, EndOfResponse):
                    self.service.dispatch_pipeline_event(self.conn, item)
        self.service.finish_response(self.conn)
        self.assertTrue(wire.empty())
        await self.dispatch({"type": "aivoice.playback.receipt", "item_id": item_id,
                             "response_id": first.response_id, "played_ms": 1500})
        await self.dispatch({"type": "response.create"})
        second = self.context.active
        receipt = await asyncio.wait_for(wire.get(), 1)
        self.assertEqual((receipt["type"], receipt["response_id"], receipt["text"], receipt["played_ms"], receipt["final"]),
                         ("aivoice.context.receipt", first.response_id, "First sentence.", 1500, True))
        await asyncio.sleep(0)
        self.assertTrue(wire.empty())
        await self.dispatch({"type": "aivoice.context.ack", "control_id": receipt["control_id"], "ok": True})
        request = await asyncio.wait_for(wire.get(), 1)
        self.assertEqual((request["type"], request["response_id"], request["transcript"]),
                         ("aivoice.context.request", second.response_id, ""))
        await complete(second.response_id)
        await asyncio.wait_for(second.generation, 1)
        await self.dispatch({"type": "conversation.item.retrieve", "item_id": item_id})
        projection = self.transport.send_events.await_args.args[0][0]
        self.assertEqual(projection.item.content[0].transcript, "First sentence.")
        self.assertIsNone(self.st.runtime_config.chat.get_item(item_id))

    async def test_unsubmitted_user_edit_keeps_wire_id_and_becomes_first_request(self):
        wire = asyncio.Queue()
        self.transport.send_context.side_effect = wire.put_nowait
        await self.dispatch({"type": "conversation.item.create", "item": {"type": "message", "role": "user",
                             "content": [{"type": "input_text", "text": "original"}]}})
        item_id = self.context.input[0]
        await self.dispatch({"type": "aivoice.item.mutate", "item_id": item_id, "event_id": "user-edit",
                             "op": "edit", "text": "corrected"})
        self.assertEqual(await asyncio.wait_for(wire.get(), 1),
                         {"type": "aivoice.item.mutation.done", "event_id": "user-edit", "ok": True})
        await self.dispatch({"type": "conversation.item.retrieve", "item_id": item_id})
        item = self.transport.send_events.await_args.args[0][0].item
        self.assertEqual((item.id, item.content[0].text), (item_id, "corrected"))
        await self.dispatch({"type": "response.create"})
        request = await asyncio.wait_for(wire.get(), 1)
        self.assertEqual((request["type"], request["transcript"]), ("aivoice.context.request", "corrected"))
        self.assertTrue(request["turn_id"].endswith(":" + item_id))
        self.assertIs(self.context.item_replies[item_id], self.context.active)

    async def test_old_multi_item_reply_mutation_waits_for_ack_and_seals_playback(self):
        wire = asyncio.Queue()
        self.transport.send_context.side_effect = wire.put_nowait
        reply = Reply(turn=ContextTurn("session:old-user", 0, "old-response"), item_id="old-audio", status="completed")
        self.context.replies["reply"] = reply
        for item_id in ["old-audio", "old-part", *(f"later-{i}" for i in range(33))]:
            item = RealtimeConversationItemAssistantMessage(
                id=item_id, type="message", role="assistant", content=[Content(type="output_audio", transcript="old")],
            )
            self.context.remember(item, reply if item_id.startswith("old-") else None)
        self.st.assistant_audio_segments["old-audio"] = [("old", 1000)]
        self.st.assistant_audio_cuts["old-part"] = 1000

        async def mutate(op, ok=True, **fields):
            await self.dispatch({"type": "aivoice.item.mutate", "item_id": "old-audio", "event_id": op,
                                 "op": op, **fields})
            command = await asyncio.wait_for(wire.get(), 1)
            self.assertEqual((command["type"], command["turn_id"], command["role"], command["op"]),
                             ("aivoice.context.mutate", "session:old-user", "assistant", op))
            self.assertIn("old-audio", self.context.items)
            await self.dispatch({"type": "aivoice.context.ack", "control_id": command["control_id"], "ok": ok})
            self.assertEqual(await asyncio.wait_for(wire.get(), 1),
                             {"type": "aivoice.item.mutation.done", "event_id": op, "ok": ok})

        await mutate("edit", ok=False, text="rejected")
        self.assertEqual(self.context.items["old-audio"].content[0].transcript, "old")
        await mutate("edit", text="corrected")
        self.assertTrue(reply.receipt_done.is_set())
        await self.dispatch({"type": "conversation.item.truncate", "item_id": "old-audio", "content_index": 0,
                             "audio_end_ms": 10})
        await self.dispatch({"type": "aivoice.playback.receipt", "item_id": "old-audio",
                             "response_id": reply.response_id, "played_ms": 10})
        self.assertTrue(wire.empty())
        await self.dispatch({"type": "conversation.item.retrieve", "item_id": "old-audio"})
        self.assertEqual(self.transport.send_events.await_args.args[0][0].item.content[0].text, "corrected")
        self.assertEqual(self.context.items["old-part"].content, [])
        self.assertNotIn("old-audio", self.st.assistant_audio_segments)
        self.assertNotIn("old-part", self.st.assistant_audio_cuts)
        await mutate("delete")
        self.assertFalse(self.context.items.keys() & {"old-audio", "old-part"})
        await self.dispatch({"type": "conversation.item.retrieve", "item_id": "old-audio"})
        self.assertEqual(self.transport.send_events.await_args.args[0][0].type, "error")

    async def test_cancel_is_local_before_gateway_confirmation_and_retry_increments_revision(self):
        started = asyncio.Event()
        closed = asyncio.Event()
        cancel_dispatched = asyncio.Event()

        async def model(turn, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
                yield ContextEvent("completed")
            finally:
                closed.set()

        async def cancel(turn):
            cancel_dispatched.set()
            await asyncio.Event().wait()

        self.context.client.stream = model
        self.context.client.cancel = cancel
        self.context.input = ("user_item", "question", "en")
        await self.dispatch({"type": "response.create"})
        await asyncio.wait_for(started.wait(), 1)
        first = self.context.active
        generation = self.scope.generation
        await self.dispatch({"type": "response.cancel"})
        self.assertTrue(self.scope.is_stale(generation))
        self.assertFalse(self.st.in_response)
        await asyncio.wait_for(closed.wait(), 1)
        await asyncio.wait_for(cancel_dispatched.wait(), 1)
        started.clear()
        await self.dispatch({"type": "response.create"})
        await asyncio.wait_for(started.wait(), 1)
        second = self.context.active
        self.assertEqual(second.turn.turn_id, first.turn.turn_id)
        self.assertGreater(second.turn.revision, first.turn.revision)
        self.assertNotEqual(second.response_id, first.response_id)

    async def test_final_stt_preview_precedes_external_reply_and_preserves_terminal(self):
        snapshot = "speech_to_speech.input_audio_transcription.snapshot"
        completed = "conversation.item.input_audio_transcription.completed"
        await self.dispatch({"type": "session.update", "session": {"type": "realtime", "extensions": [snapshot]}})
        tracker = self.service.speculative_turns = SpeculativeTurnTracker()
        turn_id, revision = tracker.start_turn()
        turn = {"turn_id": turn_id, "turn_revision": revision}

        def emit(event):
            return self.service.dispatch_pipeline_event(self.conn, event)

        started, = emit(SpeechStartedEvent(**turn))
        partial, = emit(PartialTranscriptionEvent(delta="spoken", **turn))
        self.assertEqual(partial.type, snapshot)
        self.assertFalse(partial.asr_complete)
        tracker.segment_finalized(100)
        emit(SpeechStoppedEvent(duration_s=0.1, **turn))
        preview, = emit(TranscriptionCompletedEvent(transcript="spoken question", **turn))
        self.assertEqual((preview.type, preview.asr_complete, preview.item_id, preview.transcript),
                         (snapshot, True, started.item_id, "spoken question"))
        self.assertEqual(self.service.audio.resolve_input_terminals(self.conn), [])
        self.assertFalse(tracker.is_committed(turn_id, revision))
        self.assertFalse(self.context.active.generation.done())

        response_key, = self.st.pending_response_keys
        output = emit(AssistantOutputEvent(text="First reply.", response_key=response_key, **turn))
        terminal, = [event for event in output if event.type == completed]
        self.assertEqual((terminal.item_id, terminal.transcript), (preview.item_id, preview.transcript))
        delta, = [event for event in output if event.type in (
            "response.output_text.delta", "response.output_audio_transcript.delta",
        )]
        self.assertEqual(delta.delta, "First reply.")
        self.assertLess(output.index(terminal), output.index(delta))
        self.assertEqual(self.service.audio.resolve_input_terminals(self.conn), [])

    async def test_final_stt_uses_external_path_and_vad_does_not_wait_for_cancel(self):
        began = asyncio.Event()
        cancel_started = asyncio.Event()

        async def stream(turn, *, transcript, language, voice):
            self.assertEqual((transcript, language), ("spoken question", "en"))
            began.set()
            await asyncio.Event().wait()
            yield ContextEvent("completed")

        async def cancel(turn):
            cancel_started.set()
            await asyncio.Event().wait()

        self.context.client.stream = stream
        self.context.client.cancel = cancel
        self.service.dispatch_pipeline_event(self.conn, SpeechStartedEvent(turn_id="speech", turn_revision=0))
        self.service.dispatch_pipeline_event(self.conn, SpeechStoppedEvent(turn_id="speech", turn_revision=0))
        self.service.dispatch_pipeline_event(self.conn, TranscriptionCompletedEvent(
            transcript="spoken question", language_code="en", turn_id="speech", turn_revision=0,
        ))
        await asyncio.wait_for(began.wait(), 1)
        self.assertTrue(self.service.text_prompt_queue.empty())
        reply = self.context.active
        self.service.begin_audio_output(self.conn, next(iter(self.st.pending_response_keys)), PIPELINE_OUTPUT_RATE)
        events = self.service.dispatch_pipeline_event(self.conn, SpeechStartedEvent(turn_id="next", turn_revision=0))
        self.assertEqual(reply.status, "cancelled")
        self.assertIn("input_audio_buffer.speech_started", [event.type for event in events])
        await asyncio.wait_for(cancel_started.wait(), 1)
