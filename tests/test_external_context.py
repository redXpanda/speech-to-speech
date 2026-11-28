import asyncio
from contextlib import aclosing
from unittest import IsolatedAsyncioTestCase

from speech_to_speech.pipeline.external_context import ContextEvent, ContextProtocolError, ContextTurn, ExternalContextClient


class ContextTransportTests(IsolatedAsyncioTestCase):
    async def test_stream_routes_by_response_and_control_waits_for_its_ack(self):
        sent = asyncio.Queue()

        async def send(frame):
            sent.put_nowait(frame)

        client = ExternalContextClient(send)
        identity = {"turn_id": "turn", "revision": 3, "response_id": "response"}
        turn = ContextTurn(**identity)
        async with aclosing(client.stream(turn, transcript="question", language="en", voice=None)) as stream:
            first = asyncio.create_task(anext(stream))
            self.assertEqual(await sent.get(), {"type": "aivoice.context.request", **identity,
                                                 "transcript": "question", "language": "en", "voice": None})
            client.receive({"type": "aivoice.context.text", "response_id": "other", "seq": 0, "text": "wrong"})
            client.receive({"type": "aivoice.context.text", "response_id": "response", "seq": 0, "text": "hello"})
            self.assertEqual(await first, ContextEvent("text", "hello"))
            cancel = asyncio.create_task(client.cancel(turn))
            receipt = asyncio.create_task(client.receipt(turn, text="hello", played_ms=400, final=True))
            cancel_frame, receipt_frame = await sent.get(), await sent.get()
            cancel_id, receipt_id = cancel_frame.pop("control_id"), receipt_frame.pop("control_id")
            self.assertEqual(cancel_frame, {"type": "aivoice.context.cancel", **identity})
            self.assertEqual(receipt_frame, {"type": "aivoice.context.receipt", **identity,
                                             "text": "hello", "played_ms": 400, "final": True})
            client.receive({"type": "aivoice.context.ack", "control_id": receipt_id, "ok": True})
            await receipt
            self.assertFalse(cancel.done())
            client.receive({"type": "aivoice.context.ack", "control_id": cancel_id, "ok": False})
            with self.assertRaises(ContextProtocolError):
                await cancel
            client.receive({"type": "aivoice.context.terminal", "response_id": "response", "status": "cancelled"})
            self.assertEqual(await anext(stream), ContextEvent("cancelled"))
            with self.assertRaises(StopAsyncIteration):
                await anext(stream)
        self.assertFalse(client._streams)
        self.assertFalse(client._acks)

    async def test_invalid_stream_frame_cleans_up_routing(self):
        turn = ContextTurn("turn", 0, "response")
        for frame in ({"type": "aivoice.context.text", "seq": 1, "text": "bad"},
                      {"type": "aivoice.context.terminal", "status": "unknown"}):
            async def send(_):
                client.receive({**frame, "response_id": turn.response_id})

            client = ExternalContextClient(send)
            with self.assertRaises(ContextProtocolError):
                await anext(client.stream(turn, transcript="", language=None, voice=None))
            self.assertFalse(client._streams)

    async def test_cancelling_wait_removes_stream_and_ignores_late_frames(self):
        sent = asyncio.Event()

        async def send(_):
            sent.set()

        client = ExternalContextClient(send)
        turn = ContextTurn("turn", 0, "response")
        task = asyncio.create_task(anext(client.stream(turn, transcript="", language=None, voice=None)))
        await sent.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        client.receive({"type": "aivoice.context.terminal", "response_id": "response", "status": "completed"})
        self.assertFalse(client._streams)
