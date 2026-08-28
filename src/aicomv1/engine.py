from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from threading import Event
from uuid import uuid4

from aicomv1.memory import (
    apply_summary,
    old_messages_for_summary,
    prompt_messages,
)
from aicomv1.models import ConversationSession, Role
from aicomv1.prompt import SUMMARY_PROMPT, SYSTEM_PROMPT
from aicomv1.providers.base import ChatProvider, Synthesizer, TTSError
from aicomv1.text import ClauseSegmenter, clean_spoken_text
from aicomv1.tools import LocalToolRouter

EventSink = Callable[[dict[str, object]], Awaitable[None]]


class AssistantEngine:
    def __init__(self, llm: ChatProvider, tts: Synthesizer, tools: LocalToolRouter) -> None:
        self.llm = llm
        self.tts = tts
        self.tools = tools

    async def respond(
        self,
        *,
        session: ConversationSession,
        user_text: str,
        turn_id: str,
        cancel: Event,
        emit: EventSink,
    ) -> None:
        clean_user = " ".join(user_text.split()).strip()
        if not clean_user or cancel.is_set():
            return
        session.add(Role.USER, clean_user)
        tool_context = self.tools.context_for(clean_user)
        system_prompt = SYSTEM_PROMPT
        if not tool_context.empty:
            system_prompt += "\n\n" + tool_context.render()

        started = time.perf_counter()
        first_token_ms: int | None = None
        first_audio_ms: int | None = None
        full_text: list[str] = []
        segmenter = ClauseSegmenter()
        speech_queue: asyncio.Queue[tuple[int, str] | None] = asyncio.Queue()

        async def speech_worker() -> None:
            nonlocal first_audio_ms
            if self.tts.status().name == "tts-none":
                return
            while True:
                item = await speech_queue.get()
                if item is None:
                    return
                index, segment = item
                if cancel.is_set():
                    continue
                if session.audio_directory is None:
                    continue
                filename = f"{turn_id}-{index:03d}-{uuid4().hex[:8]}.wav"
                output_path = (session.audio_directory / filename).resolve()
                if output_path.parent != session.audio_directory.resolve():
                    continue
                try:
                    audio = await asyncio.to_thread(self.tts.synthesize, segment, output_path)
                except TTSError as exc:
                    await emit(
                        {
                            "type": "warning",
                            "turn_id": turn_id,
                            "message": f"Ses üretilemedi: {exc}",
                        }
                    )
                    continue
                if cancel.is_set():
                    continue
                if first_audio_ms is None:
                    first_audio_ms = round((time.perf_counter() - started) * 1000)
                await emit(
                    {
                        "type": "audio",
                        "turn_id": turn_id,
                        "index": index,
                        "text": segment,
                        "filename": audio.path.name,
                        "provider": audio.provider,
                        "synthesis_ms": audio.elapsed_ms,
                        "sample_rate": audio.sample_rate,
                    }
                )

        worker = asyncio.create_task(speech_worker())
        segment_index = 0
        try:
            await emit({"type": "state", "state": "thinking", "turn_id": turn_id})
            async for delta in self.llm.stream(
                system_prompt=system_prompt,
                messages=prompt_messages(session),
                cancel=cancel,
            ):
                if cancel.is_set():
                    break
                if first_token_ms is None:
                    first_token_ms = round((time.perf_counter() - started) * 1000)
                full_text.append(delta)
                await emit({"type": "text_delta", "turn_id": turn_id, "delta": delta})
                for segment in segmenter.push(delta):
                    await speech_queue.put((segment_index, segment))
                    segment_index += 1
            if not cancel.is_set():
                for segment in segmenter.finish():
                    await speech_queue.put((segment_index, segment))
                    segment_index += 1
        finally:
            if cancel.is_set():
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
            else:
                await speech_queue.put(None)
                await worker

        answer = clean_spoken_text("".join(full_text))
        if cancel.is_set():
            await emit({"type": "interrupted", "turn_id": turn_id})
            return
        if answer:
            session.add(Role.ASSISTANT, answer)
        await emit({"type": "text_done", "turn_id": turn_id, "text": answer})
        await emit(
            {
                "type": "metrics",
                "turn_id": turn_id,
                "first_token_ms": first_token_ms,
                "first_audio_ms": first_audio_ms,
                "total_ms": round((time.perf_counter() - started) * 1000),
            }
        )
        await emit({"type": "state", "state": "listening", "turn_id": turn_id})
        await self._compact_history(session)

    async def _compact_history(self, session: ConversationSession) -> None:
        old = old_messages_for_summary(session)
        if not old:
            return
        transcript = "\n".join(f"{message.role.value}: {message.content}" for message in old)
        try:
            summary = await self.llm.complete(
                system_prompt=SUMMARY_PROMPT,
                messages=[{"role": "user", "content": transcript}],
                max_tokens=240,
            )
        except Exception:
            return
        apply_summary(session, summary)
