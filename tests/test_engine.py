from __future__ import annotations

import asyncio
import wave
from collections.abc import AsyncIterator
from pathlib import Path
from threading import Event

from aicomv1.engine import AssistantEngine
from aicomv1.knowledge import KnowledgeStore
from aicomv1.memory import SessionStore
from aicomv1.models import ComponentStatus, SpeechAudio
from aicomv1.tools import LocalToolRouter


class FakeLLM:
    seen_system = ""

    async def stream(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        cancel: Event,
        reasoning: bool = False,
    ) -> AsyncIterator[str]:
        self.seen_system = system_prompt
        for chunk in ("İlk cevap tamam. ", "İkinci cevap de tamam."):
            await asyncio.sleep(0)
            yield chunk

    async def complete(
        self, *, system_prompt: str, messages: list[dict[str, str]], max_tokens: int
    ) -> str:
        return "özet"

    async def status(self) -> ComponentStatus:
        return ComponentStatus("fake-llm", True, "hazır")


class FakeTTS:
    def status(self) -> ComponentStatus:
        return ComponentStatus("fake-tts", True, "hazır")

    def synthesize(self, text: str, output_path: Path) -> SpeechAudio:
        with wave.open(str(output_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24_000)
            wav.writeframes(b"\0\0" * 80)
        return SpeechAudio(output_path, 4, "fake-tts", 24_000)


async def test_engine_streams_text_and_clause_audio(tmp_path: Path) -> None:
    knowledge = KnowledgeStore(tmp_path / "knowledge.sqlite3")
    knowledge.add(title="Özel bilgi", body="Projenin kod adı Kutup Yıldızı.")
    llm = FakeLLM()
    engine = AssistantEngine(llm, FakeTTS(), LocalToolRouter(knowledge))
    session = SessionStore(tmp_path / "audio").create()
    events: list[dict[str, object]] = []

    async def emit(event: dict[str, object]) -> None:
        events.append(event)

    await engine.respond(
        session=session,
        user_text="Projenin kod adı ne?",
        turn_id="turn-1",
        cancel=Event(),
        emit=emit,
    )

    types = [str(event["type"]) for event in events]
    assert types.count("audio") == 2
    assert types.index("text_delta") < types.index("audio") < types.index("text_done")
    assert session.messages[-1].content == "İlk cevap tamam. İkinci cevap de tamam."
    assert "Kutup Yıldızı" in llm.seen_system
