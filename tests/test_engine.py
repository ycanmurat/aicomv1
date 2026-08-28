from __future__ import annotations

import asyncio
import wave
from collections.abc import AsyncIterator
from pathlib import Path
from threading import Event

import pytest

from aicomv1.engine import AssistantEngine
from aicomv1.knowledge import KnowledgeStore
from aicomv1.memory import SessionStore
from aicomv1.models import ComponentStatus, Role, SpeechAudio
from aicomv1.providers.base import TTSError
from aicomv1.tools import LocalToolRouter


class FakeLLM:
    def __init__(self, *, language: str = "en") -> None:
        self.language = language
        self.seen_system = ""
        self.seen_messages: list[dict[str, str]] = []
        self.summary_system = ""
        self.summary_messages: list[dict[str, str]] = []

    async def stream(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        cancel: Event,
        reasoning: bool = False,
    ) -> AsyncIterator[str]:
        self.seen_system = system_prompt
        self.seen_messages = messages
        chunks = (
            ("First answer complete. ", "Second answer complete.")
            if self.language == "en"
            else ("İlk cevap tamam. ", "İkinci cevap de tamam.")
        )
        for chunk in chunks:
            await asyncio.sleep(0)
            yield chunk

    async def complete(
        self, *, system_prompt: str, messages: list[dict[str, str]], max_tokens: int
    ) -> str:
        self.summary_system = system_prompt
        self.summary_messages = messages
        return "Summary." if self.language == "en" else "Özet."

    async def status(self) -> ComponentStatus:
        return ComponentStatus("fake-llm", True, "Ready")


class FakeTTS:
    def __init__(self) -> None:
        self.languages: list[str] = []
        self.status_languages: list[str] = []

    def status(self, language: str = "tr") -> ComponentStatus:
        self.status_languages.append(language)
        return ComponentStatus("fake-tts", True, "Ready")

    def synthesize(self, text: str, output_path: Path, language: str = "tr") -> SpeechAudio:
        self.languages.append(language)
        with wave.open(str(output_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24_000)
            wav.writeframes(b"\0\0" * 80)
        return SpeechAudio(output_path, 4, "fake-tts", 24_000)


@pytest.mark.parametrize("language", ["en", "tr"])
async def test_engine_streams_text_and_clause_audio(tmp_path: Path, language: str) -> None:
    knowledge = KnowledgeStore(tmp_path / "knowledge.sqlite3")
    knowledge.add(title="Özel bilgi", body="Projenin kod adı Kutup Yıldızı.")
    llm = FakeLLM(language=language)
    tts = FakeTTS()
    engine = AssistantEngine(llm, tts, LocalToolRouter(knowledge))
    session = SessionStore(tmp_path / "audio").create(language=language)
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
    expected = (
        "First answer complete. Second answer complete."
        if language == "en"
        else "İlk cevap tamam. İkinci cevap de tamam."
    )
    assert session.messages[-1].content == expected
    assert "Kutup Yıldızı" in llm.seen_system
    assert f"conversation language is {'English' if language == 'en' else 'Turkish'}" in (
        llm.seen_system
    )
    assert tts.languages == [language, language]
    assert tts.status_languages == [language]


async def test_engine_captures_language_for_whole_turn(tmp_path: Path) -> None:
    session = SessionStore(tmp_path / "audio").create(language="en")
    tts = FakeTTS()
    llm = FakeLLM()
    engine = AssistantEngine(llm, tts, LocalToolRouter(KnowledgeStore(tmp_path / "kb.sqlite3")))

    async def emit(event: dict[str, object]) -> None:
        if event["type"] == "text_delta":
            session.language = "tr"

    await engine.respond(
        session=session, user_text="Hello", turn_id="captured", cancel=Event(), emit=emit
    )
    assert tts.languages == ["en", "en"]
    assert "conversation language is English" in llm.seen_system


@pytest.mark.parametrize("language", ["en", "tr"])
async def test_summary_preserves_previous_memory_in_selected_language(
    tmp_path: Path, language: str
) -> None:
    session = SessionStore(tmp_path / "audio").create(language=language)
    session.summary = "The user prefers short answers."
    for index in range(20):
        session.add(Role.USER if index % 2 == 0 else Role.ASSISTANT, f"Message {index}")
    llm = FakeLLM(language=language)
    engine = AssistantEngine(llm, FakeTTS(), LocalToolRouter(KnowledgeStore(tmp_path / "kb.db")))
    await engine._compact_history(session, language=language)
    assert ("English" if language == "en" else "Turkish") in llm.summary_system
    assert "The user prefers short answers." in llm.summary_messages[0]["content"]
    assert len(session.messages) == 10
    assert session.summary == ("Summary." if language == "en" else "Özet.")


async def test_speech_failure_does_not_discard_text(tmp_path: Path) -> None:
    class UnavailableTTS(FakeTTS):
        def synthesize(self, text: str, output_path: Path, language: str = "tr") -> SpeechAudio:
            raise TTSError("Requested local voice is missing.")

    session = SessionStore(tmp_path / "audio").create(language="en")
    engine = AssistantEngine(
        FakeLLM(), UnavailableTTS(), LocalToolRouter(KnowledgeStore(tmp_path / "kb.db"))
    )
    events: list[dict[str, object]] = []

    async def emit(event: dict[str, object]) -> None:
        events.append(event)

    await engine.respond(
        session=session, user_text="Hello", turn_id="no-voice", cancel=Event(), emit=emit
    )
    assert any(event.get("code") == "speech_synthesis_failed" for event in events)
    assert any(event["type"] == "text_done" and event["text"] for event in events)
    assert not any(event["type"] == "audio" for event in events)
