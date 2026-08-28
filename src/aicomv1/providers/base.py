from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from threading import Event
from typing import Protocol

from aicomv1.models import ComponentStatus, SpeechAudio, Transcription


class LLMError(RuntimeError):
    pass


class STTError(RuntimeError):
    pass


class TTSError(RuntimeError):
    pass


class ChatProvider(Protocol):
    async def stream(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        cancel: Event,
        reasoning: bool = False,
    ) -> AsyncIterator[str]: ...

    async def complete(
        self, *, system_prompt: str, messages: list[dict[str, str]], max_tokens: int
    ) -> str: ...

    async def status(self) -> ComponentStatus: ...


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path, language: str = "tr") -> Transcription: ...

    def status(self) -> ComponentStatus: ...


class Synthesizer(Protocol):
    def synthesize(self, text: str, output_path: Path, language: str = "tr") -> SpeechAudio: ...

    def status(self, language: str = "tr") -> ComponentStatus: ...
