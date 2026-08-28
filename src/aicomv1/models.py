from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from threading import Event
from uuid import uuid4


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def as_ollama(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}


@dataclass(slots=True)
class ConversationSession:
    id: str = field(default_factory=lambda: uuid4().hex)
    language: str = "en"
    messages: list[Message] = field(default_factory=list)
    summary: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    audio_directory: Path | None = None
    active_cancel: Event | None = None

    def add(self, role: Role, content: str) -> None:
        clean = " ".join(content.split()).strip()
        if clean:
            self.messages.append(Message(role=role, content=clean))
            self.updated_at = datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class Transcription:
    text: str
    elapsed_ms: int
    provider: str


@dataclass(frozen=True, slots=True)
class SpeechAudio:
    path: Path
    elapsed_ms: int
    provider: str
    sample_rate: int


@dataclass(frozen=True, slots=True)
class ComponentStatus:
    name: str
    ready: bool
    detail: str
