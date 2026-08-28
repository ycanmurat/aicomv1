from __future__ import annotations

import shutil
from pathlib import Path
from threading import RLock

from aicomv1.models import ConversationSession, Message
from aicomv1.prompt import normalize_language


class SessionNotFoundError(KeyError):
    pass


class SessionStore:
    """Manage in-memory conversations and their private audio directories."""

    def __init__(self, audio_root: Path) -> None:
        self.audio_root = audio_root.resolve()
        self.audio_root.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, ConversationSession] = {}
        self._lock = RLock()

    def create(self, *, language: str = "en") -> ConversationSession:
        with self._lock:
            session = ConversationSession(language=normalize_language(language))
            session_directory = (self.audio_root / session.id).resolve()
            if session_directory.parent != self.audio_root:
                raise ValueError("Unsafe session directory was rejected.")
            session_directory.mkdir(mode=0o700)
            session.audio_directory = session_directory
            self._sessions[session.id] = session
            return session

    def get(self, session_id: str) -> ConversationSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(session_id)
            return session

    def delete(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None or session.audio_directory is None:
            return
        if session.active_cancel is not None:
            session.active_cancel.set()
        target = session.audio_directory.resolve()
        if target.parent != self.audio_root or target.name != session.id:
            raise ValueError("Unsafe session deletion request was rejected.")
        shutil.rmtree(target, ignore_errors=True)


def prompt_messages(
    session: ConversationSession, *, recent_limit: int = 14, language: str | None = None
) -> list[dict[str, str]]:
    language = normalize_language(session.language if language is None else language)
    messages: list[dict[str, str]] = []
    if session.summary:
        messages.append(
            {
                "role": "system",
                "content": (
                    f"Local memory note from earlier conversation:\n{session.summary}"
                    if language == "en"
                    else f"Önceki konuşmadan yerel hafıza notu:\n{session.summary}"
                ),
            }
        )
    messages.extend(message.as_ollama() for message in session.messages[-recent_limit:])
    return messages


def old_messages_for_summary(
    session: ConversationSession, *, keep_recent: int = 10, compact_after: int = 18
) -> list[Message]:
    if len(session.messages) < compact_after:
        return []
    return list(session.messages[:-keep_recent])


def apply_summary(session: ConversationSession, summary: str, *, keep_recent: int = 10) -> None:
    clean = " ".join(summary.split()).strip()
    if not clean:
        return
    session.summary = clean
    session.messages[:] = session.messages[-keep_recent:]
