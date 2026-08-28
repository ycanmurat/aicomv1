from __future__ import annotations

import shutil
from pathlib import Path
from threading import RLock

from aicomv1.models import ConversationSession, Message


class SessionNotFoundError(KeyError):
    pass


class SessionStore:
    """Tek cihazdaki görüşmeleri ve her görüşmenin özel ses dizinini yönetir."""

    def __init__(self, audio_root: Path) -> None:
        self.audio_root = audio_root.resolve()
        self.audio_root.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, ConversationSession] = {}
        self._lock = RLock()

    def create(self) -> ConversationSession:
        with self._lock:
            session = ConversationSession()
            session_directory = (self.audio_root / session.id).resolve()
            if session_directory.parent != self.audio_root:
                raise ValueError("Güvenli olmayan oturum dizini reddedildi.")
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
        target = session.audio_directory.resolve()
        if target.parent != self.audio_root or target.name != session.id:
            raise ValueError("Güvenli olmayan oturum silme isteği reddedildi.")
        shutil.rmtree(target, ignore_errors=True)


def prompt_messages(
    session: ConversationSession, *, recent_limit: int = 14
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if session.summary:
        messages.append(
            {
                "role": "system",
                "content": f"Önceki konuşmadan yerel hafıza notu:\n{session.summary}",
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
