from pathlib import Path

import pytest

from aicomv1.memory import SessionNotFoundError, SessionStore, prompt_messages
from aicomv1.models import Role


def test_session_lifecycle_deletes_only_own_directory(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "audio")
    session = store.create()
    assert session.audio_directory is not None
    marker = session.audio_directory / "sample.wav"
    marker.write_bytes(b"RIFF")
    session.add(Role.USER, "  Merhaba   dünya  ")
    assert session.messages[0].content == "Merhaba dünya"
    store.delete(session.id)
    assert not marker.exists()
    with pytest.raises(SessionNotFoundError):
        store.get(session.id)


@pytest.mark.parametrize(("language", "label"), [("en", "Local memory"), ("tr", "yerel hafıza")])
def test_memory_wrapper_uses_session_language(tmp_path: Path, language: str, label: str) -> None:
    session = SessionStore(tmp_path / "audio").create(language=language)
    session.summary = "A remembered preference."
    session.add(Role.USER, "Hello")
    messages = prompt_messages(session)
    assert label in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "Hello"}
