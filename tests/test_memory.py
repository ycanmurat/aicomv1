from pathlib import Path

import pytest

from aicomv1.memory import SessionNotFoundError, SessionStore
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
