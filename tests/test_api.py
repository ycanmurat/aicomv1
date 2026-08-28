from __future__ import annotations

import asyncio
import wave
from pathlib import Path
from threading import Event

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from aicomv1.api import create_app
from aicomv1.models import ComponentStatus, Transcription


class RecordingEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def respond(self, *, user_text, language, emit, **kwargs) -> None:
        self.calls.append((user_text, language))
        await emit({"type": "text_done", "text": f"Reply in {language}"})


class RecordingSTT:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def transcribe(self, path: Path, language: str) -> Transcription:
        with wave.open(str(path)) as wav:
            assert wav.getframerate() == 16_000
            self.calls.append((language, wav.getnframes()))
        return Transcription("Hello" if language == "en" else "Merhaba", 1, "fake-stt")


def test_session_knowledge_and_index(settings) -> None:
    app = create_app(settings)
    with TestClient(app) as client:
        index = client.get("/")
        assert index.status_code == 200
        assert "AICOM" in index.text
        created = client.post("/api/sessions")
        assert created.status_code == 200
        assert created.json()["language"] == "en"
        session_id = created.json()["id"]
        added = client.post(
            "/api/knowledge",
            json={"title": "Test", "body": "Yerel bilgi", "source": "pytest"},
        )
        assert added.json() == {"id": 1}
        assert client.get("/api/knowledge").json() == {"count": 1}
        assert client.delete(f"/api/sessions/{session_id}").status_code == 204


def test_websocket_rejects_unknown_session(settings) -> None:
    app = create_app(settings)
    with (
        TestClient(app) as client,
        client.websocket_connect("/api/realtime/unknown") as websocket,
        pytest.raises(WebSocketDisconnect) as caught,
    ):
        websocket.receive_json()
    assert caught.value.code == 4404


def test_websocket_ping(settings) -> None:
    app = create_app(settings)
    with TestClient(app) as client:
        session_id = client.post("/api/sessions").json()["id"]
        with client.websocket_connect(f"/api/realtime/{session_id}") as websocket:
            assert websocket.receive_json() == {"type": "language", "language": "en"}
            assert websocket.receive_json()["state"] == "listening"
            websocket.send_json({"type": "ping"})
            assert websocket.receive_json()["type"] == "pong"


@pytest.mark.parametrize(("language", "normalized"), [("en", "en"), ("tr-TR", "tr")])
def test_session_language_validation_and_reconnect(settings, language, normalized) -> None:
    app = create_app(settings)
    with TestClient(app) as client:
        created = client.post("/api/sessions", json={"language": language})
        session_id = created.json()["id"]
        assert created.json()["language"] == normalized
        assert client.post("/api/sessions", json={"language": "de"}).status_code == 422
        for _ in range(2):
            with client.websocket_connect(f"/api/realtime/{session_id}") as websocket:
                assert websocket.receive_json() == {"type": "language", "language": normalized}
                assert websocket.receive_json()["state"] == "listening"
                websocket.send_json({"type": "language.set", "language": "tr"})
                assert websocket.receive_json() == {"type": "language", "language": "tr"}
                assert websocket.receive_json()["state"] == "listening"
            normalized = "tr"


@pytest.mark.parametrize("language", ["en", "tr"])
def test_text_turn_uses_session_language(settings, language) -> None:
    app = create_app(settings)
    engine = RecordingEngine()
    app.state.services.engine = engine
    with TestClient(app) as client:
        session_id = client.post("/api/sessions", json={"language": language}).json()["id"]
        with client.websocket_connect(f"/api/realtime/{session_id}") as websocket:
            websocket.receive_json()
            websocket.receive_json()
            websocket.send_json({"type": "text", "text": "Hello"})
            transcript = websocket.receive_json()
            assert transcript["type"] == "transcript"
            assert transcript["language"] == language
            response = websocket.receive_json()
            assert response["type"] == "text_done"
            assert response["language"] == language
            assert response["turn_id"] == transcript["turn_id"]
    assert engine.calls == [("Hello", language)]


@pytest.mark.parametrize("language", ["en", "tr"])
def test_audio_turn_captures_language_at_start(settings, language) -> None:
    app = create_app(settings)
    stt, engine = RecordingSTT(), RecordingEngine()
    app.state.services.stt = stt
    app.state.services.engine = engine
    with TestClient(app) as client:
        session_id = client.post("/api/sessions").json()["id"]
        with client.websocket_connect(f"/api/realtime/{session_id}") as websocket:
            websocket.receive_json()
            websocket.receive_json()
            websocket.send_json({"type": "audio.start", "language": language})
            websocket.send_bytes(b"\0\0" * 4000)
            other = "tr" if language == "en" else "en"
            websocket.send_json({"type": "audio.commit", "language": other})
            assert websocket.receive_json()["state"] == "transcribing"
            transcript = websocket.receive_json()
            assert transcript["type"] == "transcript"
            assert transcript["language"] == language
            assert websocket.receive_json()["type"] == "text_done"
    assert stt.calls == [(language, 4000)]
    assert engine.calls == [("Hello" if language == "en" else "Merhaba", language)]


def test_language_switch_clears_partial_recording(settings) -> None:
    app = create_app(settings)
    stt = RecordingSTT()
    app.state.services.stt = stt
    with TestClient(app) as client:
        session_id = client.post("/api/sessions").json()["id"]
        with client.websocket_connect(f"/api/realtime/{session_id}") as websocket:
            websocket.receive_json()
            websocket.receive_json()
            websocket.send_json({"type": "audio.start", "language": "en"})
            websocket.send_bytes(b"\0\0" * 4000)
            websocket.send_json({"type": "language.set", "language": "tr"})
            assert websocket.receive_json() == {"type": "language", "language": "tr"}
            assert websocket.receive_json()["state"] == "listening"
            websocket.send_json({"type": "audio.commit"})
            websocket.send_json({"type": "ping"})
            assert websocket.receive_json()["type"] == "pong"
    assert stt.calls == []


def test_language_switch_suppresses_late_events_from_cancelled_turn(settings) -> None:
    cancelled = Event()

    class LateEngine:
        async def respond(self, *, emit, **kwargs) -> None:
            await emit({"type": "state", "state": "thinking"})
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await emit({"type": "text_delta", "delta": "Stale response"})
                await emit({"type": "audio", "filename": "stale.wav"})
                cancelled.set()

    app = create_app(settings)
    app.state.services.engine = LateEngine()
    with TestClient(app) as client:
        session_id = client.post("/api/sessions").json()["id"]
        with client.websocket_connect(f"/api/realtime/{session_id}") as websocket:
            websocket.receive_json()
            websocket.receive_json()
            websocket.send_json({"type": "text", "text": "Hello"})
            assert websocket.receive_json()["type"] == "transcript"
            assert websocket.receive_json()["state"] == "thinking"
            websocket.send_json({"type": "language.set", "language": "tr"})
            assert websocket.receive_json() == {"type": "language", "language": "tr"}
            assert websocket.receive_json()["state"] == "listening"
            assert cancelled.wait(timeout=2)
            websocket.send_json({"type": "ping"})
            assert websocket.receive_json()["type"] == "pong"


@pytest.mark.parametrize(
    ("event", "code"),
    [
        ([], "invalid_message"),
        ({"type": "text", "text": {}}, "invalid_message"),
        ({"type": "language.set", "language": "de"}, "unsupported_language"),
    ],
)
def test_websocket_rejects_invalid_messages_without_changing_language(
    settings, event, code
) -> None:
    app = create_app(settings)
    with TestClient(app) as client:
        session_id = client.post("/api/sessions").json()["id"]
        with client.websocket_connect(f"/api/realtime/{session_id}") as websocket:
            websocket.receive_json()
            websocket.receive_json()
            websocket.send_json(event)
            error = websocket.receive_json()
            assert error["type"] == "error"
            assert error["code"] == code
            assert app.state.services.sessions.get(session_id).language == "en"
            websocket.send_json({"type": "ping"})
            assert websocket.receive_json()["type"] == "pong"


@pytest.mark.parametrize("language", ["en", "tr"])
def test_health_reports_selected_language_provider(settings, monkeypatch, language) -> None:
    app = create_app(settings)

    async def llm_status():
        return ComponentStatus("fake-llm", True, "Ready")

    monkeypatch.setattr(app.state.services.llm, "status", llm_status)
    monkeypatch.setattr(
        app.state.services.stt, "status", lambda: ComponentStatus("fake-stt", True, "Ready")
    )
    monkeypatch.setattr(
        app.state.services.tts,
        "status",
        lambda language: ComponentStatus(f"fake-tts-{language}", True, "Ready"),
    )
    with TestClient(app) as client:
        result = client.get(f"/api/health?language={language}").json()
        assert result["language"] == language
        assert result["ready"] is True
        assert result["components"][-1]["name"] == f"fake-tts-{language}"
        assert client.get("/api/health?language=de").status_code == 422
