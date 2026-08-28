from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from aicomv1.api import create_app


def test_session_knowledge_and_index(settings) -> None:
    app = create_app(settings)
    with TestClient(app) as client:
        index = client.get("/")
        assert index.status_code == 200
        assert "AICOM" in index.text
        created = client.post("/api/sessions")
        assert created.status_code == 200
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
            assert websocket.receive_json()["state"] == "listening"
            websocket.send_json({"type": "ping"})
            assert websocket.receive_json()["type"] == "pong"
