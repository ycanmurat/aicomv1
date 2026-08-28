from __future__ import annotations

import argparse
import asyncio
import json
import logging
import wave
from contextlib import asynccontextmanager
from dataclasses import dataclass
from threading import Event
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from aicomv1.config import PROJECT_ROOT, Settings
from aicomv1.engine import AssistantEngine, EventSink
from aicomv1.knowledge import KnowledgeStore
from aicomv1.memory import SessionNotFoundError, SessionStore
from aicomv1.prompt import normalize_language
from aicomv1.providers.llm_ollama import OllamaChatProvider
from aicomv1.providers.stt_auto import AutoTranscriber
from aicomv1.providers.tts_auto import AutoSynthesizer
from aicomv1.tools import LocalToolRouter

LOGGER = logging.getLogger("aicomv1")
WEB_ROOT = PROJECT_ROOT / "web"
MAX_AUDIO_BYTES = 16000 * 2 * 90
MIN_AUDIO_BYTES = 16000 * 2 // 5


class SessionInput(BaseModel):
    language: str = "en"

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        return normalize_language(value)


class KnowledgeInput(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    body: str = Field(min_length=1, max_length=100_000)
    source: str = Field(default="local", min_length=1, max_length=300)


@dataclass(slots=True)
class AppServices:
    settings: Settings
    sessions: SessionStore
    knowledge: KnowledgeStore
    llm: OllamaChatProvider
    stt: AutoTranscriber
    tts: AutoSynthesizer
    engine: AssistantEngine


def build_services(settings: Settings) -> AppServices:
    knowledge = KnowledgeStore(settings.knowledge_db)
    llm = OllamaChatProvider(settings)
    stt = AutoTranscriber(settings)
    tts = AutoSynthesizer(settings)
    engine = AssistantEngine(llm, tts, LocalToolRouter(knowledge))
    return AppServices(
        settings=settings,
        sessions=SessionStore(settings.audio_root),
        knowledge=knowledge,
        llm=llm,
        stt=stt,
        tts=tts,
        engine=engine,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    configured = settings or Settings.from_env()
    services = build_services(configured)

    async def warm_services() -> None:
        try:
            llm_status = await services.llm.status()
            if llm_status.ready:
                await services.llm.complete(
                    system_prompt="Local model warmup check.",
                    messages=[{"role": "user", "content": "Say ready."}],
                    max_tokens=2,
                )
            tts_status = await asyncio.to_thread(services.tts.status)
            if tts_status.ready and tts_status.name != "tts-none":
                warm_path = configured.audio_root / ".warmup.wav"
                await asyncio.to_thread(services.tts.synthesize, "Hazırım.", warm_path)
                warm_path.unlink(missing_ok=True)
        except Exception:
            LOGGER.warning("Background model warmup did not complete", exc_info=True)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        warm_task = asyncio.create_task(warm_services()) if configured.warmup else None
        yield
        if warm_task is not None and not warm_task.done():
            warm_task.cancel()
            await asyncio.gather(warm_task, return_exceptions=True)

    app = FastAPI(title="AICOM v1", version="0.1.0", lifespan=lifespan)
    app.state.services = services
    app.mount("/assets", StaticFiles(directory=WEB_ROOT), name="assets")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(WEB_ROOT / "index.html")

    @app.get("/api/health")
    async def health(language: str = "en") -> dict[str, object]:
        try:
            language = normalize_language(language)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        llm_status = await services.llm.status()
        stt_status, tts_status = await asyncio.gather(
            asyncio.to_thread(services.stt.status),
            asyncio.to_thread(services.tts.status, language=language),
        )
        components = [llm_status, stt_status, tts_status]
        return {
            "ready": all(item.ready for item in components),
            "components": [
                {"name": item.name, "ready": item.ready, "detail": item.detail}
                for item in components
            ],
            "knowledge_documents": services.knowledge.count(),
            "language": language,
            "languages": ["en", "tr"],
            "privacy": "Local speech processing; Ollama uses a configurable, loopback-default URL.",
        }

    @app.post("/api/sessions")
    async def create_session(item: SessionInput | None = None) -> dict[str, str]:
        session = services.sessions.create(language=item.language if item else "en")
        return {"id": session.id, "language": session.language}

    @app.delete("/api/sessions/{session_id}", status_code=204)
    async def delete_session(session_id: str) -> None:
        services.sessions.delete(session_id)

    @app.get("/api/audio/{session_id}/{filename}")
    async def audio_file(session_id: str, filename: str) -> FileResponse:
        try:
            session = services.sessions.get(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Session was not found.") from exc
        if session.audio_directory is None:
            raise HTTPException(status_code=404, detail="Audio was not found.")
        target = (session.audio_directory / filename).resolve()
        if target.parent != session.audio_directory.resolve() or not target.is_file():
            raise HTTPException(status_code=404, detail="Audio was not found.")
        return FileResponse(target, media_type="audio/wav", headers={"Cache-Control": "no-store"})

    @app.post("/api/knowledge")
    async def add_knowledge(item: KnowledgeInput) -> dict[str, int]:
        document_id = services.knowledge.add(title=item.title, body=item.body, source=item.source)
        return {"id": document_id}

    @app.get("/api/knowledge")
    async def knowledge_status() -> dict[str, int]:
        return {"count": services.knowledge.count()}

    @app.websocket("/api/realtime/{session_id}")
    async def realtime(websocket: WebSocket, session_id: str) -> None:
        try:
            session = services.sessions.get(session_id)
        except SessionNotFoundError:
            await websocket.accept()
            await websocket.close(code=4404, reason="Session was not found")
            return

        await websocket.accept()
        send_lock = asyncio.Lock()
        connected = True
        audio_buffer = bytearray()
        accepting_audio = False
        language = session.language
        audio_language = language
        active_task: asyncio.Task[None] | None = None
        active_cancel: Event | None = None
        pending_tasks: set[asyncio.Task[None]] = set()

        async def emit(event: dict[str, object], *, cancel: Event | None = None) -> None:
            if not connected or (cancel is not None and cancel.is_set()):
                return
            event = dict(event)
            if event.get("type") == "audio":
                event["url"] = (
                    f"/api/audio/{session_id}/{event.pop('filename')}?v={uuid4().hex[:8]}"
                )
            async with send_lock:
                if not connected or (cancel is not None and cancel.is_set()):
                    return
                await websocket.send_text(json.dumps(event, ensure_ascii=False))

        def turn_emitter(turn_id: str, turn_language: str, cancel: Event) -> EventSink:
            async def emit_turn(event: dict[str, object]) -> None:
                await emit({**event, "turn_id": turn_id, "language": turn_language}, cancel=cancel)

            return emit_turn

        def interrupt_active() -> None:
            nonlocal active_task, active_cancel
            if active_cancel is not None:
                active_cancel.set()
            if active_task is not None and not active_task.done():
                active_task.cancel()
            active_task = None
            active_cancel = None

        def launch_text(text: str, *, turn_language: str) -> None:
            nonlocal active_task, active_cancel
            interrupt_active()
            current_turn = uuid4().hex
            active_cancel = Event()
            session.active_cancel = active_cancel
            cancel = active_cancel
            emit_turn = turn_emitter(current_turn, turn_language, cancel)

            async def run() -> None:
                try:
                    await emit_turn({"type": "transcript", "text": text})
                    await services.engine.respond(
                        session=session,
                        user_text=text,
                        turn_id=current_turn,
                        cancel=cancel,
                        emit=emit_turn,
                        language=turn_language,
                    )
                except asyncio.CancelledError:
                    return
                except Exception as exc:
                    LOGGER.exception("Text response turn failed")
                    await emit_turn(
                        {"type": "error", "code": "response_failed", "message": str(exc)}
                    )
                    await emit_turn({"type": "state", "state": "listening"})

            active_task = asyncio.create_task(run())
            pending_tasks.add(active_task)
            active_task.add_done_callback(pending_tasks.discard)

        def launch_audio(raw_audio: bytes, *, turn_language: str) -> None:
            nonlocal active_task, active_cancel
            interrupt_active()
            turn_id = uuid4().hex
            active_cancel = Event()
            session.active_cancel = active_cancel
            cancel = active_cancel
            emit_turn = turn_emitter(turn_id, turn_language, cancel)

            async def run() -> None:
                if session.audio_directory is None:
                    return
                input_path = session.audio_directory / f"input-{turn_id}.wav"
                try:
                    with wave.open(str(input_path), "wb") as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(16_000)
                        wav.writeframes(raw_audio)
                    await emit_turn({"type": "state", "state": "transcribing"})
                    transcription = await asyncio.to_thread(
                        services.stt.transcribe, input_path, turn_language
                    )
                    if cancel.is_set():
                        return
                    if not transcription.text:
                        await emit_turn(
                            {
                                "type": "warning",
                                "turn_id": turn_id,
                                "code": "speech_not_understood",
                                "message": "Speech was not understood; please try again.",
                            }
                        )
                        await emit_turn({"type": "state", "state": "listening"})
                        return
                    await emit_turn(
                        {
                            "type": "transcript",
                            "turn_id": turn_id,
                            "text": transcription.text,
                            "provider": transcription.provider,
                            "transcription_ms": transcription.elapsed_ms,
                            "language": turn_language,
                        }
                    )
                    await services.engine.respond(
                        session=session,
                        user_text=transcription.text,
                        turn_id=turn_id,
                        cancel=cancel,
                        emit=emit_turn,
                        language=turn_language,
                    )
                except asyncio.CancelledError:
                    return
                except Exception as exc:
                    LOGGER.exception("Voice response turn failed")
                    await emit_turn(
                        {"type": "error", "code": "response_failed", "message": str(exc)}
                    )
                    await emit_turn({"type": "state", "state": "listening"})

            active_task = asyncio.create_task(run())
            pending_tasks.add(active_task)
            active_task.add_done_callback(pending_tasks.discard)

        await emit({"type": "language", "language": language})
        await emit({"type": "state", "state": "listening"})
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if data := message.get("bytes"):
                    if accepting_audio and len(audio_buffer) + len(data) <= MAX_AUDIO_BYTES:
                        audio_buffer.extend(data)
                    continue
                raw_text = message.get("text")
                if not raw_text:
                    continue
                try:
                    event = json.loads(raw_text)
                except json.JSONDecodeError:
                    await emit(
                        {"type": "error", "code": "invalid_message", "message": "Invalid JSON."}
                    )
                    continue
                if not isinstance(event, dict):
                    await emit(
                        {
                            "type": "error",
                            "code": "invalid_message",
                            "message": "Client message must be an object.",
                        }
                    )
                    continue
                event_type = event.get("type")
                requested_language = language
                if "language" in event:
                    try:
                        requested_language = normalize_language(str(event["language"]))
                    except ValueError as exc:
                        await emit(
                            {"type": "error", "code": "unsupported_language", "message": str(exc)}
                        )
                        continue
                if event_type == "interrupt":
                    interrupt_active()
                    accepting_audio = False
                    audio_buffer.clear()
                    await emit({"type": "state", "state": "listening"})
                elif event_type == "audio.start":
                    interrupt_active()
                    language = session.language = requested_language
                    audio_buffer.clear()
                    accepting_audio = True
                    audio_language = language
                elif event_type == "audio.commit":
                    if not accepting_audio:
                        continue
                    accepting_audio = False
                    if len(audio_buffer) >= MIN_AUDIO_BYTES:
                        launch_audio(bytes(audio_buffer), turn_language=audio_language)
                    else:
                        await emit({"type": "state", "state": "listening"})
                    audio_buffer.clear()
                elif event_type == "text":
                    if not isinstance(event.get("text"), str):
                        await emit(
                            {
                                "type": "error",
                                "code": "invalid_message",
                                "message": "Text must be a string.",
                            }
                        )
                        continue
                    text = str(event.get("text", "")).strip()[:8000]
                    if text:
                        language = session.language = requested_language
                        accepting_audio = False
                        audio_buffer.clear()
                        launch_text(text, turn_language=language)
                elif event_type == "language.set":
                    interrupt_active()
                    language = session.language = requested_language
                    accepting_audio = False
                    audio_buffer.clear()
                    await emit({"type": "language", "language": language})
                    await emit({"type": "state", "state": "listening"})
                elif event_type == "ping":
                    await emit({"type": "pong"})
        except WebSocketDisconnect:
            pass
        finally:
            connected = False
            interrupt_active()
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="AICOM local bilingual voice application")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(
        "aicomv1.api:create_app",
        factory=True,
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
