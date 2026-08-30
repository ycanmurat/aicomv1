"""Loopback-only voice lab: one TTS runtime, one request, no automatic fallback."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import queue
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

import httpx
import psutil
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StrictBool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from experiments.tts_lab.moss_engine import (
    FILLER_TEXT,
    GenerationCancelled,
    MossEngine,
    check_cancelled,
    peak_rss_mb,
    prepare_text,
)
from experiments.tts_lab.research_answer import ResearchAssistant
from experiments.tts_lab.speech_input import LocalTranscriber


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=600)
    mode: Literal["tts", "ask"] = "tts"
    use_web: StrictBool = False
    seed: int = Field(default=42, ge=0, le=2147483647)


def ready_sentence(buffer: str) -> tuple[str, str]:
    """Wait for a completed sentence, not arbitrary token or character chunks."""
    for match in re.finditer(r"[.!?](?=\s)", buffer):
        candidate = buffer[: match.end()].strip()
        last_word = candidate.split()[-1].lower()
        if re.fullmatch(r"\d+\.", last_word):
            continue
        if last_word in {"dr.", "prof.", "sn.", "av.", "örn.", "vb.", "vs."}:
            continue
        if candidate:
            return candidate, buffer[match.end() :].lstrip()
    return "", buffer


def create_app(
    project_root: Path,
    *,
    reference: Path | None = None,
    cpu_threads: int = 4,
    llm_model: str = "qwen3.5:2b-q4_K_M",
    engine=None,
    assistant=None,
) -> FastAPI:
    lab_root = project_root / ".runtime" / "voice-lab"
    lab_root.mkdir(parents=True, exist_ok=True)
    if reference is None:
        existing = project_root.parent / "aicall/data/audio/voice-reference/ada.wav"
        reference = existing if existing.is_file() else None
    elif not reference.is_file():
        raise ValueError(f"Referans ses bulunamadı: {reference}")
    engine = engine or MossEngine(lab_root, reference, cpu_threads)
    assistant = assistant or ResearchAssistant(model=llm_model)
    transcriber = LocalTranscriber(project_root, lab_root)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fatma-lab")
    busy = threading.Lock()
    state = {"ready": False, "error": None, "run_id": None, "stop": None}

    def initialize():
        try:
            engine.warmup()
            state["ready"] = True
        except Exception as exc:
            state["error"] = str(exc)

    async def owned_operation(operation, stop, request):
        """The worker, not the HTTP coroutine, owns the lock until actual exit."""

        def owned():
            try:
                return operation()
            finally:
                state["stop"] = None
                busy.release()

        future = asyncio.wrap_future(executor.submit(owned))
        try:
            while not future.done():
                if await request.is_disconnected():
                    stop.set()
                await asyncio.wait({future}, timeout=0.1)
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            stop.set()
            # Consume a later worker error even when the HTTP request vanished.
            future.add_done_callback(
                lambda done: done.exception() if not done.cancelled() else None
            )
            raise

    @asynccontextmanager
    async def lifespan(_app):
        await asyncio.get_running_loop().run_in_executor(executor, initialize)
        yield
        if state["stop"] is not None:
            state["stop"].set()
        executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(
        title="Fatma — Ses Laboratuvarı", lifespan=lifespan, docs_url=None, redoc_url=None
    )
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"]
    )

    @app.middleware("http")
    async def local_requests_only(request, call_next):
        if request.method == "POST":
            origin = request.headers.get("origin")
            if request.headers.get("x-fatma-lab") != "1" or (
                origin and urlsplit(origin).netloc != request.headers.get("host")
            ):
                return JSONResponse({"detail": "Bu istek yerel deney ekranından gelmelidir."}, 403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).parent / "web/index.html")

    @app.get("/api/health")
    async def health():
        available_models = []
        ollama_memory_mb = 0.0
        try:
            async with httpx.AsyncClient(timeout=1, trust_env=False) as client:
                tags, running = await asyncio.gather(
                    client.get("http://127.0.0.1:11434/api/tags"),
                    client.get("http://127.0.0.1:11434/api/ps"),
                )
                available_models = [item["name"] for item in tags.json().get("models", [])]
                ollama_memory_mb = (
                    sum(item.get("size", 0) for item in running.json().get("models", [])) / 1024**2
                )
        except (httpx.HTTPError, ValueError, KeyError):
            pass
        return {
            "ready": state["ready"],
            "error": state["error"],
            "busy": busy.locked(),
            "tts_model": "MOSS-TTS-Nano · ONNX",
            "cpu_threads": cpu_threads,
            "reference": engine.reference_name,
            "has_reference": engine.reference is not None,
            "llm_model": llm_model,
            "llm_available": llm_model in available_models,
            "microphone_available": transcriber.available,
            "process_rss_mb": round(psutil.Process().memory_info().rss / 1024**2, 1),
            "process_peak_rss_mb": peak_rss_mb(),
            "ollama_loaded_model_mb": round(ollama_memory_mb, 1),
        }

    @app.get("/api/reference")
    def reference_audio():
        if engine.reference is None:
            raise HTTPException(404, "Henüz özel referans sesi seçilmedi.")
        return FileResponse(engine.reference, media_type="audio/wav")

    @app.post("/api/reference")
    async def upload_reference(request: Request, file: Annotated[UploadFile, File()]):
        if not busy.acquire(blocking=False):
            raise HTTPException(409, "Önce devam eden deneyi durdurun.")
        stop = threading.Event()
        state["stop"] = stop
        submitted = False
        try:
            data = await file.read(4 * 1024 * 1024 + 1)
            if len(data) > 4 * 1024 * 1024:
                raise ValueError("Referans sesi en fazla dört megabayt olabilir.")
            info = sf.info(io.BytesIO(data))
            if info.format != "WAV" or not 2 <= info.duration <= 15 or info.channels not in (1, 2):
                raise ValueError("İki ila on beş saniyelik, tek veya çift kanallı bir WAV seçin.")
            reference_dir = lab_root / "references"
            reference_dir.mkdir(exist_ok=True)
            path = reference_dir / f"reference-{hashlib.sha256(data).hexdigest()[:16]}.wav"
            if not path.exists():
                path.write_bytes(data)
            previous = (
                engine.reference,
                engine.prompt_codes,
                engine.reference_key,
                list(engine.filler),
            )

            def replace_reference():
                try:
                    engine.set_reference(path)
                    engine.warmup(stop)
                except Exception:
                    engine.reference, engine.prompt_codes, engine.reference_key, engine.filler = (
                        previous
                    )
                    raise

            submitted = True
            await owned_operation(replace_reference, stop, request)
            return {"reference": engine.reference_name}
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            await file.close()
            if not submitted:
                state["stop"] = None
                busy.release()

    @app.post("/api/transcribe")
    async def transcribe(request: Request, file: Annotated[UploadFile, File()]):
        if not busy.acquire(blocking=False):
            raise HTTPException(409, "Önce devam eden deneyi durdurun.")
        stop = threading.Event()
        state["stop"] = stop
        submitted = False
        try:
            data = await file.read(5 * 1024 * 1024 + 1)
            started = time.perf_counter()
            submitted = True
            text = await owned_operation(lambda: transcriber.transcribe(data, stop), stop, request)
            return {"text": text, "elapsed_ms": round((time.perf_counter() - started) * 1000)}
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            await file.close()
            if not submitted:
                state["stop"] = None
                busy.release()

    def run_trial(payload, events, stop, run_id):
        started = time.perf_counter()
        first_answer_audio = None
        audio_seconds = 0.0
        inference_seconds = 0.0
        pcm_chunks = 0
        metrics = []

        def emit(event):
            while not stop.is_set():
                try:
                    events.put(event, timeout=0.1)
                    return
                except queue.Full:
                    continue
            raise GenerationCancelled

        def audio(pcm, meta, role="answer"):
            nonlocal first_answer_audio
            if role == "answer" and first_answer_audio is None:
                first_answer_audio = (time.perf_counter() - started) * 1000
            emit(
                {
                    "type": "audio",
                    "role": role,
                    "pcm": base64.b64encode(pcm.tobytes()).decode("ascii"),
                    "sample_rate": meta["sample_rate"],
                    "channels": pcm.shape[1],
                }
            )

        def speak(text):
            nonlocal audio_seconds, inference_seconds, pcm_chunks
            emit({"type": "speak", "text": text})
            result = engine.synthesize(text, audio, stop, seed=payload.seed)
            audio_seconds += result["audio_seconds"]
            inference_seconds += result["generation_seconds"]
            pcm_chunks += result["pcm_chunks"]
            metrics.append(result)

        research_thread = None
        try:
            emit({"type": "start", "run_id": run_id})
            if payload.mode == "tts":
                speak(payload.text)
            else:
                research_events = queue.Queue(maxsize=64)

                def research():
                    try:
                        for event in assistant.stream(
                            payload.text, use_web=payload.use_web, stop_event=stop
                        ):
                            while not stop.is_set():
                                try:
                                    research_events.put(event, timeout=0.1)
                                    break
                                except queue.Full:
                                    continue
                    except Exception as exc:
                        if not stop.is_set():
                            research_events.put(
                                {"type": "answer_done", "error": str(exc)}, timeout=1
                            )

                research_thread = threading.Thread(
                    target=research, daemon=True, name="fatma-answer"
                )
                research_thread.start()
                buffer = ""
                filler_sent = False
                while True:
                    check_cancelled(stop)
                    if (
                        not filler_sent
                        and first_answer_audio is None
                        and time.perf_counter() - started > 0.7
                    ):
                        filler_sent = True
                        emit({"type": "status", "message": FILLER_TEXT})
                        for pcm in engine.filler:
                            audio(pcm, {"sample_rate": engine.sample_rate}, role="filler")
                    try:
                        event = research_events.get(timeout=0.1)
                    except queue.Empty:
                        if not research_thread.is_alive():
                            raise RuntimeError("Yanıt akışı tamamlanamadı.") from None
                        continue
                    emit(event)
                    if event["type"] == "text_delta":
                        buffer += event["text"]
                        while True:
                            sentence, buffer = ready_sentence(buffer)
                            if not sentence:
                                break
                            speak(sentence)
                    elif event["type"] == "answer_done":
                        if event.get("cancelled"):
                            raise GenerationCancelled
                        if buffer.strip():
                            speak(buffer.strip())
                        if event.get("error"):
                            emit(
                                {
                                    "type": "warning",
                                    "message": (
                                        "Yanıt doğrulanamadı veya tamamlanamadı; "
                                        "durum bilgisini kontrol edin."
                                    ),
                                }
                            )
                        break
            emit(
                {
                    "type": "done",
                    "elapsed_ms": round((time.perf_counter() - started) * 1000),
                    "first_answer_pcm_ms": round(first_answer_audio, 1)
                    if first_answer_audio is not None
                    else None,
                    "audio_seconds": round(audio_seconds, 3),
                    "pcm_chunks": pcm_chunks,
                    "rtf": round(inference_seconds / audio_seconds, 3) if audio_seconds else None,
                    "process_peak_rss_mb": peak_rss_mb(),
                    "utterances": metrics,
                    "frame_limit_hit": any(m["frame_limit_hit"] for m in metrics),
                }
            )
        except GenerationCancelled:
            pass
        except Exception as exc:
            if not stop.is_set():
                emit({"type": "error", "message": str(exc)})
        finally:
            stop.set()
            # Keep ownership until the research request really exits, so Stop
            # cannot create several overlapping model requests.
            if research_thread is not None:
                research_thread.join()
            state["stop"] = None
            state["run_id"] = None
            busy.release()

    @app.post("/api/stream")
    async def stream(payload: SpeechRequest, request: Request):
        payload.text = prepare_text(payload.text)
        if not payload.text or (payload.mode == "tts" and payload.use_web):
            raise HTTPException(400, "Geçerli bir metin ve deney modu seçin.")
        if not state["ready"]:
            raise HTTPException(503, state["error"] or "Ses modeli henüz hazır değil.")
        if not busy.acquire(blocking=False):
            raise HTTPException(409, "Önceki işlem kapanıyor veya başka bir deney çalışıyor.")
        events = queue.Queue(maxsize=64)
        stop = threading.Event()
        run_id = uuid.uuid4().hex
        state.update(stop=stop, run_id=run_id)
        future = executor.submit(run_trial, payload, events, stop, run_id)

        async def response():
            try:
                while not future.done() or not events.empty():
                    if await request.is_disconnected():
                        break
                    try:
                        event = await asyncio.to_thread(events.get, True, 0.1)
                    except queue.Empty:
                        continue
                    yield json.dumps(event, ensure_ascii=False) + "\n"
            finally:
                stop.set()

        return StreamingResponse(response(), media_type="application/x-ndjson")

    @app.post("/api/cancel")
    def cancel():
        if state["stop"] is not None:
            state["stop"].set()
        return {"cancelled": True}

    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "web"), name="static")
    return app
