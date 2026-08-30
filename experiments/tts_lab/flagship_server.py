"""Local MOSS-TTS v1.5 flagship quality laboratory.

The FastAPI process owns one persistent CrispASR child. The model is loaded once,
kept warm between requests, and never resolved or downloaded during inference.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import uuid
import wave
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit

import httpx
import psutil
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

MODEL_NAME = "MOSS-TTS v1.5 · Qwen3-8B Q4_K"
LANGUAGE_CODE = "tr"
LANGUAGE_PROMPT = "Turkish"
CRISP_VERSION = "0.8.30"
MAX_TEXT_LENGTH = 600
MAX_SPEECH_TOKENS = 160  # 12.8 s at MOSS's shipped 12.5 Hz codec rate.
MAX_VOICE_BYTES = 25 * 1024 * 1024
VOICE_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\.wav\Z")

TEST_CASES = (
    {
        "id": "welcome",
        "label": "Karşılama",
        "text": "Merhaba, ben Fatma. Size nasıl yardımcı olabilirim?",
    },
    {
        "id": "empathy",
        "label": "Empati",
        "text": (
            "Sizi anlıyorum. Bu durumun can sıkıcı olduğunu biliyorum; "
            "birlikte uygun bir çözüm bulalım."
        ),
    },
    {
        "id": "numbers",
        "label": "Sayı ve tarih",
        "text": (
            "Randevunuz yirmi dokuz Ağustos günü saat on dört otuz için oluşturuldu. "
            "Tutar bin iki yüz kırk dokuz lira doksan kuruş."
        ),
    },
    {
        "id": "pronunciation",
        "label": "Telaffuz",
        "text": (
            "Değerlendirmenizi doğruladıktan sonra güncel bilgileri sizinle paylaşacağım. "
            "Çağrı kaydınız güvenli biçimde güncellendi."
        ),
    },
    {
        "id": "emotion",
        "label": "Duygu geçişi",
        "text": (
            "Harika, işleminiz tamamlandı! Beklediğiniz için teşekkür ederim; "
            "şimdi içiniz rahat olabilir."
        ),
    },
    {
        "id": "waiting",
        "label": "Bekleme cümlesi",
        "text": "Elbette, hemen kontrol ediyorum. Lütfen bir saniye bekleyin.",
    },
)


class FlagshipRequest(BaseModel):
    """Strict request boundary; omitted temperature preserves model defaults."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)
    voice: str = Field(default="preset", max_length=68)
    instruction: str = Field(default="", max_length=180)
    temperature: Annotated[
        float, Field(strict=True, ge=0.1, le=3.0, allow_inf_nan=False)
    ] | None = None
    top_p: Annotated[float, Field(strict=True, gt=0.0, le=1.0, allow_inf_nan=False)] = 0.8
    top_k: Annotated[int, Field(strict=True, ge=1, le=100)] = 25
    repetition_penalty: Annotated[
        float, Field(strict=True, ge=0.5, le=2.0, allow_inf_nan=False)
    ] = 1.0

    @field_validator("text", mode="before")
    @classmethod
    def strip_text(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("Metin bir karakter dizisi olmalıdır.")
        return value.strip()

    @field_validator("voice")
    @classmethod
    def validate_voice(cls, value: str) -> str:
        if value == "preset":
            return value
        return validate_voice_filename(value)

    @field_validator("instruction")
    @classmethod
    def strip_instruction(cls, value: str) -> str:
        return value.strip()


@dataclass(frozen=True)
class WavMetadata:
    sample_rate: int
    channels: int
    sample_width: int
    frame_count: int
    duration_seconds: float


@dataclass(frozen=True)
class HistoryEntry:
    generation_id: str
    text: str
    voice: str
    duration_seconds: float
    created_at: str | float
    voice_label: str = ""
    latency_seconds: float = 0.0
    rtf: float = 0.0
    rss_mb: float = 0.0
    audio_filename: str = ""
    instruction: str = ""
    temperature: float | None = None
    top_p: float = 0.8
    top_k: int = 25
    repetition_penalty: float = 1.0

    def public(self) -> dict[str, Any]:
        created_at = self.created_at
        if isinstance(created_at, (int, float)):
            created_at = datetime.fromtimestamp(created_at, UTC).isoformat()
        return {
            "id": self.generation_id,
            "text": self.text,
            "voice": self.voice,
            "voice_label": self.voice_label or self.voice,
            "created_at": created_at,
            "audio_url": f"/api/audio/{self.generation_id}.wav",
            "metrics": {
                "latency_seconds": round(self.latency_seconds, 3),
                "audio_seconds": round(self.duration_seconds, 3),
                "rtf": round(self.rtf, 3),
                "rss_mb": round(self.rss_mb, 1),
            },
            "settings": {
                "instruction": self.instruction,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "repetition_penalty": self.repetition_penalty,
            },
        }


class HistoryStore:
    def __init__(self, max_items: int = 40):
        if max_items <= 0:
            raise ValueError("History limit must be positive.")
        self.max_items = max_items
        self._items: deque[HistoryEntry] = deque(maxlen=max_items)
        self._lock = threading.Lock()

    def add(self, entry: HistoryEntry) -> None:
        with self._lock:
            self._items.appendleft(entry)

    def items(self) -> list[HistoryEntry]:
        with self._lock:
            return list(self._items)

    def find(self, generation_id: str) -> HistoryEntry | None:
        with self._lock:
            return next(
                (item for item in self._items if item.generation_id == generation_id), None
            )


def validate_voice_filename(name: str) -> str:
    if not isinstance(name, str) or not VOICE_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "Ses adı yalnızca harf, rakam, alt çizgi veya tire içeren bir WAV olmalıdır."
        )
    return name


def crisp_voice_name(filename: str) -> str:
    """Convert our safe WAV filename into CrispASR's voice-catalog identifier."""

    return Path(validate_voice_filename(filename)).stem


def read_wav_metadata(payload: bytes) -> WavMetadata:
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("Boş veya geçersiz WAV verisi.")
    try:
        with wave.open(io.BytesIO(payload), "rb") as source:
            if source.getcomptype() != "NONE":
                raise ValueError("Sıkıştırılmış WAV desteklenmiyor.")
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            sample_rate = source.getframerate()
            frame_count = source.getnframes()
    except (EOFError, wave.Error) as exc:
        raise ValueError("Geçerli bir PCM WAV dosyası değil.") from exc
    if channels <= 0 or sample_width <= 0 or sample_rate <= 0 or frame_count <= 0:
        raise ValueError("WAV üstbilgisi geçersiz.")
    return WavMetadata(
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        frame_count=frame_count,
        duration_seconds=frame_count / sample_rate,
    )


def parse_crisp_response(payload: bytes, content_type: str) -> WavMetadata:
    media_type = (content_type or "").split(";", 1)[0].strip().lower()
    if media_type not in {"audio/wav", "audio/x-wav", "audio/wave"}:
        raise ValueError("CrispASR ses yerine hata veya bilinmeyen veri döndürdü.")
    if not payload.startswith(b"RIFF") or payload[8:12] != b"WAVE":
        raise ValueError("CrispASR geçerli WAV üretmedi.")
    return read_wav_metadata(payload)


def process_memory_status(process_factory=psutil.Process) -> dict[str, float]:
    try:
        rss = process_factory().memory_info().rss / 1024**2
    except (psutil.Error, OSError):
        rss = 0.0
    return {"process_rss_mb": round(rss + 1e-6, 1)}


@dataclass(frozen=True)
class RuntimePaths:
    project_root: Path

    @property
    def root(self) -> Path:
        return self.project_root / ".runtime/voice-lab/moss-v15-flagship"

    @property
    def binary(self) -> Path:
        return self.root / "bin/crispasr"

    @property
    def model(self) -> Path:
        return self.root / "cache/moss-tts-v1.5-q4_k.gguf"

    @property
    def codec(self) -> Path:
        return self.root / "cache/moss-tts-v1.5-codec.gguf"

    @property
    def voices(self) -> Path:
        return self.root / "voices"

    @property
    def output(self) -> Path:
        return self.root / "output/lab"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def history_file(self) -> Path:
        return self.root / "history.json"

    @property
    def voices_file(self) -> Path:
        return self.root / "voices.json"


class CrispServer:
    """Own exactly one loopback CrispASR process and its HTTP client."""

    def __init__(self, paths: RuntimePaths, *, port: int = 18084, cpu_threads: int = 4):
        self.paths = paths
        self.port = port
        self.cpu_threads = cpu_threads
        self.process: subprocess.Popen | None = None
        self.log_handle = None
        self.client: httpx.Client | None = None
        self.ready = False
        self.error: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _validate_assets(self) -> None:
        for path in (self.paths.binary, self.paths.model, self.paths.codec):
            if not path.is_file():
                raise RuntimeError(
                    f"Eksik flagship bileşeni: {path.name}. "
                    "Önce setup_moss_v15_flagship.sh çalıştırılmalı."
                )

    def _port_is_free(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", self.port))
            except OSError:
                return False
        return True

    def start(self, *, warmup: bool = True) -> None:
        self._validate_assets()
        if not self._port_is_free():
            raise RuntimeError(
                f"İç model portu {self.port} başka bir süreç tarafından kullanılıyor; "
                "o süreci otomatik olarak durdurmayacağım."
            )
        for directory in (self.paths.voices, self.paths.output, self.paths.logs):
            directory.mkdir(parents=True, exist_ok=True)
        self.log_handle = (self.paths.logs / "crispasr.log").open("a", encoding="utf-8")
        command = [
            str(self.paths.binary),
            "--server",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--server-workers",
            "1",
            "--backend",
            "moss-tts",
            "-m",
            str(self.paths.model),
            "--codec-model",
            str(self.paths.codec),
            "--gpu-backend",
            "metal",
            "-t",
            str(self.cpu_threads),
            "-l",
            LANGUAGE_CODE,
            "--no-punctuation",
            "--tts-max-input-chars",
            str(MAX_TEXT_LENGTH),
            "--voice-dir",
            str(self.paths.voices),
        ]
        environment = os.environ.copy()
        environment["CRISPASR_CACHE_DIR"] = str(self.paths.root / "cache")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        self.client = httpx.Client(base_url=self.base_url, timeout=180, trust_env=False)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    "CrispASR model süreci başlatılamadı; log dosyasını kontrol edin."
                )
            try:
                response = self.client.get("/health", timeout=1)
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        else:
            raise RuntimeError("MOSS-TTS v1.5 doksan saniye içinde yüklenemedi.")

        if warmup:
            response = self.client.post(
                "/v1/audio/speech",
                json={
                    "input": "Merhaba.",
                    "language": LANGUAGE_CODE,
                    "response_format": "wav",
                    "speaker_identity": "synthetic",
                },
            )
            if response.status_code != 200:
                raise RuntimeError(
                    "Model yüklendi ancak gerçek Türkçe ısınma üretimi başarısız oldu."
                )
            parse_crisp_response(response.content, response.headers.get("content-type", ""))
        self.ready = True
        self.error = None

    def synthesize(self, payload: dict[str, Any]) -> tuple[bytes, str]:
        if not self.ready or not self.client or not self.process or self.process.poll() is not None:
            raise RuntimeError("MOSS-TTS süreci hazır değil.")
        response = self.client.post("/v1/audio/speech", json=payload)
        if response.status_code != 200:
            detail = ""
            try:
                body = response.json()
                detail = body.get("error") or body.get("detail") or body.get("message") or ""
            except (ValueError, AttributeError):
                pass
            raise RuntimeError(detail or f"CrispASR {response.status_code} hatası döndürdü.")
        return response.content, response.headers.get("content-type", "")

    def rss_mb(self) -> float:
        if not self.process or self.process.poll() is not None:
            return 0.0
        return process_memory_status(lambda: psutil.Process(self.process.pid))["process_rss_mb"]

    def shutdown(self) -> None:
        self.ready = False
        if self.client:
            self.client.close()
            self.client = None
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None
        if self.log_handle:
            self.log_handle.close()
            self.log_handle = None


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return fallback


def _history_from_disk(paths: RuntimePaths) -> HistoryStore:
    store = HistoryStore()
    valid_fields = {item.name for item in fields(HistoryEntry)}
    records = _load_json(paths.history_file, [])
    if not isinstance(records, list):
        return store
    for record in reversed(records[-store.max_items :]):
        try:
            clean = {key: value for key, value in record.items() if key in valid_fields}
            entry = HistoryEntry(**clean)
            if entry.audio_filename and (paths.output / entry.audio_filename).is_file():
                store.add(entry)
        except (TypeError, ValueError):
            continue
    return store


def _save_history(paths: RuntimePaths, history: HistoryStore) -> None:
    _atomic_json(paths.history_file, [asdict(item) for item in history.items()])


def _voice_catalog(paths: RuntimePaths) -> dict[str, str]:
    catalog = _load_json(paths.voices_file, {})
    if not isinstance(catalog, dict):
        catalog = {}
    return {
        name: str(label)
        for name, label in catalog.items()
        if isinstance(name, str)
        and VOICE_NAME_PATTERN.fullmatch(name)
        and (paths.voices / name).is_file()
    }


def _seed_reference(paths: RuntimePaths, catalog: dict[str, str]) -> None:
    source_value = os.environ.get("MOSS_FLAGSHIP_REFERENCE", "").strip()
    if not source_value:
        return
    source = Path(source_value).expanduser().resolve()
    if not source.is_file():
        return
    payload = source.read_bytes()
    try:
        metadata = read_wav_metadata(payload)
    except ValueError:
        return
    if not (2 <= metadata.duration_seconds <= 15) or metadata.sample_width != 2:
        return
    destination = paths.voices / "ada-sentetik.wav"
    if not destination.exists():
        shutil.copyfile(source, destination)
    catalog[destination.name] = "Ada · sentetik referans"
    _atomic_json(paths.voices_file, catalog)


def create_app(
    project_root: Path,
    *,
    internal_port: int = 18084,
    cpu_threads: int = 4,
    manager: CrispServer | None = None,
) -> FastAPI:
    project_root = project_root.resolve()
    paths = RuntimePaths(project_root)
    paths.voices.mkdir(parents=True, exist_ok=True)
    paths.output.mkdir(parents=True, exist_ok=True)
    history = _history_from_disk(paths)
    voices = _voice_catalog(paths)
    _seed_reference(paths, voices)
    manager = manager or CrispServer(paths, port=internal_port, cpu_threads=cpu_threads)
    generation_lock = asyncio.Lock()
    state: dict[str, Any] = {"startup_error": None}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            await asyncio.to_thread(manager.start)
        except Exception as exc:  # UI remains available with an actionable status.
            manager.error = str(exc)
            state["startup_error"] = str(exc)
        yield
        await asyncio.to_thread(manager.shutdown)

    app = FastAPI(
        title="Fatma · MOSS-TTS v1.5 Laboratuvarı",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.crisp = manager
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "testserver"],
    )

    @app.middleware("http")
    async def local_lab_requests(request: Request, call_next):
        if request.method == "POST":
            origin = request.headers.get("origin")
            same_origin = not origin or urlsplit(origin).netloc == request.headers.get("host")
            if request.headers.get("x-fatma-lab") != "flagship-v1" or not same_origin:
                return JSONResponse(
                    {"detail": "Bu istek yerel flagship deney ekranından gelmelidir."},
                    status_code=403,
                )
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    web_root = Path(__file__).parent / "flagship_web"
    app.mount("/static/flagship", StaticFiles(directory=web_root), name="flagship-static")

    @app.get("/")
    def index():
        return FileResponse(web_root / "index.html")

    def public_voices() -> list[dict[str, str]]:
        return [{"id": "preset", "label": "MOSS v1.5 varsayılan sesi"}] + [
            {"id": name, "label": label} for name, label in sorted(voices.items())
        ]

    @app.get("/api/status")
    def status():
        alive = bool(manager.process and manager.process.poll() is None)
        if manager.ready and not alive:
            manager.ready = False
            manager.error = "MOSS-TTS model süreci beklenmedik biçimde kapandı."
        return {
            "ready": manager.ready,
            "busy": generation_lock.locked(),
            "error": manager.error or state["startup_error"],
            "model": MODEL_NAME,
            "runtime": f"CrispASR v{CRISP_VERSION} · Metal",
            "language": f"{LANGUAGE_PROMPT} ({LANGUAGE_CODE})",
            "cpu_threads": cpu_threads,
            "backend_rss_mb": manager.rss_mb(),
            "voices": public_voices(),
        }

    @app.get("/api/cases")
    def cases():
        return {"cases": list(TEST_CASES)}

    @app.get("/api/history")
    def get_history():
        return {"history": [entry.public() for entry in history.items()]}

    @app.get("/api/audio/{generation_id}.wav")
    def audio(generation_id: str):
        if not re.fullmatch(r"[a-f0-9]{20}", generation_id):
            raise HTTPException(404, "Ses bulunamadı.")
        entry = history.find(generation_id)
        if not entry or not entry.audio_filename:
            raise HTTPException(404, "Ses bulunamadı.")
        path = paths.output / entry.audio_filename
        if not path.is_file():
            raise HTTPException(404, "Ses dosyası bulunamadı.")
        return FileResponse(path, media_type="audio/wav", filename=entry.audio_filename)

    @app.post("/api/voices")
    async def upload_voice(
        file: Annotated[UploadFile, File()],
        rights_confirmed: Annotated[str, Form()],
    ):
        try:
            if rights_confirmed.lower() not in {"true", "1", "yes"}:
                raise ValueError("Ses kullanım ve klonlama hakkını doğrulamalısınız.")
            payload = await file.read(MAX_VOICE_BYTES + 1)
            if len(payload) > MAX_VOICE_BYTES:
                raise ValueError("Referans WAV en fazla 25 MB olabilir.")
            metadata = read_wav_metadata(payload)
            if metadata.sample_width != 2 or metadata.channels not in {1, 2}:
                raise ValueError("Referans 16-bit PCM, tek veya çift kanallı WAV olmalıdır.")
            if not (8_000 <= metadata.sample_rate <= 96_000):
                raise ValueError("Referans örnekleme hızı 8–96 kHz arasında olmalıdır.")
            if not (2 <= metadata.duration_seconds <= 15):
                raise ValueError("Referans ses iki ila on beş saniye arasında olmalıdır.")
            digest = hashlib.sha256(payload).hexdigest()[:16]
            voice_name = validate_voice_filename(f"voice-{digest}.wav")
            destination = paths.voices / voice_name
            if not destination.exists():
                temporary = destination.with_suffix(".wav.tmp")
                temporary.write_bytes(payload)
                temporary.replace(destination)
            original_stem = Path(file.filename or "Referans ses").stem.strip()
            label = (original_stem[:48] or "Referans ses") + " · klon"
            voices[voice_name] = label
            _atomic_json(paths.voices_file, voices)
            return {"voice": {"id": voice_name, "label": label}}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            await file.close()

    @app.post("/api/generate")
    async def generate(payload: FlagshipRequest):
        if not manager.ready:
            raise HTTPException(503, manager.error or "MOSS-TTS modeli henüz hazır değil.")
        if generation_lock.locked():
            raise HTTPException(409, "Başka bir ses üretimi devam ediyor.")
        if payload.voice != "preset" and payload.voice not in voices:
            raise HTTPException(400, "Seçilen referans ses bulunamadı.")

        body: dict[str, Any] = {
            "input": payload.text,
            "language": LANGUAGE_CODE,
            "response_format": "wav",
            "max_speech_tokens": MAX_SPEECH_TOKENS,
            "top_p": payload.top_p,
            "top_k": payload.top_k,
            "repetition_penalty": payload.repetition_penalty,
            "speaker_identity": "synthetic" if payload.voice == "preset" else "unknown",
        }
        if payload.temperature is not None:
            body["temperature"] = payload.temperature
        if payload.instruction:
            body["instructions"] = payload.instruction
        voice_label = "MOSS v1.5 varsayılan sesi"
        if payload.voice != "preset":
            body.update(
                {
                    # /v1/voices publishes profile names without the .wav suffix.
                    "voice": crisp_voice_name(payload.voice),
                    "consent_attestation": (
                        "Kullanım ve klonlama hakkı yerel laboratuvarda doğrulandı."
                    ),
                    "spoken_disclaimer": False,
                    "marking_attestation": (
                        "Yerel kalite testi; C2PA içeriği ve ses filigranı korunmaktadır."
                    ),
                }
            )
            voice_label = voices[payload.voice]

        async with generation_lock:
            started = time.perf_counter()
            try:
                wav_payload, content_type = await asyncio.to_thread(manager.synthesize, body)
                elapsed = time.perf_counter() - started
                metadata = parse_crisp_response(wav_payload, content_type)
            except (RuntimeError, ValueError, httpx.HTTPError) as exc:
                raise HTTPException(502, str(exc)) from exc

            generation_id = uuid.uuid4().hex[:20]
            filename = f"{generation_id}.wav"
            temporary = paths.output / f"{filename}.tmp"
            temporary.write_bytes(wav_payload)
            temporary.replace(paths.output / filename)
            rss_mb = manager.rss_mb()
            entry = HistoryEntry(
                generation_id=generation_id,
                text=payload.text,
                voice=payload.voice,
                voice_label=voice_label,
                duration_seconds=metadata.duration_seconds,
                created_at=datetime.now(UTC).isoformat(),
                latency_seconds=elapsed,
                rtf=elapsed / metadata.duration_seconds,
                rss_mb=rss_mb,
                audio_filename=filename,
                instruction=payload.instruction,
                temperature=payload.temperature,
                top_p=payload.top_p,
                top_k=payload.top_k,
                repetition_penalty=payload.repetition_penalty,
            )
            history.add(entry)
            _save_history(paths, history)
            return entry.public()

    return app


app = create_app(Path(__file__).resolve().parents[2])
