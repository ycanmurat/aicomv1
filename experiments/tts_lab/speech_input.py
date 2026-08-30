"""Optional, local-only Turkish transcription for a user-reviewed prompt draft.

No model is downloaded or kept resident. Each call owns its temporary files and
child processes; cancelling never stops another application or shared service.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
import wave
from array import array
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory

MAX_AUDIO_BYTES = 5 * 1024 * 1024
MAX_AUDIO_SECONDS = 20
TIMEOUT_SECONDS = 60.0


class LocalTranscriber:
    """Transcribe an uploaded recording, without recording or submitting it.

    ``available`` only checks installed dependencies. ``transcribe`` returns a
    draft for the user to review; it must not trigger tools or send a prompt.
    Invalid/silent audio raises ValueError; operational failures raise RuntimeError.
    """

    def __init__(self, project_root: Path, runtime_root: Path) -> None:
        self.model = project_root / "models" / "ggml-large-v3-turbo-q8_0.bin"
        self.runtime_root = runtime_root
        self.whisper = _installed_binary("whisper-cli")
        self.ffmpeg = _installed_binary("ffmpeg")
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return bool(self.whisper and self.ffmpeg and self.model.is_file())

    def transcribe(self, audio_bytes: bytes, stop_event: threading.Event | None = None) -> str:
        if not isinstance(audio_bytes, bytes) or not audio_bytes:
            raise ValueError("Ses kaydı boş veya geçersiz.")
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise ValueError("Ses kaydı en fazla 5 MB olabilir.")
        if not self.available:
            raise RuntimeError("Yerel ses tanıma için whisper-cli, ffmpeg ve mevcut model gerekli.")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Başka bir ses kaydı çözümleniyor. Lütfen bitmesini bekleyin.")

        deadline = time.monotonic() + TIMEOUT_SECONDS
        try:
            _check_deadline(deadline, stop_event)
            self.runtime_root.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix="speech-input-", dir=self.runtime_root) as temporary:
                source = Path(temporary) / "recording"
                normalized = Path(temporary) / "recording.wav"
                source.write_bytes(audio_bytes)
                self._run(
                    [
                        self.ffmpeg,
                        "-nostdin",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-protocol_whitelist",
                        "file,pipe",
                        "-format_whitelist",
                        "matroska,webm,mov,wav,ogg",
                        "-threads",
                        "1",
                        "-i",
                        str(source),
                        "-map",
                        "0:a:0",
                        "-vn",
                        "-sn",
                        "-dn",
                        "-t",
                        str(MAX_AUDIO_SECONDS),
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        "-c:a",
                        "pcm_s16le",
                        "-threads",
                        "1",
                        str(normalized),
                    ],
                    deadline,
                    stop_event,
                    "Ses kaydı çözümlenemedi. WebM, MP4, Ogg veya WAV kaydı deneyin.",
                )
                _require_audible_audio(normalized)
                output = self._run(
                    [
                        self.whisper,
                        "-m",
                        str(self.model),
                        "-f",
                        str(normalized),
                        "-l",
                        "tr",
                        "-nt",
                        "-np",
                        "-ng",
                        "-t",
                        "4",
                    ],
                    deadline,
                    stop_event,
                    "Yerel ses tanıma tamamlanamadı. Daha kısa bir kayıt deneyin.",
                )
                _check_deadline(deadline, stop_event)
                text = _clean_transcript(output)
                if not text:
                    raise ValueError("Kayıtta anlaşılır konuşma bulunamadı. Lütfen tekrar deneyin.")
                return text
        finally:
            self._lock.release()

    @staticmethod
    def _run(
        command: list[str],
        deadline: float,
        stop_event: threading.Event | None,
        failure_message: str,
    ) -> str:
        _check_deadline(deadline, stop_event)
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            raise RuntimeError(failure_message) from exc
        try:
            while True:
                _check_deadline(deadline, stop_event)
                try:
                    output, _ = process.communicate(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    continue
            if process.returncode != 0:
                raise RuntimeError(failure_message)
            return output
        finally:
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    process.terminate()
                try:
                    process.communicate(timeout=1)
                except subprocess.TimeoutExpired:
                    with suppress(ProcessLookupError):
                        process.kill()
                    process.communicate(timeout=1)


def _installed_binary(name: str) -> str | None:
    candidate = shutil.which(name) or f"/opt/homebrew/bin/{name}"
    return candidate if Path(candidate).is_file() and os.access(candidate, os.X_OK) else None


def _check_deadline(deadline: float, stop_event: threading.Event | None) -> None:
    if stop_event is not None and stop_event.is_set():
        raise RuntimeError("Ses tanıma iptal edildi.")
    if time.monotonic() >= deadline:
        raise RuntimeError("Ses tanıma zaman aşımına uğradı. Daha kısa bir kayıt deneyin.")


def _require_audible_audio(path: Path) -> None:
    try:
        with wave.open(str(path), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
                raise ValueError("Ses kaydı desteklenen biçime dönüştürülemedi.")
            samples = array("h", audio.readframes(MAX_AUDIO_SECONDS * 16000))
    except (OSError, EOFError, wave.Error) as exc:
        raise ValueError("Ses kaydı okunamadı.") from exc
    if sys.byteorder != "little":
        samples.byteswap()
    # Reject silence before Whisper can hallucinate a sentence over an empty clip.
    if len(samples) < 1600 or math.sqrt(sum(s * s for s in samples) / len(samples)) < 32:
        raise ValueError("Kayıtta yeterli ses bulunamadı. Mikrofona biraz daha yakın konuşun.")


def _clean_transcript(text: str) -> str:
    text = re.sub(r"<\|.*?\|>|\[[^\]]*\]", " ", text)
    text = re.sub(r"\((?:müzik|sessizlik|alkış|music|silence|no speech)\)", " ", text, flags=re.I)
    return unicodedata.normalize("NFC", " ".join(text.split())).strip()
