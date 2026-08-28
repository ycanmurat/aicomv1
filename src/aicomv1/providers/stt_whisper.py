from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from aicomv1.config import Settings
from aicomv1.models import ComponentStatus, Transcription
from aicomv1.prompt import normalize_language
from aicomv1.providers.base import STTError

_TOKEN = re.compile(r"<\|[^>]+\|>")
_BRACKETS = re.compile(r"\s*[\[(][^\])]{0,80}[\])]\s*")
_NO_SPEECH = {
    "altyazı m.k.",
    "altyazı",
    "müzik",
}


def _clean_transcript(raw: str) -> str:
    text = _TOKEN.sub(" ", raw)
    text = _BRACKETS.sub(" ", text)
    lines = []
    for line in text.splitlines():
        clean = " ".join(line.split()).strip(" -")
        if clean and clean.casefold() not in _NO_SPEECH:
            lines.append(clean)
    merged = " ".join(lines).strip()
    words = merged.split()
    if len(words) >= 8 and len(set(word.casefold() for word in words)) <= 2:
        return ""
    return merged


class WhisperCppTranscriber:
    name = "whisper.cpp"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def status(self) -> ComponentStatus:
        executable = shutil.which(self.settings.whisper_executable)
        if not executable:
            return ComponentStatus("stt-whisper", False, "whisper-cli was not found.")
        if not shutil.which(self.settings.ffmpeg_executable):
            return ComponentStatus("stt-whisper", False, "ffmpeg was not found.")
        if not self.settings.whisper_model.is_file():
            return ComponentStatus(
                "stt-whisper", False, f"Model was not found: {self.settings.whisper_model}"
            )
        return ComponentStatus("stt-whisper", True, f"Ready: {self.settings.whisper_model.name}")

    def transcribe(self, audio_path: Path, language: str = "tr") -> Transcription:
        language = normalize_language(language)
        status = self.status()
        if not status.ready:
            raise STTError(status.detail)
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="aicom-whisper-") as temp_dir:
            normalized = Path(temp_dir) / "input.wav"
            normalize = subprocess.run(
                [
                    self.settings.ffmpeg_executable,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(audio_path),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    str(normalized),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if normalize.returncode != 0:
                raise STTError(f"Audio conversion failed: {normalize.stderr.strip()}")
            command = [
                self.settings.whisper_executable,
                "-m",
                str(self.settings.whisper_model),
                "-f",
                str(normalized),
                "-l",
                language,
                "-nt",
                "-np",
            ]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=150,
                check=False,
            )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[-1000:]
            raise STTError(f"Whisper failed: {detail}")
        text = _clean_transcript(result.stdout)
        return Transcription(
            text=text,
            elapsed_ms=round((time.perf_counter() - started) * 1000),
            provider=self.name,
        )
