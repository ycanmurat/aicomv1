from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from threading import Lock

from aicomv1.config import Settings
from aicomv1.models import ComponentStatus, SpeechAudio
from aicomv1.prompt import normalize_language
from aicomv1.providers.base import TTSError

_VOICE_LINE = re.compile(r"^(.+?)\s+([a-z]{2,3}[_-][A-Za-z0-9_-]+)\s+#", re.MULTILINE)


class MacOSSynthesizer:
    name = "macos-say"
    sample_rate = 24_000

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._voices: dict[str, str] = {}
        self._voices_checked_at = 0.0
        self._voice_lock = Lock()

    def _installed_voices(self) -> dict[str, str]:
        with self._voice_lock:
            if self._voices and time.monotonic() - self._voices_checked_at < 30:
                return self._voices
            try:
                result = subprocess.run(
                    ["say", "-v", "?"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise TTSError(f"Could not inspect installed macOS voices: {exc}") from exc
            if result.returncode != 0:
                raise TTSError(f"Could not inspect installed macOS voices: {result.stderr.strip()}")
            self._voices = {
                name.strip(): locale for name, locale in _VOICE_LINE.findall(result.stdout)
            }
            self._voices_checked_at = time.monotonic()
            return self._voices

    def _voice_for(self, language: str) -> str:
        return self.settings.tts_voice_en if language == "en" else self.settings.tts_voice

    def status(self, language: str = "tr") -> ComponentStatus:
        language = normalize_language(language)
        if not shutil.which("say"):
            return ComponentStatus("tts-macos", False, "The macOS say command was not found.")
        if not shutil.which(self.settings.ffmpeg_executable):
            return ComponentStatus("tts-macos", False, "ffmpeg was not found.")
        voice = self._voice_for(language)
        try:
            voices = self._installed_voices()
        except TTSError as exc:
            return ComponentStatus("tts-macos", False, str(exc))
        if voice not in voices:
            return ComponentStatus(
                "tts-macos",
                False,
                f"The {language} voice '{voice}' is not installed. "
                "Install it in macOS settings before using offline speech.",
            )
        voice_language = voices[voice].replace("_", "-").split("-")[0]
        if voice_language != language:
            return ComponentStatus(
                "tts-macos",
                False,
                f"Voice '{voice}' speaks {voices[voice]}, not the selected language {language}.",
            )
        return ComponentStatus(
            "tts-macos",
            True,
            f"Ready: {voice} ({voices[voice]}), rate {self.settings.tts_rate}.",
        )

    def synthesize(self, text: str, output_path: Path, language: str = "tr") -> SpeechAudio:
        language = normalize_language(language)
        status = self.status(language=language)
        if not status.ready:
            raise TTSError(status.detail)
        clean = " ".join(text.split()).strip()
        if not clean:
            raise TTSError("Empty text cannot be synthesized.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="aicom-tts-") as temp_dir:
            aiff = Path(temp_dir) / "speech.aiff"
            spoken = subprocess.run(
                [
                    "say",
                    "-v",
                    self._voice_for(language),
                    "-r",
                    str(self.settings.tts_rate),
                    "-o",
                    str(aiff),
                    clean,
                ],
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )
            if spoken.returncode != 0:
                raise TTSError(f"macOS TTS failed: {spoken.stderr.strip()}")
            converted = subprocess.run(
                [
                    self.settings.ffmpeg_executable,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(aiff),
                    "-ar",
                    str(self.sample_rate),
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        if converted.returncode != 0:
            raise TTSError(f"TTS audio conversion failed: {converted.stderr.strip()}")
        return SpeechAudio(
            path=output_path,
            elapsed_ms=round((time.perf_counter() - started) * 1000),
            provider=self.name,
            sample_rate=self.sample_rate,
        )
