from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from aicomv1.config import Settings
from aicomv1.models import ComponentStatus, SpeechAudio
from aicomv1.providers.base import TTSError


class MacOSSynthesizer:
    name = "macos-say"
    sample_rate = 24_000

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def status(self) -> ComponentStatus:
        if not shutil.which("say"):
            return ComponentStatus("tts-macos", False, "macOS say komutu bulunamadı.")
        if not shutil.which(self.settings.ffmpeg_executable):
            return ComponentStatus("tts-macos", False, "ffmpeg bulunamadı.")
        return ComponentStatus(
            "tts-macos", True, f"Hazır: {self.settings.tts_voice}, hız {self.settings.tts_rate}"
        )

    def synthesize(self, text: str, output_path: Path) -> SpeechAudio:
        status = self.status()
        if not status.ready:
            raise TTSError(status.detail)
        clean = " ".join(text.split()).strip()
        if not clean:
            raise TTSError("Boş metin seslendirilemez.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="aicom-tts-") as temp_dir:
            aiff = Path(temp_dir) / "speech.aiff"
            spoken = subprocess.run(
                [
                    "say",
                    "-v",
                    self.settings.tts_voice,
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
                raise TTSError(f"macOS TTS çalışmadı: {spoken.stderr.strip()}")
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
            raise TTSError(f"TTS sesi dönüştürülemedi: {converted.stderr.strip()}")
        return SpeechAudio(
            path=output_path,
            elapsed_ms=round((time.perf_counter() - started) * 1000),
            provider=self.name,
            sample_rate=self.sample_rate,
        )
