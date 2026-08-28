from __future__ import annotations

from pathlib import Path

from aicomv1.config import Settings
from aicomv1.models import ComponentStatus, SpeechAudio
from aicomv1.providers.base import Synthesizer, TTSError
from aicomv1.providers.tts_freya import FreyaSynthesizer
from aicomv1.providers.tts_macos import MacOSSynthesizer


class SilentSynthesizer:
    def status(self) -> ComponentStatus:
        return ComponentStatus("tts-none", True, "Ses üretimi kapalı.")

    def synthesize(self, text: str, output_path: Path) -> SpeechAudio:
        raise TTSError("Ses üretimi kapalı.")


class AutoSynthesizer:
    def __init__(self, settings: Settings) -> None:
        self.fallback: Synthesizer | None = None
        freya = FreyaSynthesizer(settings)
        macos = MacOSSynthesizer(settings)
        if settings.tts_provider == "none":
            self.active: Synthesizer = SilentSynthesizer()
        elif settings.tts_provider == "freya":
            self.active = freya
        elif settings.tts_provider == "macos":
            self.active = macos
        elif freya.status().ready:
            self.active = freya
            self.fallback = macos if macos.status().ready else None
        else:
            self.active = macos

    def status(self) -> ComponentStatus:
        return self.active.status()

    def synthesize(self, text: str, output_path: Path) -> SpeechAudio:
        try:
            return self.active.synthesize(text, output_path)
        except TTSError:
            if self.fallback is None:
                raise
            return self.fallback.synthesize(text, output_path)
