from __future__ import annotations

from pathlib import Path

from aicomv1.config import Settings
from aicomv1.models import ComponentStatus, SpeechAudio
from aicomv1.prompt import normalize_language
from aicomv1.providers.base import Synthesizer, TTSError
from aicomv1.providers.tts_freya import FreyaSynthesizer
from aicomv1.providers.tts_macos import MacOSSynthesizer


class SilentSynthesizer:
    def status(self, language: str = "tr") -> ComponentStatus:
        normalize_language(language)
        return ComponentStatus("tts-none", True, "Speech synthesis is disabled.")

    def synthesize(self, text: str, output_path: Path, language: str = "tr") -> SpeechAudio:
        normalize_language(language)
        raise TTSError("Speech synthesis is disabled.")


class AutoSynthesizer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.freya = FreyaSynthesizer(settings)
        self.macos = MacOSSynthesizer(settings)
        self.silent = SilentSynthesizer()
        self._freya_failed = False

    def _provider_for(self, language: str) -> Synthesizer:
        if self.settings.tts_provider == "none":
            return self.silent
        if language == "en" or self.settings.tts_provider == "macos":
            return self.macos
        if not self._freya_failed and self.freya.status(language=language).ready:
            return self.freya
        return self.macos

    def status(self, language: str = "tr") -> ComponentStatus:
        language = normalize_language(language)
        return self._provider_for(language).status(language=language)

    def synthesize(self, text: str, output_path: Path, language: str = "tr") -> SpeechAudio:
        language = normalize_language(language)
        provider = self._provider_for(language)
        try:
            return provider.synthesize(text, output_path, language=language)
        except TTSError:
            if provider is not self.freya:
                raise
            # Avoid repeatedly retrying a broken model for each sentence in a turn.
            self._freya_failed = True
            if not self.macos.status(language=language).ready:
                raise
            return self.macos.synthesize(text, output_path, language=language)
