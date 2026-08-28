from __future__ import annotations

from pathlib import Path

from aicomv1.config import Settings
from aicomv1.models import ComponentStatus, Transcription
from aicomv1.prompt import normalize_language
from aicomv1.providers.base import Transcriber
from aicomv1.providers.stt_nemotron import NemotronCppTranscriber
from aicomv1.providers.stt_whisper import WhisperCppTranscriber


class AutoTranscriber:
    def __init__(self, settings: Settings) -> None:
        whisper = WhisperCppTranscriber(settings)
        nemotron = NemotronCppTranscriber(settings)
        if settings.stt_provider == "nemotron":
            self.active: Transcriber = nemotron
        elif settings.stt_provider == "whisper":
            self.active = whisper
        else:
            # Whisper was more accurate in our Turkish voice tests. Nemotron remains
            # available as an explicit lower-latency option.
            self.active = whisper if whisper.status().ready else nemotron

    def status(self) -> ComponentStatus:
        return self.active.status()

    def transcribe(self, audio_path: Path, language: str = "tr") -> Transcription:
        return self.active.transcribe(audio_path, normalize_language(language))
