from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Any

from aicomv1.config import Settings
from aicomv1.models import ComponentStatus, SpeechAudio
from aicomv1.providers.base import TTSError


class FreyaSynthesizer:
    name = "freyatts-small"
    sample_rate = 48_000

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Any | None = None
        self._lock = Lock()

    def status(self) -> ComponentStatus:
        if self.settings.freya_root.is_dir() and str(self.settings.freya_root) not in sys.path:
            sys.path.insert(0, str(self.settings.freya_root))
            importlib.invalidate_caches()
        package_ready = all(
            importlib.util.find_spec(name) is not None
            for name in ("freyatts", "torch", "voxcpm", "soundfile")
        )
        cache_ready = self.settings.hf_home.is_dir() and any(self.settings.hf_home.iterdir())
        ready = package_ready and cache_ready
        detail = (
            f"FreyaTTS yerel modeli hazır: {self.settings.freya_model}."
            if ready
            else "FreyaTTS paketi/modeli hazır değil; macOS yerel sesine düşülecek."
        )
        return ComponentStatus("tts-freya", ready, detail)

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                try:
                    os.environ.setdefault("HF_HOME", str(self.settings.hf_home))
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
                    if str(self.settings.freya_root) not in sys.path:
                        sys.path.insert(0, str(self.settings.freya_root))
                    from freyatts import FreyaTTS

                    self._model = FreyaTTS.from_pretrained(
                        self.settings.freya_model, device=self.settings.freya_device
                    )
                except Exception as exc:
                    raise TTSError(f"FreyaTTS yüklenemedi: {exc}") from exc
            return self._model

    def synthesize(self, text: str, output_path: Path) -> SpeechAudio:
        clean = " ".join(text.split()).strip()
        if not clean:
            raise TTSError("Boş metin seslendirilemez.")
        started = time.perf_counter()
        model = self._load()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            wav = model.synthesize(clean, steps=self.settings.freya_steps)
            model.save_wav(wav, str(output_path))
        except Exception as exc:
            raise TTSError(f"FreyaTTS ses üretimi başarısız: {exc}") from exc
        return SpeechAudio(
            path=output_path,
            elapsed_ms=round((time.perf_counter() - started) * 1000),
            provider=self.name,
            sample_rate=self.sample_rate,
        )
