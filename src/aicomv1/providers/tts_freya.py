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
from aicomv1.prompt import normalize_language
from aicomv1.providers.base import TTSError


class FreyaSynthesizer:
    name = "freyatts-small"
    sample_rate = 48_000

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Any | None = None
        self._lock = Lock()
        self._inference_lock = Lock()

    def _cached_files(self, repo: str, filenames: tuple[str, ...]) -> bool:
        cache_root = self.settings.hf_home / "hub" / f"models--{repo.replace('/', '--')}"
        try:
            revision = (cache_root / "refs/main").read_text(encoding="utf-8").strip()
        except OSError:
            return False
        snapshot = (cache_root / "snapshots" / revision).resolve()
        if not snapshot.is_relative_to((cache_root / "snapshots").resolve()):
            return False
        return all((snapshot / filename).is_file() for filename in filenames)

    def status(self, language: str = "tr") -> ComponentStatus:
        if normalize_language(language) != "tr":
            return ComponentStatus("tts-freya", False, "FreyaTTS supports Turkish speech only.")
        if self.settings.freya_root.is_dir() and str(self.settings.freya_root) not in sys.path:
            sys.path.insert(0, str(self.settings.freya_root))
            importlib.invalidate_caches()
        package_ready = all(
            importlib.util.find_spec(name) is not None
            for name in ("freyatts", "torch", "voxcpm", "soundfile")
        )
        model_directory = Path(self.settings.freya_model).expanduser()
        model_files = ("config.json", "model.safetensors")
        model_ready = (
            all((model_directory / filename).is_file() for filename in model_files)
            if model_directory.is_dir()
            else self._cached_files(self.settings.freya_model, model_files)
        )
        cache_ready = model_ready and self._cached_files("openbmb/VoxCPM2", ("audiovae.pth",))
        ready = package_ready and cache_ready
        detail = (
            f"Local FreyaTTS model is ready: {self.settings.freya_model}."
            if ready
            else "FreyaTTS code, model, or AudioVAE is not installed locally."
        )
        return ComponentStatus("tts-freya", ready, detail)

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                try:
                    os.environ["HF_HOME"] = str(self.settings.hf_home)
                    os.environ["HF_HUB_CACHE"] = str(self.settings.hf_home / "hub")
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"
                    from huggingface_hub import constants as hub_constants

                    # Enforce offline lookup even if another module imported the hub first.
                    hub_constants.HF_HUB_OFFLINE = True
                    hub_constants.HF_HUB_CACHE = str(self.settings.hf_home / "hub")
                    if str(self.settings.freya_root) not in sys.path:
                        sys.path.insert(0, str(self.settings.freya_root))
                    from freyatts import FreyaTTS

                    self._model = FreyaTTS.from_pretrained(
                        self.settings.freya_model, device=self.settings.freya_device
                    )
                except Exception as exc:
                    raise TTSError(f"FreyaTTS could not be loaded: {exc}") from exc
            return self._model

    def synthesize(self, text: str, output_path: Path, language: str = "tr") -> SpeechAudio:
        if normalize_language(language) != "tr":
            raise TTSError("FreyaTTS supports Turkish only; use a local English voice instead.")
        clean = " ".join(text.split()).strip()
        if not clean:
            raise TTSError("Empty text cannot be synthesized.")
        started = time.perf_counter()
        model = self._load()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # A cancelled worker may still be running; never share the model concurrently.
            with self._inference_lock:
                wav = model.synthesize(clean, steps=self.settings.freya_steps)
                model.save_wav(wav, str(output_path))
        except Exception as exc:
            raise TTSError(f"FreyaTTS synthesis failed: {exc}") from exc
        return SpeechAudio(
            path=output_path,
            elapsed_ms=round((time.perf_counter() - started) * 1000),
            provider=self.name,
            sample_rate=self.sample_rate,
        )
