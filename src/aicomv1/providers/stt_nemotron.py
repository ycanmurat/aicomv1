from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

from aicomv1.config import Settings
from aicomv1.models import ComponentStatus, Transcription
from aicomv1.providers.base import STTError


class NemotronCppTranscriber:
    name = "nemotron-3.5-asr"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def status(self) -> ComponentStatus:
        if not self.settings.nemo_binary.is_file():
            return ComponentStatus(
                "stt-nemotron", False, f"NeMo-Speech.cpp bulunamadı: {self.settings.nemo_binary}"
            )
        explicit_model = Path(self.settings.nemo_model).expanduser()
        if explicit_model.suffix == ".gguf":
            if not explicit_model.is_absolute():
                explicit_model = (self.settings.nemo_model_dir.parent / explicit_model).resolve()
            if not explicit_model.is_file():
                return ComponentStatus(
                    "stt-nemotron", False, f"Nemotron modeli bulunamadı: {explicit_model}"
                )
        elif not self.settings.nemo_model_dir.is_dir() or not any(
            self.settings.nemo_model_dir.rglob("*.gguf")
        ):
            return ComponentStatus(
                "stt-nemotron",
                False,
                f"{self.settings.nemo_model} yerel önbellekte yok: {self.settings.nemo_model_dir}",
            )
        return ComponentStatus("stt-nemotron", True, f"Hazır: {self.settings.nemo_model}")

    @staticmethod
    def _extract_text(output: str) -> str:
        # CLI 0.1.0 okunabilir, çok satırlı JSON üretir. Bazı sürümler son dizi
        # elemanından sonra virgül bıraktığı için önce yalnız asıl metin alanını alırız.
        match = re.search(r'"text"\s*:\s*"((?:\\.|[^"\\])*)"', output)
        if match:
            try:
                return str(json.loads(f'"{match.group(1)}"')).strip()
            except json.JSONDecodeError:
                return match.group(1).strip()
        candidates: list[str] = []
        for line in output.splitlines():
            clean = line.strip()
            if not clean:
                continue
            try:
                item = json.loads(clean)
            except json.JSONDecodeError:
                if not clean.startswith("["):
                    candidates.append(clean)
                continue
            if isinstance(item, dict):
                for key in ("text", "transcript", "final_text", "result"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        candidates.append(value.strip())
                        break
        return " ".join(candidates).strip()

    def transcribe(self, audio_path: Path) -> Transcription:
        status = self.status()
        if not status.ready:
            raise STTError(status.detail)
        started = time.perf_counter()
        command = [
            str(self.settings.nemo_binary),
            "transcribe",
            str(audio_path),
            "--model",
            self.settings.nemo_model,
            "--language",
            "tr-TR",
            "--device",
            "metal",
            "--format",
            "json",
        ]
        environment = os.environ.copy()
        environment["NEMO_SPEECH_MODEL_DIR"] = str(self.settings.nemo_model_dir)
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=150,
            check=False,
            env=environment,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[-1000:]
            raise STTError(f"Nemotron çalışmadı: {detail}")
        text = self._extract_text(result.stdout)
        return Transcription(
            text=text,
            elapsed_ms=round((time.perf_counter() - started) * 1000),
            provider=self.name,
        )
