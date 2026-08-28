from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

from aicomv1.config import PROJECT_ROOT, Settings
from aicomv1.models import ComponentStatus, Transcription
from aicomv1.prompt import normalize_language
from aicomv1.providers.base import STTError


class NemotronCppTranscriber:
    name = "nemotron-3.5-asr"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _local_model(self) -> Path | None:
        """Resolve only installed weights; never let the CLI download at runtime."""
        explicit_model = Path(self.settings.nemo_model).expanduser()
        if explicit_model.suffix == ".gguf":
            if not explicit_model.is_absolute():
                explicit_model = PROJECT_ROOT / explicit_model
            return explicit_model.resolve() if explicit_model.is_file() else None

        index_path = self.settings.nemo_binary.parent.parent / "share/nemo-speech/model-index.json"
        try:
            catalog = json.loads(index_path.read_text(encoding="utf-8"))
            for model in catalog.get("models", []):
                if self.settings.nemo_model not in [model.get("repo"), *model.get("aliases", [])]:
                    continue
                for artifact in model.get("artifacts", []):
                    filename = artifact.get("filename", "")
                    if artifact.get("role") != "asr" or not filename.endswith(".gguf"):
                        continue
                    candidate = (
                        self.settings.nemo_model_dir / model["repo"] / model["revision"] / filename
                    ).resolve()
                    if (
                        candidate.is_relative_to(self.settings.nemo_model_dir.resolve())
                        and candidate.is_file()
                    ):
                        return candidate
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None
        return None

    def status(self) -> ComponentStatus:
        if not self.settings.nemo_binary.is_file():
            return ComponentStatus(
                "stt-nemotron", False, f"NeMo-Speech.cpp was not found: {self.settings.nemo_binary}"
            )
        if self._local_model() is None:
            return ComponentStatus(
                "stt-nemotron",
                False,
                f"{self.settings.nemo_model} is not installed locally. "
                "Run ./scripts/bootstrap.sh full or configure a local GGUF path.",
            )
        return ComponentStatus("stt-nemotron", True, f"Ready: {self.settings.nemo_model}")

    @staticmethod
    def _extract_text(output: str) -> str:
        # CLI 0.1.0 emits pretty-printed JSON. Some builds leave a trailing comma,
        # so extract the primary text field before attempting line-based parsing.
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

    def transcribe(self, audio_path: Path, language: str = "tr") -> Transcription:
        language = normalize_language(language)
        status = self.status()
        if not status.ready:
            raise STTError(status.detail)
        model_path = self._local_model()
        if model_path is None:
            raise STTError("The local Nemotron model is no longer available.")
        started = time.perf_counter()
        command = [
            str(self.settings.nemo_binary),
            "transcribe",
            str(audio_path),
            "--model",
            str(model_path),
            "--language",
            "en-US" if language == "en" else "tr-TR",
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
            raise STTError(f"Nemotron failed: {detail}")
        text = self._extract_text(result.stdout)
        return Transcription(
            text=text,
            elapsed_ms=round((time.perf_counter() - started) * 1000),
            provider=self.name,
        )
