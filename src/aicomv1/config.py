from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _path_env(name: str, default: str) -> Path:
    raw = os.getenv(name, default)
    path = Path(raw).expanduser()
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float_env(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _bool_env(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True, slots=True)
class Settings:
    host: str
    port: int
    ollama_url: str
    llm_model: str
    llm_context: int
    llm_max_tokens: int
    llm_temperature: float
    stt_provider: str
    whisper_model: Path
    whisper_executable: str
    ffmpeg_executable: str
    nemo_binary: Path
    nemo_model: str
    nemo_model_dir: Path
    tts_provider: str
    tts_voice: str
    tts_rate: int
    freya_model: str
    freya_device: str
    freya_steps: int
    freya_root: Path
    hf_home: Path
    knowledge_db: Path
    audio_root: Path
    log_level: str
    warmup: bool

    @classmethod
    def from_env(cls) -> Settings:
        settings = cls(
            host=os.getenv("AICOM_HOST", "127.0.0.1"),
            port=_int_env("AICOM_PORT", 7870),
            ollama_url=os.getenv("AICOM_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/"),
            llm_model=os.getenv("AICOM_LLM_MODEL", "qwen3.5:9b"),
            llm_context=_int_env("AICOM_LLM_CONTEXT", 8192),
            llm_max_tokens=_int_env("AICOM_LLM_MAX_TOKENS", 480),
            llm_temperature=_float_env("AICOM_LLM_TEMPERATURE", 0.35),
            stt_provider=os.getenv("AICOM_STT_PROVIDER", "auto").lower(),
            whisper_model=_path_env("AICOM_WHISPER_MODEL", "models/ggml-large-v3-turbo-q8_0.bin"),
            whisper_executable=os.getenv("AICOM_WHISPER_EXECUTABLE", "whisper-cli"),
            ffmpeg_executable=os.getenv("AICOM_FFMPEG_EXECUTABLE", "ffmpeg"),
            nemo_binary=_path_env("AICOM_NEMO_BINARY", ".runtime/nemo-speech/bin/nemo-speech"),
            nemo_model=os.getenv("AICOM_NEMO_MODEL", "nemotron-3.5"),
            nemo_model_dir=_path_env("AICOM_NEMO_MODEL_DIR", "models/nemo-cache"),
            tts_provider=os.getenv("AICOM_TTS_PROVIDER", "auto").lower(),
            tts_voice=os.getenv("AICOM_TTS_VOICE", "Yelda"),
            tts_rate=_int_env("AICOM_TTS_RATE", 185),
            freya_model=os.getenv("AICOM_FREYA_MODEL", "freyavoice/freya-tts"),
            freya_device=os.getenv("AICOM_FREYA_DEVICE", "cpu").lower(),
            freya_steps=_int_env("AICOM_FREYA_STEPS", 16),
            freya_root=_path_env("AICOM_FREYA_ROOT", ".runtime/FreyaTTS"),
            hf_home=_path_env("AICOM_HF_HOME", "models/huggingface"),
            knowledge_db=_path_env("AICOM_KNOWLEDGE_DB", "data/knowledge/knowledge.sqlite3"),
            audio_root=_path_env("AICOM_AUDIO_ROOT", "data/audio"),
            log_level=os.getenv("AICOM_LOG_LEVEL", "INFO").upper(),
            warmup=_bool_env("AICOM_WARMUP", True),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.stt_provider not in {"auto", "whisper", "nemotron"}:
            raise ValueError("AICOM_STT_PROVIDER auto, whisper veya nemotron olmalıdır.")
        if self.tts_provider not in {"auto", "freya", "macos", "none"}:
            raise ValueError("AICOM_TTS_PROVIDER auto, freya, macos veya none olmalıdır.")
        if not 1024 <= self.llm_context <= 131_072:
            raise ValueError("LLM bağlamı 1024 ile 131072 arasında olmalıdır.")
        if not 32 <= self.llm_max_tokens <= 4096:
            raise ValueError("LLM çıktı sınırı 32 ile 4096 arasında olmalıdır.")
        if not 0 <= self.llm_temperature <= 2:
            raise ValueError("LLM sıcaklığı 0 ile 2 arasında olmalıdır.")
        if not 80 <= self.tts_rate <= 350:
            raise ValueError("TTS konuşma hızı 80 ile 350 arasında olmalıdır.")
        if self.freya_device not in {"cpu", "mps", "cuda"}:
            raise ValueError("AICOM_FREYA_DEVICE cpu, mps veya cuda olmalıdır.")
        if not 4 <= self.freya_steps <= 32:
            raise ValueError("AICOM_FREYA_STEPS 4 ile 32 arasında olmalıdır.")
