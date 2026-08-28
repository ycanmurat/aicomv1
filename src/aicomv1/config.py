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
    llm_keep_alive_seconds: int
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
    tts_voice_en: str
    tts_rate: int
    freya_model: str
    freya_device: str
    freya_steps: int
    freya_idle_seconds: float
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
            llm_context=_int_env("AICOM_LLM_CONTEXT", 6144),
            llm_keep_alive_seconds=_int_env("AICOM_LLM_KEEP_ALIVE_SECONDS", 180),
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
            tts_voice_en=os.getenv("AICOM_TTS_VOICE_EN", "Samantha"),
            tts_rate=_int_env("AICOM_TTS_RATE", 185),
            freya_model=os.getenv("AICOM_FREYA_MODEL", "freyavoice/freya-tts"),
            freya_device=os.getenv("AICOM_FREYA_DEVICE", "cpu").lower(),
            freya_steps=_int_env("AICOM_FREYA_STEPS", 16),
            freya_idle_seconds=_float_env("AICOM_FREYA_IDLE_SECONDS", 120),
            freya_root=_path_env("AICOM_FREYA_ROOT", ".runtime/FreyaTTS"),
            hf_home=_path_env("AICOM_HF_HOME", "models/huggingface"),
            knowledge_db=_path_env("AICOM_KNOWLEDGE_DB", "data/knowledge/knowledge.sqlite3"),
            audio_root=_path_env("AICOM_AUDIO_ROOT", "data/audio"),
            log_level=os.getenv("AICOM_LOG_LEVEL", "INFO").upper(),
            warmup=_bool_env("AICOM_WARMUP", False),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.stt_provider not in {"auto", "whisper", "nemotron"}:
            raise ValueError("AICOM_STT_PROVIDER must be auto, whisper, or nemotron.")
        if self.tts_provider not in {"auto", "freya", "macos", "none"}:
            raise ValueError("AICOM_TTS_PROVIDER must be auto, freya, macos, or none.")
        if not 1024 <= self.llm_context <= 131_072:
            raise ValueError("LLM context must be between 1024 and 131072.")
        if not 0 <= self.llm_keep_alive_seconds <= 3600:
            raise ValueError("LLM keep-alive must be between 0 and 3600 seconds.")
        if not 32 <= self.llm_max_tokens <= 4096:
            raise ValueError("LLM output limit must be between 32 and 4096.")
        if not 0 <= self.llm_temperature <= 2:
            raise ValueError("LLM temperature must be between 0 and 2.")
        if not 80 <= self.tts_rate <= 350:
            raise ValueError("TTS speech rate must be between 80 and 350.")
        if self.freya_device not in {"cpu", "mps", "cuda"}:
            raise ValueError("AICOM_FREYA_DEVICE must be cpu, mps, or cuda.")
        if not 4 <= self.freya_steps <= 32:
            raise ValueError("AICOM_FREYA_STEPS must be between 4 and 32.")
        if not 0 <= self.freya_idle_seconds <= 3600:
            raise ValueError("Freya idle timeout must be between 0 and 3600 seconds.")
