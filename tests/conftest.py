from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from aicomv1.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return replace(
        Settings.from_env(),
        llm_model="qwen3.5:2b-q4_K_M",
        stt_provider="whisper",
        whisper_model=tmp_path / "whisper.bin",
        tts_provider="none",
        nemo_binary=tmp_path / "nemo-speech",
        nemo_model_dir=tmp_path / "nemo-cache",
        freya_root=tmp_path / "FreyaTTS",
        freya_idle_seconds=0,
        hf_home=tmp_path / "huggingface",
        knowledge_db=tmp_path / "knowledge.sqlite3",
        audio_root=tmp_path / "audio",
        warmup=False,
    )
