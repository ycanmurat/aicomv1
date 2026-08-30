"""Deterministic unit tests for the MOSS-TTS-v1.5 flagship lab server.

These tests deliberately do not start CrispASR or load the model. They define
the small, public boundary helpers the HTTP layer can exercise independently.
"""

from __future__ import annotations

import io
import json
import math
import wave
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from experiments.tts_lab.flagship_server import (
    MAX_SPEECH_TOKENS,
    FlagshipRequest,
    HistoryEntry,
    HistoryStore,
    create_app,
    crisp_voice_name,
    parse_crisp_response,
    process_memory_status,
    read_wav_metadata,
    validate_voice_filename,
)


def wav_bytes(
    *, sample_rate: int = 24_000, channels: int = 1, frames: int = 2_400
) -> bytes:
    """Create a small 16-bit PCM fixture without audio or model dependencies."""

    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\x00\x00" * channels * frames)
    return output.getvalue()


def test_request_strips_text_and_uses_official_sampling_defaults() -> None:
    request = FlagshipRequest(text="  Merhaba, ben Fatma.  ")

    assert request.text == "Merhaba, ben Fatma."
    # Omitting the shared CrispASR override preserves MOSS's distinct text and
    # audio temperature defaults.
    assert request.temperature is None
    assert request.top_p == 0.8
    assert request.top_k == 25
    assert request.repetition_penalty == 1.0


@pytest.mark.parametrize("text", ["", "   \t\n", None, "a" * 601])
def test_request_rejects_empty_missing_or_oversized_text(text) -> None:
    with pytest.raises(ValueError):
        FlagshipRequest(text=text)


def test_request_limits_text_after_trimming_without_language_guessing() -> None:
    assert len(FlagshipRequest(text=f"  {'ğ' * 600}  ").text) == 600
    assert FlagshipRequest(text="Hello").text == "Hello"


@pytest.mark.parametrize(
    ("field", "invalid_values"),
    [
        ("temperature", (0.099, 3.001, math.nan, math.inf, -math.inf, True, "1.7")),
        ("top_p", (0.0, 1.001, math.nan, math.inf, -math.inf, True, "0.8")),
        ("top_k", (0, 101, 2.5, True, "25")),
        (
            "repetition_penalty",
            (0.499, 2.001, math.nan, math.inf, -math.inf, True, "1.0"),
        ),
    ],
)
def test_request_rejects_out_of_range_or_coerced_sampling_values(
    field: str, invalid_values: tuple[object, ...]
) -> None:
    for value in invalid_values:
        with pytest.raises(ValueError):
            FlagshipRequest(text="Merhaba", **{field: value})


def test_request_accepts_sampling_boundaries() -> None:
    low = FlagshipRequest(
        text="Merhaba",
        temperature=0.1,
        top_p=0.000001,
        top_k=1,
        repetition_penalty=0.5,
    )
    high = FlagshipRequest(
        text="Merhaba",
        temperature=3.0,
        top_p=1.0,
        top_k=100,
        repetition_penalty=2.0,
    )

    assert (low.temperature, low.top_p, low.top_k, low.repetition_penalty) == (
        0.1,
        0.000001,
        1,
        0.5,
    )
    assert (high.temperature, high.top_p, high.top_k, high.repetition_penalty) == (
        3.0,
        1.0,
        100,
        2.0,
    )


def test_wav_metadata_reports_real_format_and_duration() -> None:
    metadata = read_wav_metadata(wav_bytes(sample_rate=24_000, channels=1, frames=6_000))

    assert metadata.sample_rate == 24_000
    assert metadata.channels == 1
    assert metadata.sample_width == 2
    assert metadata.frame_count == 6_000
    assert metadata.duration_seconds == pytest.approx(0.25)


@pytest.mark.parametrize("payload", [b"", b"not a wav", b"RIFF\x00\x00\x00\x00WAVE"])
def test_wav_metadata_rejects_empty_or_malformed_audio(payload: bytes) -> None:
    with pytest.raises(ValueError):
        read_wav_metadata(payload)


def test_history_is_newest_first_and_discards_the_oldest_entry() -> None:
    history = HistoryStore(max_items=2)
    entries = [
        HistoryEntry(
            generation_id=f"generation-{index}",
            text=f"Cümle {index}",
            voice="fatma.wav",
            duration_seconds=float(index),
            created_at=float(index),
        )
        for index in range(1, 4)
    ]

    for entry in entries:
        history.add(entry)

    assert [entry.generation_id for entry in history.items()] == [
        "generation-3",
        "generation-2",
    ]


def test_history_rejects_a_nonpositive_limit() -> None:
    with pytest.raises(ValueError):
        HistoryStore(max_items=0)


@pytest.mark.parametrize("name", ["fatma.wav", "fatma_01.wav", "demo-v2.wav"])
def test_voice_filename_accepts_plain_wav_basenames(name: str) -> None:
    assert validate_voice_filename(name) == name


def test_crisp_voice_name_removes_only_the_validated_wav_suffix() -> None:
    assert crisp_voice_name("ada-sentetik.wav") == "ada-sentetik"


def test_crisp_voice_name_rejects_a_path_before_converting_it() -> None:
    with pytest.raises(ValueError):
        crisp_voice_name("../ada-sentetik.wav")


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "..",
        ".hidden.wav",
        "../fatma.wav",
        "voices/fatma.wav",
        r"voices\fatma.wav",
        "/tmp/fatma.wav",
        "fatma.mp3",
        "fatma.wav\x00",
        "%2e%2e%2ffatma.wav",
        "fatma voice.wav",
    ],
)
def test_voice_filename_rejects_paths_traversal_and_non_wav_names(name: str) -> None:
    with pytest.raises(ValueError):
        validate_voice_filename(name)


@pytest.mark.parametrize("content_type", ["audio/wav", "audio/x-wav", "audio/wave"])
def test_crisp_response_accepts_raw_wav_audio(content_type: str) -> None:
    payload = wav_bytes()
    metadata = parse_crisp_response(payload, content_type)

    assert metadata.sample_rate == 24_000
    assert metadata.channels == 1
    assert metadata.duration_seconds == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("payload", "content_type"),
    [
        (b'{"error":"model failed"}', "application/json"),
        (b'{"error":"model failed"}', "audio/wav"),
        (b"upstream failed", "text/plain"),
        (b"not a wave", "audio/wav"),
        (b"", "audio/wav"),
    ],
)
def test_crisp_response_rejects_json_errors_wrong_types_and_malformed_audio(
    payload: bytes, content_type: str
) -> None:
    with pytest.raises(ValueError):
        parse_crisp_response(payload, content_type)


def test_process_memory_status_uses_only_the_supplied_process_rss() -> None:
    process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=129_446_707))
    calls = []

    def process_factory():
        calls.append(True)
        return process

    status = process_memory_status(process_factory=process_factory)

    assert calls == [True]
    assert status["process_rss_mb"] == 123.5
    assert "system_memory_mb" not in status


def test_generate_uses_crisp_catalog_name_and_bounded_audio_frames(tmp_path) -> None:
    runtime = tmp_path / ".runtime/voice-lab/moss-v15-flagship"
    voices = runtime / "voices"
    voices.mkdir(parents=True)
    (voices / "ada-sentetik.wav").write_bytes(wav_bytes())
    (runtime / "voices.json").write_text(
        json.dumps({"ada-sentetik.wav": "Ada · sentetik referans"}),
        encoding="utf-8",
    )

    class FakeManager:
        ready = True
        error = None
        process = SimpleNamespace(poll=lambda: None)
        request = None

        def start(self) -> None:
            pass

        def shutdown(self) -> None:
            pass

        def rss_mb(self) -> float:
            return 123.5

        def synthesize(self, payload):
            self.request = payload
            return wav_bytes(frames=24_000), "audio/wav"

    manager = FakeManager()
    app = create_app(tmp_path, manager=manager)
    with TestClient(app) as client:
        response = client.post(
            "/api/generate",
            headers={"X-Fatma-Lab": "flagship-v1"},
            json={"text": "Merhaba.", "voice": "ada-sentetik.wav"},
        )

    assert response.status_code == 200
    assert manager.request["voice"] == "ada-sentetik"
    assert manager.request["max_speech_tokens"] == MAX_SPEECH_TOKENS
    assert manager.request["language"] == "tr"
    assert "temperature" not in manager.request
