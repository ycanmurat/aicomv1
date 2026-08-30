"""Isolated tests for the laboratory's optional microphone draft transcription."""

import subprocess
import threading
import wave
from array import array
from pathlib import Path
from unittest.mock import Mock

import pytest

from experiments.tts_lab import speech_input


@pytest.fixture
def transcriber(tmp_path, monkeypatch):
    model = tmp_path / "project/models/ggml-large-v3-turbo-q8_0.bin"
    model.parent.mkdir(parents=True)
    model.touch()
    monkeypatch.setattr(speech_input, "_installed_binary", lambda name: f"/tools/{name}")
    return speech_input.LocalTranscriber(tmp_path / "project", tmp_path / "runtime")


def write_pcm(path, *, silent=False):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(array("h", [0 if silent else 1000, 0] * 1600).tobytes())


def completed_process(stdout="", returncode=0):
    process = Mock()
    process.returncode = returncode
    process.communicate.return_value = (stdout, "")
    process.poll.return_value = returncode
    return process


def test_available_requires_existing_model_and_binaries(transcriber):
    assert transcriber.available
    transcriber.whisper = None
    assert not transcriber.available
    transcriber.whisper = "/tools/whisper-cli"
    transcriber.model.unlink()
    assert not transcriber.available


@pytest.mark.parametrize("audio", [b"", None, "not bytes", b"x" * (5 * 1024 * 1024 + 1)])
def test_invalid_recording_never_starts_process(transcriber, monkeypatch, audio):
    popen = Mock()
    monkeypatch.setattr(speech_input.subprocess, "Popen", popen)
    with pytest.raises(ValueError):
        transcriber.transcribe(audio)
    popen.assert_not_called()
    assert not transcriber.runtime_root.exists()


def test_transcribes_locally_with_bounds_and_cleans_temporary_files(transcriber, monkeypatch):
    calls = []

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        if command[0].endswith("ffmpeg"):
            assert Path(command[command.index("-i") + 1]).read_bytes() == b"browser media"
            write_pcm(command[-1])
            return completed_process()
        return completed_process(" <|tr|> Merhaba,\n bugün nasılsınız? [Müzik] ")

    monkeypatch.setattr(speech_input.subprocess, "Popen", popen)
    assert transcriber.transcribe(b"browser media") == "Merhaba, bugün nasılsınız?"
    assert len(calls) == 2
    ffmpeg, whisper = (call[0] for call in calls)
    assert ffmpeg[ffmpeg.index("-protocol_whitelist") + 1] == "file,pipe"
    assert ffmpeg[ffmpeg.index("-format_whitelist") + 1] == "matroska,webm,mov,wav,ogg"
    assert ffmpeg[ffmpeg.index("-t") + 1] == "20"
    assert ffmpeg[ffmpeg.index("-ar") + 1] == "16000"
    assert ffmpeg[ffmpeg.index("-ac") + 1] == "1"
    assert whisper[whisper.index("-l") + 1] == "tr"
    assert whisper[whisper.index("-t") + 1] == "4"
    assert all(flag in whisper for flag in ("-nt", "-np", "-ng"))
    assert all(not kwargs.get("shell") for _, kwargs in calls)
    assert all(kwargs["stdin"] == subprocess.DEVNULL for _, kwargs in calls)
    assert transcriber.model.exists()
    assert not list(transcriber.runtime_root.iterdir())


def test_silence_is_rejected_before_whisper(transcriber, monkeypatch):
    def popen(command, **kwargs):
        assert command[0].endswith("ffmpeg")
        write_pcm(command[-1], silent=True)
        return completed_process()

    monkeypatch.setattr(speech_input.subprocess, "Popen", popen)
    with pytest.raises(ValueError, match="yeterli ses"):
        transcriber.transcribe(b"silent recording")
    assert not list(transcriber.runtime_root.iterdir())


def test_empty_whisper_output_is_rejected(transcriber, monkeypatch):
    def popen(command, **kwargs):
        if command[0].endswith("ffmpeg"):
            write_pcm(command[-1])
        return completed_process("[BLANK_AUDIO] <|nospeech|>")

    monkeypatch.setattr(speech_input.subprocess, "Popen", popen)
    with pytest.raises(ValueError, match="anlaşılır konuşma"):
        transcriber.transcribe(b"recording")


def test_cancelled_before_start_does_not_spawn(transcriber, monkeypatch):
    stop = threading.Event()
    stop.set()
    popen = Mock()
    monkeypatch.setattr(speech_input.subprocess, "Popen", popen)
    with pytest.raises(RuntimeError, match="iptal"):
        transcriber.transcribe(b"recording", stop)
    popen.assert_not_called()
    assert not transcriber._lock.locked()


def test_cancellation_terminates_only_owned_process(transcriber, monkeypatch):
    stop = threading.Event()
    process = Mock()
    process.poll.return_value = None

    def communicate(timeout):
        if not stop.is_set():
            stop.set()
            raise subprocess.TimeoutExpired("ffmpeg", timeout)
        return "", ""

    process.communicate.side_effect = communicate
    monkeypatch.setattr(speech_input.subprocess, "Popen", Mock(return_value=process))
    with pytest.raises(RuntimeError, match="iptal"):
        transcriber.transcribe(b"recording", stop)
    process.terminate.assert_called_once_with()
    process.kill.assert_not_called()
    assert not transcriber._lock.locked()
    assert not list(transcriber.runtime_root.iterdir())


def test_timeout_kills_unresponsive_owned_process(transcriber, monkeypatch):
    process = Mock()
    process.poll.return_value = None
    ticks = iter([0.0, 0.0, 0.0, 61.0])
    monkeypatch.setattr(speech_input.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(speech_input.subprocess, "Popen", Mock(return_value=process))
    process.communicate.side_effect = [subprocess.TimeoutExpired("ffmpeg", 1), ("", "")]
    with pytest.raises(RuntimeError, match="zaman aşımı"):
        transcriber.transcribe(b"recording")
    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    assert not list(transcriber.runtime_root.iterdir())


def test_tool_failure_is_sanitized_and_releases_lock(transcriber, monkeypatch):
    process = completed_process("secret debug output", returncode=1)
    monkeypatch.setattr(speech_input.subprocess, "Popen", Mock(return_value=process))
    with pytest.raises(RuntimeError, match="Ses kaydı çözümlenemedi") as failure:
        transcriber.transcribe(b"bad recording")
    assert "secret" not in str(failure.value)
    assert not transcriber._lock.locked()
    assert not list(transcriber.runtime_root.iterdir())


def test_parallel_transcriptions_are_rejected(transcriber):
    with transcriber._lock, pytest.raises(RuntimeError, match="Başka bir ses kaydı"):
        transcriber.transcribe(b"recording")


def test_cleanup_preserves_turkish_and_real_words():
    assert speech_input._clean_transcript("<|tr|> İyi günler! [00:01] (Müzik)") == "İyi günler!"
    assert speech_input._clean_transcript("  Teşekkür ederim. ") == "Teşekkür ederim."
