import json
from dataclasses import replace
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from aicomv1.providers.stt_auto import AutoTranscriber
from aicomv1.providers.stt_nemotron import NemotronCppTranscriber
from aicomv1.providers.stt_whisper import WhisperCppTranscriber, _clean_transcript


def test_whisper_transcript_cleanup() -> None:
    assert _clean_transcript("<|tr|>  Merhaba dünya. [Müzik]") == "Merhaba dünya."
    assert _clean_transcript("<|en|> Hello world. [Music]") == "Hello world."
    assert _clean_transcript("Thank you.") == "Thank you."
    assert _clean_transcript("Teşekkürler") == "Teşekkürler"
    assert _clean_transcript("la la la la la la la la la") == ""


def test_nemotron_json_output_cleanup() -> None:
    output = '{"text":"Merhaba"}\n{"final_text":"dünya"}\n'
    assert NemotronCppTranscriber._extract_text(output) == "Merhaba"


def test_nemotron_pretty_json_output_cleanup() -> None:
    output = """{
      "file": "input.wav",
      "text": "Bugün sesli iletişim deniyoruz.",
      "words": [{"word": "Bugün"},]
    }"""
    assert NemotronCppTranscriber._extract_text(output) == "Bugün sesli iletişim deniyoruz."


@pytest.mark.parametrize(
    ("language", "expected"),
    [("en", "en"), ("EN-us", "en"), ("tr", "tr"), (" tr_TR ", "tr")],
)
def test_whisper_passes_normalized_language(settings, monkeypatch, language, expected) -> None:
    settings.whisper_model.touch()
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return CompletedProcess(command, 0, "Hello." if "-l" in command else "", "")

    monkeypatch.setattr("aicomv1.providers.stt_whisper.shutil.which", lambda name: name)
    monkeypatch.setattr("aicomv1.providers.stt_whisper.subprocess.run", fake_run)
    result = WhisperCppTranscriber(settings).transcribe(Path("input.wav"), language)
    command = calls[-1]
    assert command[command.index("-l") + 1] == expected
    assert result.text == "Hello."


@pytest.mark.parametrize(
    ("language", "expected"),
    [("en", "en-US"), ("en_GB", "en-US"), ("tr", "tr-TR"), ("TR-tr", "tr-TR")],
)
def test_nemotron_passes_locale_and_local_weights(
    settings, monkeypatch, tmp_path, language, expected
) -> None:
    weights = tmp_path / "model.gguf"
    weights.touch()
    settings.nemo_binary.touch()
    configured = replace(settings, nemo_model=str(weights))
    run = Mock(return_value=CompletedProcess([], 0, '{"text":"Hello."}', ""))
    monkeypatch.setattr("aicomv1.providers.stt_nemotron.subprocess.run", run)
    result = NemotronCppTranscriber(configured).transcribe(Path("input.wav"), language)
    command = run.call_args.args[0]
    assert command[command.index("--language") + 1] == expected
    assert command[command.index("--model") + 1] == str(weights.resolve())
    assert result.text == "Hello."


@pytest.mark.parametrize("provider", [WhisperCppTranscriber, NemotronCppTranscriber])
def test_transcribers_reject_unsupported_language_before_running(settings, provider) -> None:
    with pytest.raises(ValueError, match="Language must be en or tr"):
        provider(settings).transcribe(Path("input.wav"), "de-DE")


def test_auto_transcriber_normalizes_locale(settings) -> None:
    provider = AutoTranscriber(settings)
    transcribe = Mock()
    provider.active = SimpleNamespace(transcribe=transcribe)
    provider.transcribe(Path("input.wav"), "EN_US")
    transcribe.assert_called_once_with(Path("input.wav"), "en")


def test_nemotron_resolves_only_the_installed_catalog_artifact(settings, tmp_path) -> None:
    binary = tmp_path / "runtime/bin/nemo-speech"
    binary.parent.mkdir(parents=True)
    binary.touch()
    catalog = binary.parent.parent / "share/nemo-speech/model-index.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "repo": "nvidia/example",
                        "aliases": ["nemotron-3.5"],
                        "revision": "abc",
                        "artifacts": [{"role": "asr", "filename": "model.gguf"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    configured = replace(settings, nemo_binary=binary, nemo_model="nemotron-3.5")
    provider = NemotronCppTranscriber(configured)
    weights = settings.nemo_model_dir / "nvidia/example/abc/model.gguf"
    weights.parent.mkdir(parents=True)
    (weights.parent / "unrelated.gguf").touch()
    assert not provider.status().ready
    weights.touch()
    assert provider.status().ready
    assert provider._local_model() == weights.resolve()
