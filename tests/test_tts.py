import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from subprocess import CompletedProcess
from threading import Event, Lock
from types import SimpleNamespace

import pytest

from aicomv1.models import ComponentStatus, SpeechAudio
from aicomv1.providers.base import TTSError
from aicomv1.providers.tts_auto import AutoSynthesizer
from aicomv1.providers.tts_freya import FreyaSynthesizer
from aicomv1.providers.tts_macos import MacOSSynthesizer


class FakeSynthesizer:
    def __init__(self, name: str, languages: tuple[str, ...]) -> None:
        self.name = name
        self.languages = languages
        self.calls: list[str] = []
        self.fail = False

    def status(self, language: str = "tr") -> ComponentStatus:
        return ComponentStatus(self.name, language in self.languages, "Local test provider")

    def synthesize(self, text: str, output_path: Path, language: str = "tr") -> SpeechAudio:
        self.calls.append(language)
        if self.fail or language not in self.languages:
            raise TTSError("Test voice is unavailable.")
        return SpeechAudio(output_path, 1, self.name, 24_000)


@pytest.fixture
def fake_voices(monkeypatch):
    freya = FakeSynthesizer("tts-freya", ("tr",))
    macos = FakeSynthesizer("tts-macos", ("en", "tr"))
    monkeypatch.setattr("aicomv1.providers.tts_auto.FreyaSynthesizer", lambda _: freya)
    monkeypatch.setattr("aicomv1.providers.tts_auto.MacOSSynthesizer", lambda _: macos)
    return freya, macos


@pytest.mark.parametrize(
    ("setting", "language", "expected"),
    [
        ("auto", "en", "tts-macos"),
        ("auto", "en_US", "tts-macos"),
        ("auto", "tr-TR", "tts-freya"),
        ("freya", "en", "tts-macos"),
        ("freya", "tr", "tts-freya"),
        ("macos", "tr", "tts-macos"),
    ],
)
def test_auto_routes_and_reports_selected_language(
    settings, fake_voices, tmp_path, setting, language, expected
) -> None:
    provider = AutoSynthesizer(replace(settings, tts_provider=setting))
    status = provider.status(language=language)
    assert status.name == expected
    assert status.ready
    assert provider.synthesize("Test", tmp_path / "test.wav", language).provider == expected


@pytest.mark.parametrize("language", ["en", "tr"])
def test_none_stays_disabled_for_every_language(settings, fake_voices, tmp_path, language) -> None:
    provider = AutoSynthesizer(replace(settings, tts_provider="none"))
    assert provider.status(language).name == "tts-none"
    with pytest.raises(TTSError, match="disabled"):
        provider.synthesize("Test", tmp_path / "test.wav", language)
    assert all(not voice.calls for voice in fake_voices)


def test_missing_english_voice_never_falls_back_to_turkish_freya(
    settings, fake_voices, tmp_path
) -> None:
    freya, macos = fake_voices
    macos.languages = ("tr",)
    provider = AutoSynthesizer(replace(settings, tts_provider="auto"))
    assert not provider.status("en").ready
    assert provider.status("en").name == "tts-macos"
    with pytest.raises(TTSError):
        provider.synthesize("Hello", tmp_path / "test.wav", "en")
    assert not freya.calls


def test_unavailable_freya_uses_local_turkish_voice(settings, fake_voices) -> None:
    freya, _ = fake_voices
    freya.languages = ()
    provider = AutoSynthesizer(replace(settings, tts_provider="freya"))
    assert provider.status("tr").name == "tts-macos"


def test_failed_freya_falls_back_without_retrying_every_sentence(
    settings, fake_voices, tmp_path
) -> None:
    freya, macos = fake_voices
    freya.fail = True
    provider = AutoSynthesizer(replace(settings, tts_provider="auto"))
    provider.synthesize("Merhaba", tmp_path / "first.wav", "tr")
    provider.synthesize("Devam", tmp_path / "second.wav", "tr")
    assert freya.calls == ["tr"]
    assert macos.calls == ["tr", "tr"]
    assert provider.status("tr").name == "tts-macos"


def test_freya_rejects_english_before_model_loading(settings, monkeypatch, tmp_path) -> None:
    provider = FreyaSynthesizer(settings)
    monkeypatch.setattr(provider, "_load", lambda: pytest.fail("Model should not be loaded"))
    assert not provider.status("en_US").ready
    with pytest.raises(TTSError, match="Turkish only"):
        provider.synthesize("Hello", tmp_path / "test.wav", "en-US")


def test_freya_requires_cached_weights_and_audio_vae(settings, monkeypatch) -> None:
    monkeypatch.setattr("aicomv1.providers.tts_freya.importlib.util.find_spec", lambda _: object())
    settings.hf_home.mkdir(parents=True)
    (settings.hf_home / "unrelated-file").touch()
    provider = FreyaSynthesizer(settings)
    assert not provider.status("tr").ready
    for repo, files in (
        (settings.freya_model, ("config.json", "model.safetensors")),
        ("openbmb/VoxCPM2", ("audiovae.pth",)),
    ):
        cache = settings.hf_home / "hub" / f"models--{repo.replace('/', '--')}"
        (cache / "refs").mkdir(parents=True)
        (cache / "refs/main").write_text("revision", encoding="utf-8")
        snapshot = cache / "snapshots/revision"
        snapshot.mkdir(parents=True)
        for filename in files:
            (snapshot / filename).touch()
    assert provider.status("tr").ready


def test_freya_serializes_model_inference(settings, monkeypatch, tmp_path) -> None:
    provider = FreyaSynthesizer(settings)
    active = 0
    highest_active = 0
    counter_lock = Lock()

    def synthesize(text, **kwargs):
        nonlocal active, highest_active
        with counter_lock:
            active += 1
            highest_active = max(highest_active, active)
        time.sleep(0.01)
        with counter_lock:
            active -= 1
        return b"audio"

    model = SimpleNamespace(synthesize=synthesize, save_wav=lambda *args: None)
    monkeypatch.setattr(provider, "_load", lambda: model)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(provider.synthesize, "Merhaba", tmp_path / f"{index}.wav", "tr")
            for index in range(2)
        ]
        for future in futures:
            future.result(timeout=2)
    assert highest_active == 1


def test_freya_releases_loaded_model_after_idle_timeout(settings, monkeypatch, tmp_path) -> None:
    provider = FreyaSynthesizer(replace(settings, freya_idle_seconds=0.02))
    model = SimpleNamespace(
        synthesize=lambda *args, **kwargs: b"audio", save_wav=lambda *args: None
    )
    released = []
    provider._model = model
    monkeypatch.setattr(provider, "_release_runtime_memory", lambda: released.append(True))

    provider.synthesize("Merhaba", tmp_path / "idle.wav", "tr")
    deadline = time.monotonic() + 1
    while provider._model is not None and time.monotonic() < deadline:
        time.sleep(0.01)

    assert provider._model is None
    assert released == [True]


def test_zero_freya_idle_timeout_keeps_model_loaded(settings, monkeypatch, tmp_path) -> None:
    provider = FreyaSynthesizer(replace(settings, freya_idle_seconds=0))
    model = SimpleNamespace(
        synthesize=lambda *args, **kwargs: b"audio", save_wav=lambda *args: None
    )
    released = []
    provider._model = model
    monkeypatch.setattr(provider, "_release_runtime_memory", lambda: released.append(True))

    provider.synthesize("Merhaba", tmp_path / "resident.wav", "tr")

    assert provider._model is model
    assert released == []
    assert provider.unload()
    assert released == [True]


def test_stale_freya_timer_cannot_unload_reused_model(settings, monkeypatch) -> None:
    provider = FreyaSynthesizer(replace(settings, freya_idle_seconds=60))
    model = object()
    released = []
    provider._model = model
    monkeypatch.setattr(provider, "_release_runtime_memory", lambda: released.append(True))

    provider._schedule_idle_unload()
    stale_generation = provider._idle_generation
    provider._schedule_idle_unload()
    provider._unload_if_idle(stale_generation)

    assert provider._model is model
    assert released == []
    assert provider.unload()
    assert released == [True]


def test_freya_unload_waits_for_active_inference(settings, monkeypatch, tmp_path) -> None:
    provider = FreyaSynthesizer(replace(settings, freya_idle_seconds=0))
    inference_started = Event()
    inference_can_finish = Event()
    released = []

    def synthesize(*args, **kwargs):
        inference_started.set()
        assert inference_can_finish.wait(timeout=1)
        return b"audio"

    provider._model = SimpleNamespace(synthesize=synthesize, save_wav=lambda *args: None)
    monkeypatch.setattr(provider, "_release_runtime_memory", lambda: released.append(True))

    with ThreadPoolExecutor(max_workers=2) as pool:
        synthesis = pool.submit(provider.synthesize, "Merhaba", tmp_path / "active.wav", "tr")
        assert inference_started.wait(timeout=1)
        unloading = pool.submit(provider.unload)
        time.sleep(0.02)
        assert not unloading.done()
        inference_can_finish.set()
        synthesis.result(timeout=1)
        assert unloading.result(timeout=1)

    assert released == [True]


@pytest.fixture
def macos_commands(monkeypatch):
    commands = []
    inventory = (
        "Samantha            en_US    # Hello!\n"
        "Yelda               tr_TR    # Merhaba!\n"
        "Eddy (English (UK)) en_GB    # Hello!\n"
    )

    def run(command, **kwargs):
        commands.append(command)
        output = inventory if command == ["say", "-v", "?"] else ""
        return CompletedProcess(command, 0, output, "")

    monkeypatch.setattr("aicomv1.providers.tts_macos.shutil.which", lambda name: name)
    monkeypatch.setattr("aicomv1.providers.tts_macos.subprocess.run", run)
    return commands


@pytest.mark.parametrize(("language", "voice"), [("en_US", "Samantha"), ("tr-TR", "Yelda")])
def test_macos_uses_installed_voice_for_selected_language(
    settings, macos_commands, tmp_path, language, voice
) -> None:
    provider = MacOSSynthesizer(settings)
    status = provider.status(language)
    assert status.ready
    assert voice in status.detail
    provider.synthesize("Test", tmp_path / "test.wav", language)
    synthesis = next(command for command in macos_commands if "-o" in command)
    assert synthesis[synthesis.index("-v") + 1] == voice
    assert macos_commands.count(["say", "-v", "?"]) == 1


def test_macos_supports_installed_voice_names_with_spaces(settings, macos_commands) -> None:
    provider = MacOSSynthesizer(replace(settings, tts_voice_en="Eddy (English (UK))"))
    assert provider.status("en").ready


@pytest.mark.parametrize("voice", ["Not Installed", "Yelda"])
def test_macos_rejects_missing_or_wrong_language_voice(
    settings, macos_commands, tmp_path, voice
) -> None:
    provider = MacOSSynthesizer(replace(settings, tts_voice_en=voice))
    assert not provider.status("en").ready
    with pytest.raises(TTSError):
        provider.synthesize("Hello", tmp_path / "test.wav", "en")
    assert not any("-o" in command for command in macos_commands)


@pytest.mark.parametrize("provider_type", [MacOSSynthesizer, FreyaSynthesizer, AutoSynthesizer])
def test_tts_rejects_unsupported_locale(settings, provider_type) -> None:
    with pytest.raises(ValueError, match="Language must be en or tr"):
        provider_type(settings).status("de-DE")
