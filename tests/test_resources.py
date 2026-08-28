from dataclasses import replace

import pytest

from aicomv1.config import Settings
from aicomv1.providers.llm_ollama import OllamaChatProvider


def test_resource_defaults_are_tuned_for_16_gb(monkeypatch) -> None:
    for name in (
        "AICOM_LLM_CONTEXT",
        "AICOM_LLM_KEEP_ALIVE_SECONDS",
        "AICOM_FREYA_IDLE_SECONDS",
        "AICOM_WARMUP",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert settings.llm_context == 6144
    assert settings.llm_keep_alive_seconds == 180
    assert settings.freya_idle_seconds == 120
    assert settings.warmup is False


def test_ollama_payload_uses_configured_resource_limits(settings) -> None:
    configured = replace(settings, llm_context=4096, llm_keep_alive_seconds=37)
    payload = OllamaChatProvider(configured)._payload(
        system_prompt="Test",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        reasoning=False,
    )

    assert payload["keep_alive"] == "37s"
    assert payload["options"]["num_ctx"] == 4096

    default_payload = OllamaChatProvider(
        replace(settings, llm_context=6144, llm_keep_alive_seconds=180)
    )._payload(
        system_prompt="Test",
        messages=[],
        stream=False,
        reasoning=False,
    )
    assert default_payload["keep_alive"] == "180s"
    assert default_payload["options"]["num_ctx"] == 6144


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("llm_keep_alive_seconds", -1, "LLM keep-alive"),
        ("llm_keep_alive_seconds", 3601, "LLM keep-alive"),
        ("freya_idle_seconds", -0.1, "Freya idle timeout"),
        ("freya_idle_seconds", 3601, "Freya idle timeout"),
    ],
)
def test_resource_limits_are_validated(settings, field, value, message) -> None:
    with pytest.raises(ValueError, match=message):
        replace(settings, **{field: value}).validate()
