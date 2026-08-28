from __future__ import annotations

import json
from collections.abc import AsyncIterator
from threading import Event

import httpx

from aicomv1.config import Settings
from aicomv1.models import ComponentStatus
from aicomv1.providers.base import LLMError


class OllamaChatProvider:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._timeout = httpx.Timeout(connect=3, read=180, write=20, pool=5)

    def _payload(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        stream: bool,
        reasoning: bool,
        max_tokens: int | None = None,
    ) -> dict[str, object]:
        return {
            "model": self.settings.llm_model,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
            "stream": stream,
            "think": reasoning,
            "keep_alive": "15m",
            "options": {
                "num_ctx": self.settings.llm_context,
                "num_predict": max_tokens or self.settings.llm_max_tokens,
                "temperature": self.settings.llm_temperature,
            },
        }

    async def stream(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        cancel: Event,
        reasoning: bool = False,
    ) -> AsyncIterator[str]:
        payload = self._payload(
            system_prompt=system_prompt,
            messages=messages,
            stream=True,
            reasoning=reasoning,
        )
        try:
            async with (
                httpx.AsyncClient(timeout=self._timeout) as client,
                client.stream(
                    "POST", f"{self.settings.ollama_url}/api/chat", json=payload
                ) as response,
            ):
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if cancel.is_set():
                        return
                    if not line:
                        continue
                    item = json.loads(line)
                    if error := item.get("error"):
                        raise LLMError(str(error))
                    content = item.get("message", {}).get("content", "")
                    if content:
                        yield str(content)
                    if item.get("done"):
                        return
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise LLMError(f"Ollama yanıt akışı başarısız: {exc}") from exc

    async def complete(
        self, *, system_prompt: str, messages: list[dict[str, str]], max_tokens: int
    ) -> str:
        payload = self._payload(
            system_prompt=system_prompt,
            messages=messages,
            stream=False,
            reasoning=False,
            max_tokens=max_tokens,
        )
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(f"{self.settings.ollama_url}/api/chat", json=payload)
                response.raise_for_status()
                item = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise LLMError(f"Ollama tamamlayıcı yanıtı başarısız: {exc}") from exc
        if error := item.get("error"):
            raise LLMError(str(error))
        return str(item.get("message", {}).get("content", "")).strip()

    async def status(self) -> ComponentStatus:
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                version_response = await client.get(f"{self.settings.ollama_url}/api/version")
                version_response.raise_for_status()
                tags_response = await client.get(f"{self.settings.ollama_url}/api/tags")
                tags_response.raise_for_status()
            version = str(version_response.json().get("version", "bilinmiyor"))
            names = {str(model.get("name", "")) for model in tags_response.json().get("models", [])}
            ready = self.settings.llm_model in names
            detail = (
                f"Ollama {version}; {self.settings.llm_model} hazır."
                if ready
                else f"Ollama {version}; {self.settings.llm_model} henüz indirilmemiş."
            )
            return ComponentStatus("llm", ready, detail)
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            return ComponentStatus("llm", False, f"Ollama erişilemiyor: {exc}")
