from __future__ import annotations

import re

_MARKDOWN_LINK = re.compile(r"\[([^\]]+)]\([^\s)]+\)")
_BARE_URL = re.compile(r"https?://\S+")
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]+)`")
_MARKDOWN_PREFIX = re.compile(r"(?m)^\s{0,3}(?:#{1,6}|[-*+]\s|\d+[.)]\s)\s*")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?:[\"'”’)]*)\s+")


def clean_spoken_text(text: str, *, language: str = "en") -> str:
    text = _CODE_FENCE.sub(" ", text)
    text = _MARKDOWN_LINK.sub(r"\1", text)
    text = _BARE_URL.sub(" bağlantı " if language == "tr" else " link ", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _MARKDOWN_PREFIX.sub("", text)
    text = text.replace("**", "").replace("__", "").replace("~~", "")
    return " ".join(text.split()).strip()


class ClauseSegmenter:
    """Split streaming model text into complete, speech-friendly clauses."""

    def __init__(
        self, *, soft_limit: int = 150, hard_limit: int = 240, language: str = "en"
    ) -> None:
        self.soft_limit = soft_limit
        self.hard_limit = hard_limit
        self.language = language
        self.buffer = ""

    def push(self, delta: str) -> list[str]:
        self.buffer += delta
        return self._drain(final=False)

    def finish(self) -> list[str]:
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> list[str]:
        ready: list[str] = []
        while self.buffer.strip():
            match = _SENTENCE_BOUNDARY.search(self.buffer)
            if match and match.start() <= self.hard_limit:
                candidate = clean_spoken_text(
                    self.buffer[: match.start()], language=self.language
                )
                self.buffer = self.buffer[match.end() :]
                if candidate:
                    ready.append(candidate)
                continue

            cut = self._soft_cut()
            if cut is not None:
                candidate = clean_spoken_text(self.buffer[:cut], language=self.language)
                self.buffer = self.buffer[cut:]
                if candidate:
                    ready.append(candidate)
                continue

            if final:
                candidate = clean_spoken_text(self.buffer, language=self.language)
                self.buffer = ""
                if candidate:
                    ready.append(candidate)
            break
        return ready

    def _soft_cut(self) -> int | None:
        if len(self.buffer) < self.soft_limit:
            return None
        search_end = min(len(self.buffer), self.hard_limit)
        for separator in ("; ", ": ", ", ", " — ", " "):
            index = self.buffer.rfind(separator, self.soft_limit, search_end)
            if index != -1:
                return index + len(separator)
        return self.hard_limit if len(self.buffer) >= self.hard_limit else None
