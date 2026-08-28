import pytest

from aicomv1.text import ClauseSegmenter, clean_spoken_text


def test_clean_spoken_text_removes_markdown_and_urls() -> None:
    raw = "## Merhaba **Murat**. [Kaynak](https://example.com) `kod`"
    assert clean_spoken_text(raw) == "Merhaba Murat. Kaynak kod"


def test_segmenter_emits_complete_sentences_then_tail() -> None:
    segmenter = ClauseSegmenter(soft_limit=40, hard_limit=70)
    assert segmenter.push("İlk cümle tamam. İkinci") == ["İlk cümle tamam."]
    assert segmenter.push(" cümle de tamam!") == []
    assert segmenter.finish() == ["İkinci cümle de tamam!"]


def test_segmenter_soft_cuts_long_spoken_text() -> None:
    segmenter = ClauseSegmenter(soft_limit=12, hard_limit=20)
    parts = segmenter.push("Bu oldukça uzun, devam eden bir metindir")
    parts.extend(segmenter.finish())
    assert " ".join(parts) == "Bu oldukça uzun, devam eden bir metindir"
    assert len(parts) >= 2


@pytest.mark.parametrize(("language", "word"), [("en", "link"), ("tr", "bağlantı")])
def test_bare_url_speech_is_localized(language: str, word: str) -> None:
    assert clean_spoken_text("https://example.com", language=language) == word
    segmenter = ClauseSegmenter(language=language)
    segmenter.push("https://example.com")
    assert segmenter.finish() == [word]
