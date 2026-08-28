from pathlib import Path

import pytest

from aicomv1.knowledge import KnowledgeStore
from aicomv1.tools import LocalToolRouter, UnsafeExpressionError, safe_calculate


def test_knowledge_roundtrip_and_context(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.sqlite3")
    document_id = store.add(
        title="AICOM mimarisi",
        body="Ses verisi cihaz dışına çıkmadan yerel olarak işlenir.",
        source="proje",
    )
    assert document_id == 1
    assert store.count() == 1
    hits = store.search("ses verisi nasıl işlenir")
    assert hits[0].title == "AICOM mimarisi"
    context = LocalToolRouter(store).context_for("Ses verisi nasıl işlenir?")
    assert "cihaz dışına çıkmadan" in context.render()


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("2 + 3 * 4", 14), ("10 / 4", 2.5), ("2^5", 32)],
)
def test_safe_calculator(expression: str, expected: float) -> None:
    assert safe_calculate(expression) == expected


def test_safe_calculator_rejects_code() -> None:
    with pytest.raises(UnsafeExpressionError):
        safe_calculate("__import__('os').system('id')")


@pytest.mark.parametrize(
    ("language", "question", "label", "expected"),
    [
        ("en", "Calculate 2 plus 3 times 4.", "Calculator", "14"),
        ("en", "What is 12 divided by 4?", "Calculator", "3.0"),
        ("tr", "2 artı 3 çarpı 4 kaç eder?", "Hesap makinesi", "14"),
        ("tr", "12 bölü 4 kaç eder?", "Hesap makinesi", "3.0"),
    ],
)
def test_calculator_understands_both_languages(
    tmp_path, language, question, label, expected
) -> None:
    router = LocalToolRouter(KnowledgeStore(tmp_path / "knowledge.db"))
    rendered = router.context_for(question, language=language).render(language=language)
    assert label in rendered
    assert f"= {expected}" in rendered


@pytest.mark.parametrize(
    ("language", "question", "label"),
    [("en", "What time is it?", "Local time"), ("tr", "Saat kaç?", "Yerel saat")],
)
def test_clock_understands_both_languages(tmp_path, language, question, label) -> None:
    router = LocalToolRouter(KnowledgeStore(tmp_path / "knowledge.db"))
    assert label in router.context_for(question, language=language).render(language=language)


def test_calculator_does_not_treat_a_year_as_an_expression(tmp_path) -> None:
    router = LocalToolRouter(KnowledgeStore(tmp_path / "knowledge.db"))
    assert router.context_for("What is planned for 2026?").empty
