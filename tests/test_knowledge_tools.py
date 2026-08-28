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
