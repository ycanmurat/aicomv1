import pytest

from aicomv1.prompt import normalize_language, summary_prompt, system_prompt


@pytest.mark.parametrize(
    ("value", "expected"),
    [("en", "en"), ("tr", "tr"), ("en-US", "en"), ("tr_TR", "tr"), (" EN ", "en")],
)
def test_language_normalization(value: str, expected: str) -> None:
    assert normalize_language(value) == expected


@pytest.mark.parametrize("value", ["", "de", "auto", "english", "None"])
def test_unsupported_language_is_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="Language must be en or tr"):
        normalize_language(value)


@pytest.mark.parametrize(("code", "name"), [("en", "English"), ("tr", "Turkish")])
def test_prompts_follow_selected_language(code: str, name: str) -> None:
    assert f"selected conversation language is {name}" in system_prompt(code)
    assert f"Reply naturally in {name}" in system_prompt(code)
    assert f"memory note in\n{name}" in summary_prompt(code)
