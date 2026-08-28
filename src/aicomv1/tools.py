from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from aicomv1.knowledge import KnowledgeStore

_MATH_REQUEST = re.compile(
    r"(?:hesapla|kaç\s+eder|sonucu\s+ne|işlemin\s+sonucu|çarpı|bölü|artı|eksi|"
    r"calculate|what\s+is|result|times|divided\s+by|plus|minus)",
    re.I,
)
_TIME_REQUEST = re.compile(
    r"\b(?:saat\s+kaç|bugün\s+hangi\s+gün|tarih\s+ne|what\s+time|"
    r"what\s+day|what(?:'s|\s+is)\s+the\s+date)\b",
    re.I,
)
_EXPRESSION = re.compile(r"[\d\s.,()+\-*/%^]{3,}")
_SPOKEN_OPERATORS = (
    (re.compile(r"\b(?:multiplied\s+by|times|çarpı)\b", re.I), "*"),
    (re.compile(r"\b(?:divided\s+by|bölü)\b", re.I), "/"),
    (re.compile(r"\b(?:plus|artı)\b", re.I), "+"),
    (re.compile(r"\b(?:minus|eksi)\b", re.I), "-"),
)
_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class UnsafeExpressionError(ValueError):
    pass


def safe_calculate(expression: str) -> int | float:
    normalized = expression.replace(",", ".").replace("^", "**")
    if len(normalized) > 120:
        raise UnsafeExpressionError("Expression is too long.")
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError("Invalid mathematical expression.") from exc

    def evaluate(node: ast.AST, depth: int = 0) -> int | float:
        if depth > 12:
            raise UnsafeExpressionError("Expression is too complex.")
        if isinstance(node, ast.Expression):
            return evaluate(node.body, depth + 1)
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left = evaluate(node.left, depth + 1)
            right = evaluate(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise UnsafeExpressionError("Exponent limit exceeded.")
            result = _BINARY[type(node.op)](left, right)
            if abs(result) > 1e18:
                raise UnsafeExpressionError("Result limit exceeded.")
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](evaluate(node.operand, depth + 1))
        raise UnsafeExpressionError("Only basic mathematical operations are supported.")

    return evaluate(tree)


@dataclass(frozen=True, slots=True)
class ToolContext:
    items: tuple[str, ...]

    @property
    def empty(self) -> bool:
        return not self.items

    def render(self, *, language: str = "en") -> str:
        if not self.items:
            return ""
        heading = (
            "Yerel araç ve bilgi sonuçları:"
            if language == "tr"
            else "Local tool and knowledge results:"
        )
        return heading + "\n" + "\n\n".join(self.items)


class LocalToolRouter:
    def __init__(self, knowledge: KnowledgeStore, *, timezone: str = "Europe/Istanbul") -> None:
        self.knowledge = knowledge
        self.timezone = ZoneInfo(timezone)

    def context_for(self, user_text: str, *, language: str = "en") -> ToolContext:
        items: list[str] = []
        if _TIME_REQUEST.search(user_text):
            now = datetime.now(self.timezone)
            label = "Yerel saat" if language == "tr" else "Local time"
            items.append(f"{label}: {now:%d.%m.%Y %H:%M:%S} ({self.timezone.key}).")

        if _MATH_REQUEST.search(user_text):
            math_text = user_text
            for pattern, symbol in _SPOKEN_OPERATORS:
                math_text = pattern.sub(symbol, math_text)
            for expression in sorted(_EXPRESSION.findall(math_text), key=len, reverse=True):
                expression = expression.strip().rstrip(".,")
                if not re.search(r"[+\-*/%^]", expression):
                    continue
                if len(re.findall(r"\d+(?:[.,]\d+)?", expression)) < 2:
                    continue
                try:
                    result = safe_calculate(expression)
                except (UnsafeExpressionError, ZeroDivisionError, OverflowError):
                    pass
                else:
                    label = "Hesap makinesi" if language == "tr" else "Calculator"
                    items.append(f"{label}: {expression} = {result}")
                    break

        for hit in self.knowledge.search(user_text, limit=3):
            excerpt = hit.body[:900]
            label = "Yerel kaynak" if language == "tr" else "Local source"
            items.append(f"{label} [{hit.source}] — {hit.title}: {excerpt}")
        return ToolContext(tuple(items))
