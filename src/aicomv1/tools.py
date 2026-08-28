from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from aicomv1.knowledge import KnowledgeStore

_MATH_REQUEST = re.compile(
    r"(?:hesapla|kaç\s+eder|sonucu\s+ne|işlemin\s+sonucu|çarpı|bölü|artı|eksi)", re.I
)
_TIME_REQUEST = re.compile(r"\b(?:saat\s+kaç|bugün\s+hangi\s+gün|tarih\s+ne)\b", re.I)
_EXPRESSION = re.compile(r"[\d\s.,()+\-*/%^]{3,}")
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
        raise UnsafeExpressionError("İfade çok uzun.")
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError("Geçersiz matematik ifadesi.") from exc

    def evaluate(node: ast.AST, depth: int = 0) -> int | float:
        if depth > 12:
            raise UnsafeExpressionError("İfade çok karmaşık.")
        if isinstance(node, ast.Expression):
            return evaluate(node.body, depth + 1)
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left = evaluate(node.left, depth + 1)
            right = evaluate(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise UnsafeExpressionError("Üs sınırı aşıldı.")
            result = _BINARY[type(node.op)](left, right)
            if abs(result) > 1e18:
                raise UnsafeExpressionError("Sonuç sınırı aşıldı.")
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](evaluate(node.operand, depth + 1))
        raise UnsafeExpressionError("Yalnız temel matematik işlemleri desteklenir.")

    return evaluate(tree)


@dataclass(frozen=True, slots=True)
class ToolContext:
    items: tuple[str, ...]

    @property
    def empty(self) -> bool:
        return not self.items

    def render(self) -> str:
        if not self.items:
            return ""
        return "Yerel araç ve bilgi sonuçları:\n" + "\n\n".join(self.items)


class LocalToolRouter:
    def __init__(self, knowledge: KnowledgeStore, *, timezone: str = "Europe/Istanbul") -> None:
        self.knowledge = knowledge
        self.timezone = ZoneInfo(timezone)

    def context_for(self, user_text: str) -> ToolContext:
        items: list[str] = []
        if _TIME_REQUEST.search(user_text):
            now = datetime.now(self.timezone)
            items.append(f"Yerel saat: {now:%d.%m.%Y %H:%M:%S} ({self.timezone.key}).")

        if _MATH_REQUEST.search(user_text):
            matches = _EXPRESSION.findall(user_text)
            if matches:
                expression = max(matches, key=len).strip()
                try:
                    result = safe_calculate(expression)
                except (UnsafeExpressionError, ZeroDivisionError, OverflowError):
                    pass
                else:
                    items.append(f"Hesap makinesi: {expression} = {result}")

        for hit in self.knowledge.search(user_text, limit=3):
            excerpt = hit.body[:900]
            items.append(f"Yerel kaynak [{hit.source}] — {hit.title}: {excerpt}")
        return ToolContext(tuple(items))
