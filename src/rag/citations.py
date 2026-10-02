"""Цитаты и источники — задел на День 24.

Заглушка: на Дне 24 сюда добавляется извлечение цитат и анти-галлюцинации.
"""

from __future__ import annotations

from typing import List

from src.models import Citation


def extract_citations(answer: str, chunks: List) -> List[Citation]:
    """Извлечь цитаты/ссылки на источники из ответа.

    Заглушка для Дня 24 — реализация будет добавлена позже.
    """
    raise NotImplementedError(
        "Извлечение цитат (citations) реализуется на Дне 24."
    )


__all__ = ["extract_citations"]
