"""Абстракция поиска (retrieval)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from src.retrieval.models import RetrievedChunk


class Retriever(ABC):
    """Интерфейс поиска релевантных чанков.

    Позволяет позже вставить reranker между retrieval и построением промта.
    """

    name: str = "base"

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """Найти top_k наиболее релевантных чанков по запросу."""
        raise NotImplementedError
