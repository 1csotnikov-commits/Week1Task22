"""Типы данных для retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from src.models import Chunk


@dataclass
class RetrievedChunk:
    """Найденный чанк: сам чанк + оценка релевантности + ранг."""

    chunk: Chunk
    score: float
    rank: int

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def source(self) -> str:
        return self.chunk.source

    @property
    def title(self) -> str:
        return self.chunk.title

    @property
    def section(self) -> str:
        return self.chunk.section

    @property
    def strategy(self) -> str:
        return self.chunk.strategy
