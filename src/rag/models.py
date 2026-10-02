"""Типы данных RAG-пайплайна."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.retrieval.models import RetrievedChunk


@dataclass
class RAGResult:
    """Результат RAG-запроса (или запроса без RAG)."""

    answer: str
    mode: str  # "rag" | "no_rag"
    used_chunks: List[RetrievedChunk] = field(default_factory=list)
    prompt: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
