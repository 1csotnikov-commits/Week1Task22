"""RAG — сборка ответа с контекстом (реализация Дня 22)."""

from __future__ import annotations

from src.rag.models import RAGResult
from src.rag.no_rag_pipeline import NoRAGPipeline
from src.rag.pipeline import RAGPipeline

__all__ = ["RAGResult", "RAGPipeline", "NoRAGPipeline"]
