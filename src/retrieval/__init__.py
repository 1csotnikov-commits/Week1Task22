"""Retrieval — поиск релевантных чанков (реализация Дня 22)."""

from __future__ import annotations

from src.retrieval.base import Retriever
from src.retrieval.models import RetrievedChunk
from src.retrieval.vector_retriever import VectorRetriever

__all__ = ["Retriever", "RetrievedChunk", "VectorRetriever"]
