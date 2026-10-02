"""Режим без RAG: вопрос отправляется в LLM без контекста."""

from __future__ import annotations

from typing import Optional

from src.config import Config
from src.llm.base import LLMProvider
from src.rag.models import RAGResult
from src.rag.prompt_builder import build_no_rag_messages, render_messages


class NoRAGPipeline:
    """Ответ LLM без поиска по индексу (для сравнения с RAG)."""

    def __init__(self, llm: LLMProvider, config: Optional[Config] = None):
        self.llm = llm
        self.config = config

    def run(self, question: str) -> RAGResult:
        messages = build_no_rag_messages(question)
        prompt_text = render_messages(messages)

        response = self.llm.complete(messages)

        return RAGResult(
            answer=response.text,
            mode="no_rag",
            used_chunks=[],
            prompt=prompt_text,
            meta={
                "model": response.model,
                "tokens_in": response.prompt_tokens,
                "tokens_out": response.completion_tokens,
                "elapsed_ms": response.elapsed_ms,
                "top_k": 0,
                "num_chunks": 0,
            },
        )
