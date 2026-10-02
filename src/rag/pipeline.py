"""RAG-пайплайн: вопрос -> retrieval -> prompt -> LLM -> ответ."""

from __future__ import annotations

from typing import Optional

from src.config import Config
from src.llm.base import LLMProvider
from src.rag.models import RAGResult
from src.rag.prompt_builder import build_messages, render_messages
from src.retrieval.base import Retriever


class RAGPipeline:
    """Классический RAG: поиск чанков + ответ с контекстом."""

    def __init__(self, retriever: Retriever, llm: LLMProvider, config: Optional[Config] = None):
        self.retriever = retriever
        self.llm = llm
        self.config = config

    def _top_k(self, top_k: Optional[int]) -> int:
        if top_k is not None:
            return int(top_k)
        if self.config is not None:
            return int(self.config.get("rag.top_k", 5))
        return 5

    def run(self, question: str, top_k: Optional[int] = None) -> RAGResult:
        k = self._top_k(top_k)
        chunks = self.retriever.search(question, k)
        messages = build_messages(question, chunks)
        prompt_text = render_messages(messages)

        response = self.llm.complete(messages)

        return RAGResult(
            answer=response.text,
            mode="rag",
            used_chunks=chunks,
            prompt=prompt_text,
            meta={
                "model": response.model,
                "tokens_in": response.prompt_tokens,
                "tokens_out": response.completion_tokens,
                "elapsed_ms": response.elapsed_ms,
                "top_k": k,
                "num_chunks": len(chunks),
            },
        )
