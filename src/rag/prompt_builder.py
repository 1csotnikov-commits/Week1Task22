"""Сборка промта из вопроса и найденных чанков.

Шаблон легко редактируется. Чанки нумеруются ([1], [2], ...), чтобы на Дне 24
добавить обязательные цитаты и ссылки на источники.
"""

from __future__ import annotations

from typing import Dict, List

from src.retrieval.models import RetrievedChunk

RAG_SYSTEM_PROMPT = (
    "Ты — полезный ассистент. Отвечай на вопросы пользователя на русском языке, "
    "используя ТОЛЬКО предоставленный ниже контекст. "
    "Если в контексте недостаточно информации для ответа, честно скажи об этом и не выдумывай факты."
)

NO_RAG_SYSTEM_PROMPT = (
    "Ты — полезный ассистент. Отвечай на вопросы пользователя на русском языке."
)


def format_chunk(idx: int, rc: RetrievedChunk) -> str:
    """Отформатировать один чанк с метаданными."""
    meta = [f"Источник: {rc.source}"]
    if rc.section:
        meta.append(f"Раздел: {rc.section}")
    meta.append(f"chunk_id: {rc.chunk_id}")
    return f"[{idx}] {' | '.join(meta)}\n{rc.text}"


def build_context(chunks: List[RetrievedChunk]) -> str:
    if not chunks:
        return "(контекст отсутствует)"
    return "\n\n".join(format_chunk(i + 1, c) for i, c in enumerate(chunks))


def build_messages(question: str, chunks: List[RetrievedChunk]) -> List[Dict[str, str]]:
    """Собрать сообщения (system + user) для RAG-запроса."""
    user_content = f"Контекст:\n\n{build_context(chunks)}\n\nВопрос: {question}"
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_no_rag_messages(question: str) -> List[Dict[str, str]]:
    """Собрать сообщения для запроса без RAG (без контекста)."""
    return [
        {"role": "system", "content": NO_RAG_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def render_messages(messages: List[Dict[str, str]]) -> str:
    """Склеить сообщения в единый текст (для отладки)."""
    return "\n\n".join(f"--- {m['role']} ---\n{m['content']}" for m in messages)
