"""LLM-генерация черновика контрольных вопросов на основе индекса."""

from __future__ import annotations

import json
import re
from typing import List

from src.evaluation.questions import Question
from src.llm.base import LLMProvider
from src.indexing.faiss_sqlite_store import FAISSSQLiteStore

_GENERATE_SYSTEM = "Ты — ассистент, который составляет контрольные вопросы по базе знаний."


def sample_chunks(store: FAISSSQLiteStore, index_name: str, max_chunks: int = 20) -> list:
    """Выбрать разнообразные чанки (разные source/section) для генерации вопросов."""
    store.load(index_name)
    chunks = store.list_chunks()
    by_source: dict = {}
    for c in chunks:
        by_source.setdefault(c.source, []).append(c)

    sample = []
    for cs in by_source.values():
        seen_sections = set()
        for c in cs:
            key = c.section or ""
            if key in seen_sections:
                continue
            seen_sections.add(key)
            sample.append(c)
            if len(sample) >= max_chunks:
                return sample
    return sample


def generate_questions(store: FAISSSQLiteStore, llm: LLMProvider, index_name: str, count: int = 10) -> List[Question]:
    chunks = sample_chunks(store, index_name, max_chunks=count * 2)
    if not chunks:
        raise ValueError(f"Индекс '{index_name}' пуст — нет чанков для генерации вопросов.")

    context = "\n\n".join(
        f"[{i + 1}] source={c.source} section={c.section or '-'}\n{c.text[:500]}"
        for i, c in enumerate(chunks)
    )
    user = (
        f"На основе следующих фрагментов документов составь ровно {count} контрольных вопросов.\n"
        "Для каждого вопроса укажи поля: id (q01, q02, ...), question, expected_answer "
        "(краткий ответ по смыслу), expected_sources (список строк вида 'файл.md' или 'файл.md#Раздел'), "
        "notes (опционально).\n"
        "Верни ответ ТОЛЬКО как JSON-массив объектов, без пояснений.\n\n"
        f"Фрагменты:\n\n{context}"
    )
    response = llm.complete(
        [{"role": "system", "content": _GENERATE_SYSTEM}, {"role": "user", "content": user}],
        temperature=0.4,
        max_tokens=2000,
    )
    data = _parse_json(response.text)
    questions = []
    for item in data[:count]:
        questions.append(
            Question(
                id=str(item.get("id") or f"q{len(questions) + 1:02d}"),
                question=str(item.get("question", "")).strip(),
                expected_answer=str(item.get("expected_answer", "")).strip(),
                expected_sources=list(item.get("expected_sources") or []),
                notes=str(item.get("notes") or ""),
            )
        )
    return questions


def _parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise ValueError("Не удалось распарсить JSON из ответа LLM.") from None
