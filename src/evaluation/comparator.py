"""Эвристики сравнения ответов с ожиданиями (без LLM-as-judge).

Это грубая автоматическая оценка: пересечение ключевых слов, подстрока, Жаккар.
Точное сравнение пользователь делает вручную.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

STOP_WORDS = set(
    "и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее кажется сейчас были куда зачем сказать всех".split()
)

_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+")


def tokenize(text: str) -> List[str]:
    return [w for w in _TOKEN_RE.findall(text.lower()) if w not in STOP_WORDS and len(w) > 1]


def keyword_overlap(answer: str, expected: str) -> float:
    a = set(tokenize(answer))
    e = set(tokenize(expected))
    if not e:
        return 0.0
    return len(a & e) / len(e)


def jaccard(answer: str, expected: str) -> float:
    a = set(tokenize(answer))
    e = set(tokenize(expected))
    if not (a | e):
        return 0.0
    return len(a & e) / len(a | e)


def substring_match(answer: str, expected: str) -> bool:
    return bool(expected.strip()) and expected.strip().lower() in answer.lower()


def score_answer(answer: str, expected: str) -> Dict[str, Any]:
    return {
        "keyword_overlap": round(keyword_overlap(answer, expected), 4),
        "jaccard": round(jaccard(answer, expected), 4),
        "substring_match": substring_match(answer, expected),
    }


def is_hit(score: Dict[str, Any]) -> bool:
    """Считается ли ответ «попавшим» в ожидание (грубо)."""
    return bool(score.get("substring_match")) or float(score.get("keyword_overlap", 0.0)) >= 0.5


def source_hit(used_chunks: List, expected_sources: List[str]) -> Dict[str, Any]:
    """Проверить попадание использованных чанков в ожидаемые источники."""
    used: set = set()
    for c in used_chunks:
        used.add(c.source)
        if c.section:
            used.add(f"{c.source}#{c.section}")
    expected = set(expected_sources or [])
    if not expected:
        return {"expected_sources": [], "matched": [], "hit": None}
    matched = sorted(used & expected)
    return {"expected_sources": sorted(expected), "matched": matched, "hit": bool(matched)}


def evaluate_question(qresult) -> Dict[str, Any]:
    """Собрать строку оценки по одному вопросу (оба режима)."""
    q = qresult.question
    rag_score = score_answer(qresult.rag.answer, q.expected_answer)
    no_rag_score = score_answer(qresult.no_rag.answer, q.expected_answer)
    rag_src = source_hit(qresult.rag.used_chunks, q.expected_sources)
    return {
        "id": q.id,
        "question": q.question,
        "expected_answer": q.expected_answer,
        "expected_sources": q.expected_sources,
        "rag": {
            "answer": qresult.rag.answer,
            "score": rag_score,
            "hit": is_hit(rag_score),
            "sources": rag_src,
            "time_ms": qresult.rag.meta.get("elapsed_ms", 0),
            "tokens_in": qresult.rag.meta.get("tokens_in", 0),
            "tokens_out": qresult.rag.meta.get("tokens_out", 0),
            "used_chunks": [
                {"chunk_id": c.chunk_id, "source": c.source, "section": c.section, "score": round(c.score, 4)}
                for c in qresult.rag.used_chunks
            ],
        },
        "no_rag": {
            "answer": qresult.no_rag.answer,
            "score": no_rag_score,
            "hit": is_hit(no_rag_score),
            "time_ms": qresult.no_rag.meta.get("elapsed_ms", 0),
            "tokens_in": qresult.no_rag.meta.get("tokens_in", 0),
            "tokens_out": qresult.no_rag.meta.get("tokens_out", 0),
        },
    }


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if not n:
        return {
            "num_questions": 0,
            "rag_hit_rate": 0.0,
            "no_rag_hit_rate": 0.0,
            "source_hit_rate": 0.0,
            "avg_rag_time_ms": 0.0,
            "avg_no_rag_time_ms": 0.0,
            "avg_rag_tokens_in": 0.0,
            "avg_no_rag_tokens_in": 0.0,
        }
    rag_hits = sum(1 for r in rows if r["rag"]["hit"])
    no_rag_hits = sum(1 for r in rows if r["no_rag"]["hit"])
    src_hit = sum(1 for r in rows if r["rag"]["sources"]["hit"] is True)
    src_applicable = sum(1 for r in rows if r["rag"]["sources"]["hit"] is not None)

    return {
        "num_questions": n,
        "rag_hit_rate": round(rag_hits / n, 4),
        "no_rag_hit_rate": round(no_rag_hits / n, 4),
        "source_hit_rate": round(src_hit / src_applicable, 4) if src_applicable else 0.0,
        "avg_rag_time_ms": round(sum(r["rag"]["time_ms"] for r in rows) / n, 1),
        "avg_no_rag_time_ms": round(sum(r["no_rag"]["time_ms"] for r in rows) / n, 1),
        "avg_rag_tokens_in": round(sum(r["rag"]["tokens_in"] for r in rows) / n, 1),
        "avg_no_rag_tokens_in": round(sum(r["no_rag"]["tokens_in"] for r in rows) / n, 1),
    }
