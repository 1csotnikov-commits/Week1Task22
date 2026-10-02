"""Прогон контрольных вопросов в двух режимах (RAG и No-RAG)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from src.evaluation.comparator import aggregate, evaluate_question
from src.evaluation.questions import Question
from src.evaluation.report import render_report
from src.rag.models import RAGResult


@dataclass
class QuestionResult:
    """Результат одного вопроса в обоих режимах."""

    question: Question
    rag: RAGResult
    no_rag: RAGResult


def run_evaluation(questions: List[Question], rag_pipeline, no_rag_pipeline, top_k: Optional[int] = None) -> List[QuestionResult]:
    results = []
    for q in questions:
        rag = rag_pipeline.run(q.question, top_k=top_k)
        no_rag = no_rag_pipeline.run(q.question)
        results.append(QuestionResult(question=q, rag=rag, no_rag=no_rag))
    return results


def build_payload(results: List[QuestionResult], index_name: Optional[str] = None, top_k: Optional[int] = None) -> dict:
    rows = [evaluate_question(r) for r in results]
    return {
        "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "index_name": index_name,
        "top_k": top_k,
        "aggregate": aggregate(rows),
        "questions": rows,
    }


def save_results(
    results: List[QuestionResult],
    results_dir: Union[str, Path],
    index_name: Optional[str] = None,
    top_k: Optional[int] = None,
):
    """Сохранить результаты в JSON и Markdown. Возвращает (json_path, md_path, payload)."""
    payload = build_payload(results, index_name, top_k)
    d = Path(results_dir)
    d.mkdir(parents=True, exist_ok=True)
    ts = payload["timestamp"]
    json_path = d / f"{ts}.json"
    md_path = d / f"{ts}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_report(payload, "markdown"), encoding="utf-8")
    return json_path, md_path, payload


def load_payload(path: Union[str, Path]) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
