"""Оценка качества RAG (контрольные вопросы + эвристики сравнения)."""

from __future__ import annotations

from src.evaluation.comparator import aggregate, evaluate_question
from src.evaluation.generator import generate_questions, sample_chunks
from src.evaluation.questions import Question, load_questions, save_questions
from src.evaluation.report import render_report, render_report_file
from src.evaluation.runner import QuestionResult, build_payload, load_payload, run_evaluation, save_results

__all__ = [
    "Question",
    "load_questions",
    "save_questions",
    "generate_questions",
    "sample_chunks",
    "evaluate_question",
    "aggregate",
    "QuestionResult",
    "run_evaluation",
    "save_results",
    "build_payload",
    "load_payload",
    "render_report",
    "render_report_file",
]
