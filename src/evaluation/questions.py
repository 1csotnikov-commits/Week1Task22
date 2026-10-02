"""Контрольные вопросы: загрузка/сохранение/валидация questions.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

import yaml


@dataclass
class Question:
    """Один контрольный вопрос."""

    id: str
    question: str
    expected_answer: str
    expected_sources: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "question": self.question,
            "expected_answer": self.expected_answer,
        }
        if self.expected_sources:
            d["expected_sources"] = self.expected_sources
        if self.notes:
            d["notes"] = self.notes
        return d


def load_questions(path: Union[str, Path]) -> List[Question]:
    """Загрузить и провалидировать questions.yaml."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Файл вопросов не найден: {p}")

    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if isinstance(data, dict):
        items = data.get("questions", [])
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"Некорректный формат {p}: ожидался список вопросов.")

    questions: List[Question] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Вопрос #{i + 1} должен быть словарём (map), получено: {item!r}")
        q = Question(
            id=str(item.get("id") or f"q{i + 1:02d}"),
            question=str(item.get("question", "")).strip(),
            expected_answer=str(item.get("expected_answer", "")).strip(),
            expected_sources=list(item.get("expected_sources") or []),
            notes=str(item.get("notes") or ""),
        )
        if not q.question:
            raise ValueError(f"Вопрос #{i + 1}: поле 'question' обязательно.")
        if not q.expected_answer:
            raise ValueError(f"Вопрос {q.id}: поле 'expected_answer' обязательно.")
        questions.append(q)
    return questions


def save_questions(questions: List[Question], path: Union[str, Path]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {"questions": [q.to_dict() for q in questions]}
    p.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return p
