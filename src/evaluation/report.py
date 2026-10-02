"""Отрисовка отчётов об оценке (console / markdown / json)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union


def render_report(payload: Dict[str, Any], fmt: str = "console") -> str:
    fmt = (fmt or "console").lower()
    if fmt == "json":
        return json.dumps(payload, ensure_ascii=False, indent=2)
    if fmt == "markdown":
        return _render_markdown(payload)
    if fmt == "console":
        return _render_console(payload)
    raise ValueError(f"Неизвестный формат: {fmt!r}. Доступно: console, markdown, json")


def render_report_file(path: Union[str, Path], fmt: str = "console") -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return render_report(payload, fmt)


def _short(text: str, limit: int = 180) -> str:
    t = " ".join(text.split())
    return t if len(t) <= limit else t[:limit] + "…"


def _render_console(payload: Dict[str, Any]) -> str:
    agg = payload["aggregate"]
    lines = ["===== Отчёт об оценке =====", ""]
    lines.append(f"Индекс: {payload.get('index_name') or '-'}")
    lines.append(f"Время: {payload.get('timestamp') or '-'}")
    lines.append("")
    lines.append("--- Сводка ---")
    lines.append(f"Вопросов: {agg['num_questions']}")
    lines.append(f"Hit rate RAG:    {agg['rag_hit_rate']:.2%}")
    lines.append(f"Hit rate No-RAG: {agg['no_rag_hit_rate']:.2%}")
    lines.append(f"Попадание в источники (RAG): {agg['source_hit_rate']:.2%}")
    lines.append(f"Среднее время RAG / No-RAG: {agg['avg_rag_time_ms']} / {agg['avg_no_rag_time_ms']} мс")
    lines.append(f"Средние токены (in) RAG / No-RAG: {agg['avg_rag_tokens_in']} / {agg['avg_no_rag_tokens_in']}")
    lines.append("")
    for row in payload["questions"]:
        rag = row["rag"]
        no = row["no_rag"]
        lines.append(f"[{row['id']}] {row['question']}")
        lines.append(f"  Ожидание: {row['expected_answer']}")
        lines.append(f"  RAG     ({'попал' if rag['hit'] else 'мимо'}): {_short(rag['answer'])}")
        if row.get("expected_sources"):
            src = rag["sources"]
            lines.append(f"    Источники: {'OK' if src['hit'] else 'мимо'} ({', '.join(src['matched']) or '-'})")
        lines.append(f"  No-RAG  ({'попал' if no['hit'] else 'мимо'}): {_short(no['answer'])}")
        lines.append("")
    return "\n".join(lines)


def _render_markdown(payload: Dict[str, Any]) -> str:
    agg = payload["aggregate"]
    lines = ["# Отчёт об оценке", ""]
    lines.append(f"- Индекс: `{payload.get('index_name') or '-'}`")
    lines.append(f"- Время: {payload.get('timestamp') or '-'}")
    lines.append("")
    lines.append("## Сводка")
    lines.append("| Метрика | Значение |")
    lines.append("| --- | --- |")
    lines.append(f"| Вопросов | {agg['num_questions']} |")
    lines.append(f"| Hit rate RAG | {agg['rag_hit_rate']:.2%} |")
    lines.append(f"| Hit rate No-RAG | {agg['no_rag_hit_rate']:.2%} |")
    lines.append(f"| Попадание в источники (RAG) | {agg['source_hit_rate']:.2%} |")
    lines.append(f"| Среднее время RAG | {agg['avg_rag_time_ms']} мс |")
    lines.append(f"| Среднее время No-RAG | {agg['avg_no_rag_time_ms']} мс |")
    lines.append("")
    for row in payload["questions"]:
        rag = row["rag"]
        no = row["no_rag"]
        lines.append(f"## {row['id']}. {row['question']}")
        lines.append("")
        lines.append(f"**Ожидание:** {row['expected_answer']}")
        if row.get("expected_sources"):
            lines.append(f"**Ожидаемые источники:** {', '.join(row['expected_sources'])}")
        lines.append("")
        lines.append("**RAG** (" + ("попал" if rag["hit"] else "мимо") + "):")
        lines.append("")
        lines.append(f"> {rag['answer']}")
        if row.get("expected_sources"):
            lines.append("")
            lines.append(f"Использованные источники: {', '.join(rag['sources']['matched']) or '-'}")
        lines.append("")
        lines.append("**No-RAG** (" + ("попал" if no["hit"] else "мимо") + "):")
        lines.append("")
        lines.append(f"> {no['answer']}")
        lines.append("")
    return "\n".join(lines)
