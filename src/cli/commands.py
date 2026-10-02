"""Единый источник описаний команд для /help и парсера.

Все описания, usage-строки и допустимые значения флагов задаются здесь,
чтобы /help и разбор команд не расходились.
"""

from __future__ import annotations

from typing import Any, Dict, List

STRATEGY_CHOICES = ("fixed", "structural")
FORMAT_CHOICES = ("console", "json", "csv", "markdown")
REPORT_FORMAT_CHOICES = ("console", "markdown", "json")

# Структура команд. Используется и для /help, и для валидации в парсере.
COMMANDS: Dict[str, Dict[str, Any]] = {
    "/help": {
        "description": "Показать справку по всем командам или по конкретной команде",
        "usage": "/help [command]",
        "example": "/help /index build",
        "arguments": [
            {"name": "command", "required": False, "description": "команда, для которой показать справку"},
        ],
    },
    "/index": {
        "description": "Управление индексами документов",
        "usage": "/index <subcommand> [options]",
        "example": "/index build data/documents --strategy structural --name my_index",
        "subcommands": {
            "build": {
                "description": "Построить индекс из документов (загрузка -> чанкинг -> эмбеддинги -> FAISS+SQLite)",
                "usage": "/index build <path> [--strategy fixed|structural] [--name <index_name>] [--config <path>]",
                "example": "/index build data/documents --strategy fixed --name my_index",
                "arguments": [
                    {"name": "path", "required": False, "description": "путь к файлу/папке с документами (по умолчанию из конфига)"},
                ],
                "flags": [
                    {"name": "strategy", "choices": STRATEGY_CHOICES, "description": "стратегия чанкинга"},
                    {"name": "name", "description": "имя индекса (по умолчанию генерируется)"},
                    {"name": "config", "description": "путь к YAML-конфигу"},
                ],
            },
            "list": {
                "description": "Показать список доступных индексов",
                "usage": "/index list",
                "example": "/index list",
            },
            "info": {
                "description": "Показать метаданные индекса",
                "usage": "/index info <index_name>",
                "example": "/index info my_index",
                "arguments": [{"name": "index_name", "required": True, "description": "имя индекса"}],
            },
            "stats": {
                "description": "Вывести статистику по индексу",
                "usage": "/index stats <index_name> [--format console|json|csv|markdown]",
                "example": "/index stats my_index --format markdown",
                "arguments": [{"name": "index_name", "required": True, "description": "имя индекса"}],
                "flags": [{"name": "format", "choices": FORMAT_CHOICES, "description": "формат вывода"}],
            },
            "compare": {
                "description": "Сравнить статистику двух индексов (стратегий)",
                "usage": "/index compare <index1> <index2> [--format console|json|csv|markdown]",
                "example": "/index compare fixed_idx structural_idx --format markdown",
                "arguments": [
                    {"name": "index1", "required": True, "description": "первый индекс"},
                    {"name": "index2", "required": True, "description": "второй индекс"},
                ],
                "flags": [{"name": "format", "choices": FORMAT_CHOICES, "description": "формат вывода"}],
            },
            "show-chunk": {
                "description": "Показать полный текст и метаданные чанка",
                "usage": "/index show-chunk <index_name> <chunk_id>",
                "example": "/index show-chunk my_index my_index-structural-0000",
                "arguments": [
                    {"name": "index_name", "required": True, "description": "имя индекса"},
                    {"name": "chunk_id", "required": True, "description": "идентификатор чанка"},
                ],
            },
            "list-chunks": {
                "description": "Список чанков с фильтрами",
                "usage": "/index list-chunks <index_name> [--source <source>] [--section <section>] [--limit N]",
                "example": "/index list-chunks my_index --source intro.txt --limit 10",
                "arguments": [{"name": "index_name", "required": True, "description": "имя индекса"}],
                "flags": [
                    {"name": "source", "description": "фильтр по source"},
                    {"name": "section", "description": "фильтр по section"},
                    {"name": "limit", "description": "максимальное число чанков"},
                ],
            },
            "delete": {
                "description": "Удалить индекс",
                "usage": "/index delete <index_name>",
                "example": "/index delete my_index",
                "arguments": [{"name": "index_name", "required": True, "description": "имя индекса"}],
            },
        },
    },
    "/ask": {
        "description": "Задать вопрос и получить RAG-ответ с источниками",
        "usage": "/ask \"<вопрос>\" [--index <index_name>] [--top-k N] [--debug]",
        "example": "/ask \"Какие стратегии чанкинга есть?\" --index my_index",
        "arguments": [{"name": "question", "required": True, "description": "текст вопроса (в кавычках)"}],
        "flags": [
            {"name": "index", "description": "имя индекса (по умолчанию из конфига rag.default_index)"},
            {"name": "top-k", "description": "число чанков для контекста"},
            {"name": "debug", "description": "показать полный промт и все использованные чанки"},
        ],
    },
    "/ask-raw": {
        "description": "Ответ LLM без RAG (без контекста из индекса)",
        "usage": "/ask-raw \"<вопрос>\" [--debug]",
        "example": "/ask-raw \"Что такое эмбеддинг?\"",
        "arguments": [{"name": "question", "required": True, "description": "текст вопроса (в кавычках)"}],
        "flags": [{"name": "debug", "description": "показать полный промт"}],
    },
    "/ask-compare": {
        "description": "Показать оба режима (RAG и No-RAG) рядом",
        "usage": "/ask-compare \"<вопрос>\" [--index <index_name>] [--top-k N]",
        "example": "/ask-compare \"Какая модель эмбеддингов используется?\" --index my_index",
        "arguments": [{"name": "question", "required": True, "description": "текст вопроса (в кавычках)"}],
        "flags": [
            {"name": "index", "description": "имя индекса (по умолчанию из конфига rag.default_index)"},
            {"name": "top-k", "description": "число чанков для контекста"},
        ],
    },
    "/eval": {
        "description": "Оценка качества: генерация вопросов, прогон, отчёт",
        "usage": "/eval <subcommand> [options]",
        "example": "/eval run my_index",
        "subcommands": {
            "generate": {
                "description": "LLM-генерация черновика контрольных вопросов на основе индекса",
                "usage": "/eval generate [--index <index_name>] [--count N] [--out <path>] [--force]",
                "example": "/eval generate --index my_index --count 10",
                "flags": [
                    {"name": "index", "description": "имя индекса (по умолчанию из конфига rag.default_index)"},
                    {"name": "count", "description": "число вопросов (по умолчанию 10)"},
                    {"name": "out", "description": "путь для сохранения (по умолчанию из конфига evaluation.questions)"},
                    {"name": "force", "description": "перезаписать существующий файл"},
                ],
            },
            "run": {
                "description": "Прогнать вопросы в режимах RAG и No-RAG и сохранить результаты",
                "usage": "/eval run <index_name> [--questions <path>] [--top-k N]",
                "example": "/eval run my_index --top-k 5",
                "arguments": [{"name": "index_name", "required": True, "description": "имя индекса"}],
                "flags": [
                    {"name": "questions", "description": "путь к questions.yaml (по умолчанию из конфига)"},
                    {"name": "top-k", "description": "число чанков для контекста"},
                ],
            },
            "report": {
                "description": "Сравнительный отчёт по результатам прогона",
                "usage": "/eval report <result_file> [--format console|markdown|json]",
                "example": "/eval report data/eval/results/20260101-120000.json",
                "arguments": [{"name": "result_file", "required": True, "description": "путь к .json результата"}],
                "flags": [{"name": "format", "choices": REPORT_FORMAT_CHOICES, "description": "формат вывода"}],
            },
        },
    },
    "/debug": {
        "description": "Включить/выключить отладочный режим (показывает эмбеддинги и стектрейсы ошибок)",
        "usage": "/debug on|off",
        "example": "/debug on",
        "arguments": [{"name": "mode", "required": True, "description": "on или off"}],
    },
    "/exit": {
        "description": "Завершить работу",
        "usage": "/exit",
        "example": "/exit",
    },
}


def list_index_subcommands() -> List[str]:
    return list(COMMANDS["/index"]["subcommands"].keys())


def _examples_block() -> List[str]:
    """Собрать список примеров для каждой команды (для общей справки)."""
    lines: List[str] = []
    for name, spec in COMMANDS.items():
        if "subcommands" in spec:
            for sub, subspec in spec["subcommands"].items():
                example = subspec.get("example")
                if example:
                    lines.append(f"  {name} {sub:<14} {example}")
        else:
            example = spec.get("example")
            if example:
                lines.append(f"  {name:<18} {example}")
    return lines


def _render_command_help(name: str, spec: Dict[str, Any]) -> str:
    lines = [f"{name} — {spec['description']}"]
    lines.append(f"Использование: {spec['usage']}")
    if "subcommands" in spec:
        lines.append("")
        lines.append("Подкоманды:")
        for sub, subspec in spec["subcommands"].items():
            lines.append(f"  {subspec['usage']}")
            lines.append(f"      {subspec['description']}")
        lines.append("")
        lines.append("Примеры:")
        for sub, subspec in spec["subcommands"].items():
            example = subspec.get("example")
            if example:
                lines.append(f"  {name} {sub:<12} {example}")
    else:
        for arg in spec.get("arguments", []):
            req = "обязательный" if arg.get("required") else "опциональный"
            lines.append(f"  {arg['name']}: {arg.get('description', '')} ({req})")
        for flag in spec.get("flags", []):
            choices = f" [{', '.join(flag['choices'])}]" if "choices" in flag else ""
            lines.append(f"  --{flag['name']}{choices}: {flag.get('description', '')}")
        example = spec.get("example")
        if example:
            lines.append(f"Пример: {example}")
    return "\n".join(lines)


def build_help(command: str = None) -> str:
    """Сформировать текст справки из COMMANDS (единый источник описаний)."""
    if command is None:
        lines = ["Доступные команды:", ""]
        for name in COMMANDS:
            spec = COMMANDS[name]
            lines.append(f"  {name:<20} {spec['description']}")
            lines.append(f"  {'':<20} {spec['usage']}")
        for name in COMMANDS:
            spec = COMMANDS[name]
            if "subcommands" in spec:
                lines.append("")
                lines.append(f"Подкоманды {name}:")
                for sub in spec["subcommands"]:
                    lines.append(f"  {name} {sub:<14} {spec['subcommands'][sub]['description']}")
        lines.append("")
        lines.append("Примеры использования:")
        lines.extend(_examples_block())
        return "\n".join(lines)

    # Точное имя команды.
    if command in COMMANDS:
        return _render_command_help(command, COMMANDS[command])

    # Команда с подкомандой: "/index build", "/eval run".
    for name, spec in COMMANDS.items():
        subs = spec.get("subcommands", {})
        if not command.startswith(name + " "):
            continue
        sub = command.split(" ", 1)[1].strip()
        if sub in subs:
            return _render_command_help(f"{name} {sub}", subs[sub])
        return f"Неизвестная подкоманда {name}: {sub!r}\n" + build_help(name)

    return f"Неизвестная команда: {command!r}\n" + build_help()
