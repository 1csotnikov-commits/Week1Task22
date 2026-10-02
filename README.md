# Week1Task22 — Первый RAG-запрос

Расширение проекта Week1Task21: локальный пайплайн индексации + RAG-агент.

Папка проекта — `C:\II\Week 1 Task 21` (продолжение Task 21, публикация в отдельный репозиторий Week1Task22).

Возможности:

1. Индексация документов (Task 21): загрузка, чанкинг (`fixed_size` / `structural`), эмбеддинги `nomic-ai/nomic-embed-text-v2-moe`, FAISS + SQLite, статистика;
2. RAG-запрос (Task 22): вопрос → поиск релевантных чанков → промт → DeepSeek → ответ;
3. Два режима: ответ LLM **с RAG** и **без RAG**;
4. Контрольные вопросы (`questions.yaml`) и сравнение качества (эвристики, без LLM-as-judge).

Интерфейс — только CLI (интерактивный REPL), без web. PDF не поддерживаются.

## Стек

- ОС: Windows Server 2022 (совместимо с Linux/macOS)
- Python 3.10+
- `sentence-transformers` + `einops` (эмбеддинги)
- `faiss-cpu` (векторный поиск, с fallback на `numpy`)
- SQLite (метаданные и текст чанков)

## Архитектура

```
C:\II\Week 1 Task 21\
├── config/
│   └── config.yaml          # конфигурация (embedding, chunking, index, llm, rag, evaluation)
├── data/
│   ├── documents/           # пользователь кладёт сюда документы
│   ├── index/               # сгенерированные индексы (faiss.index + metadata.db)
│   ├── eval/                # контрольные вопросы (questions.yaml) и результаты
│   └── logs/                # логи RAG-запросов (опционально)
├── src/
│   ├── ingestion/           # DocumentLoader: text_loader (.md/.txt/.py/.js/.json)
│   ├── chunking/            # BaseChunker, fixed_size, structural
│   ├── embeddings/          # EmbeddingProvider, nomic_provider
│   ├── indexing/            # IndexStore, faiss_sqlite_store, pipeline
│   ├── stats/               # chunk_stats (метрики + форматы console/json/csv/markdown)
│   ├── llm/                 # LLMProvider, deepseek_provider
│   ├── retrieval/           # Retriever, VectorRetriever, RetrievedChunk
│   ├── reranking/           # заглушка (задел на День 23)
│   ├── rag/                 # RAGPipeline, NoRAGPipeline, prompt_builder, citations (заглушка)
│   ├── evaluation/          # вопросы, генерация, прогон, эвристики, отчёты
│   ├── chat/                # заглушка (задел на День 25)
│   └── cli/                 # main.py (REPL), commands.py (единый источник описаний)
├── tests/                   # unittest-тесты
├── requirements.txt
└── README.md
```

## Установка

```bash
pip install -r requirements.txt
```

или вручную (основные зависимости):

```bash
pip install "sentence-transformers>=3.0,<4.0" "transformers>=4.40,<5.0" einops
```

> **Важно:** при первом запуске модель `nomic-ai/nomic-embed-text-v2-moe`
> скачается автоматически (**~1–2 ГБ**). Убедитесь, что есть доступ в интернет.

> **Совместимость версий:** не ставьте `sentence-transformers 6.x` / `transformers 5.x` —
> кастомный код модели nomic (`modeling_hf_nomic_bert.py`) несовместим с transformers 5.x
> (отсутствует метод `get_extended_attention_mask`). Используйте версии из `requirements.txt`.

### Windows Server 2022: возможные проблемы

- **`faiss-cpu` не устанавливается** — это не критично: при ошибке импорта
  автоматически включается fallback на `numpy` + перебор (для 20–30 страниц
  скорость приемлема). В консоли появится предупреждение.
- **`OSError: [WinError 1114] ... c10.dll`** при импорте `torch` — обычно
  означает отсутствие Microsoft Visual C++ Redistributable. Установите
  актуальный **VC++ Redistributable (x64)**, например:
  `https://aka.ms/vs/17/release/vc_redist.x64.exe` (тихая установка:
  `vc_redist.x64.exe /install /quiet /norestart`).

## Модель эмбеддингов

Используется только локальная модель:

```
nomic-ai/nomic-embed-text-v2-moe
```

- 768 измерений;
- поддержка русского языка;
- лицензия Apache 2.0;
- нормализованные векторы (L2-норма = 1), поэтому `IndexFlatIP`
  (inner product) эквивалентен косинусной близости.

## Запуск CLI

```bash
python -m src.cli.main
```

Откроется интерактивная консоль. Список команд — `/help`.

### Команды

| Команда | Описание |
| --- | --- |
| `/help [command]` | Справка по командам (описания — из единого источника `commands.py`) |
| `/index build <path> [--strategy fixed\|structural] [--name <name>] [--config <path>]` | Построить индекс |
| `/index list` | Список индексов |
| `/index info <name>` | Метаданные индекса |
| `/index stats <name> [--format console\|json\|csv\|markdown]` | Статистика по индексу |
| `/index compare <a> <b> [--format ...]` | Сравнить два индекса |
| `/index show-chunk <name> <chunk_id>` | Полный текст + метаданные чанка |
| `/index list-chunks <name> [--source ...] [--section ...] [--limit N]` | Список чанков с фильтрами |
| `/index delete <name>` | Удалить индекс |
| `/ask "<вопрос>" [--index <name>] [--top-k N] [--debug]` | RAG-ответ + источники |
| `/ask-raw "<вопрос>" [--debug]` | Ответ LLM без RAG |
| `/ask-compare "<вопрос>" [--index <name>] [--top-k N]` | Оба режима рядом |
| `/eval generate [--index <name>] [--count N] [--out <path>] [--force]` | Генерация контрольных вопросов |
| `/eval run <index_name> [--questions <path>] [--top-k N]` | Прогон вопросов в двух режимах |
| `/eval report <result_file> [--format console\|markdown\|json]` | Сравнительный отчёт |
| `/debug on\|off` | Отладочный режим (показывает эмбеддинги и стектрейсы) |
| `/exit` | Выход |

### Пример использования

```bash
python -m src.cli.main
> /index build data/documents --strategy fixed --name fixed_idx
> /index build data/documents --strategy structural --name structural_idx
> /index list
> /index stats fixed_idx
> /index stats fixed_idx --format markdown
> /index compare fixed_idx structural_idx --format console
> /index show-chunk fixed_idx intro-fixed_size-0000
> /index list-chunks structural_idx --section "Введение" --limit 5
> /index delete structural_idx
> /exit
```

## Стратегии чанкинга

### A — `fixed_size`

Параметры: `chunk_size` (по умолчанию 800 символов), `overlap` (150).
Рез — по границам предложений/строк (не рвёт слова). Для кода — по строкам.

### B — `structural`

- **Markdown/README** — по заголовкам `#`, `##`, `###`; крупные разделы дробятся `fixed_size`;
- **Python** — по функциям/классам через AST;
- **Прочий код** — эвристика по отступам / `def` / `class` / blank-строкам;
- **Текст** — по абзацам с эвристикой заголовков (пустые строки, ALL CAPS).

Метаданные каждого чанка: `source`, `title`, `section`, `chunk_id`, `strategy`,
`char_count`, `token_count` (эвристика `len(text) // 4`).

## Статистика для сравнения стратегий

Метрики:

- количество документов, чанков, время индексации;
- размеры чанков: min / max / avg / median + гистограмма (ASCII в консоли);
- метаданные: чанков с `section`, с `title`, число уникальных `source`;
- перекрытие (`fixed_size`): перекрывающихся чанков, средний overlap;
- покрытие: суммарная длина чанков vs длина исходных текстов;
- семантическая целостность: доля чанков, начинающихся/заканчивающихся в середине
  предложения, доля с незакрытыми скобками/кавычками; для `structural` — доля
  чанков с ровно одним разделом vs с несколькими разделами.

### Пример (фрагмент, console)

```
===== Статистика индекса: fixed_idx =====
Стратегия: fixed
Документов: 5
Чанков: 23
Время индексации: 12.34 сек.

--- Размеры чанков (символы) ---
min: 120
max: 812
avg: 640.5
median: 655.0
Гистограмма распределения:
  [   120-    189] #####                                    5
  [   189-    258] ##########                              10
  ...
```

### JSON / CSV / Markdown

```bash
> /index stats fixed_idx --format json
> /index stats fixed_idx --format csv
> /index stats fixed_idx --format markdown
```

Файлы отчётов сохраняются рядом с индексом:
`data/index/<index_name>/stats.json|csv|md`. Сравнение — в
`data/index/compare_<a>_vs_<b>.<ext>`.

## Тесты

```bash
python -m unittest discover -s tests -t .
```

Покрытие: загрузка документов, `fixed_size`, `structural`, эмбеддинги
(размерность 768 и нормализация; пропускается, если модель недоступна),
сохранение/загрузка FAISS + SQLite, статистика, retrieval, RAG-пайплайн,
LLM-провайдер (мок), evaluation, CLI-команды.

## RAG-пайплайн (Task 22)

Первый RAG-запрос: `вопрос → retrieval → prompt → LLM → ответ`.

```
вопрос → retrieval (поиск top-k чанков в FAISS) → сборка промта → DeepSeek → ответ (+ источники)
```

### Формат эмбеддингов

Модель `nomic-embed-text-v2-moe` различает типы задач префиксами:

- при индексации документов: `search_document: <текст чанка>`;
- при поиске (запрос пользователя): `search_query: <запрос>`.

В Task 21 префиксы не применялись. В Task 22 формат приведён к единому:

- `/index build` добавляет префикс `search_document:` при расчёте эмбеддингов;
- retrieval всегда эмбеддит запрос с префиксом `search_query:`;
- формат сохраняется в индексе (поле `embedding_format` в SQLite).

Если индекс создан без префиксов (старый индекс Task 21) — при поиске выводится
предупреждение и предложение переиндексировать через `/index build`.

### LLM-провайдер (DeepSeek)

- Модель `deepseek-chat`, endpoint `https://api.deepseek.com`;
- ключ — из переменной окружения `DEEPSEEK_API_KEY` (в коде не хранится);
- установка ключа (PowerShell): `$env:DEEPSEEK_API_KEY="sk-..."`;
- при отсутствии ключа — понятная ошибка с инструкцией.

### Команды запросов

```bash
> /ask "Какие стратегии чанкинга есть?" --index my_index --top-k 5
> /ask-raw "Что такое эмбеддинг?"
> /ask-compare "Какая модель эмбеддингов используется?" --index my_index
```

`/ask` выводит ответ и список источников (source, section, chunk_id, score).
При `/debug on` (или флаге `--debug`) — полный промт и все использованные чанки.

### Контрольные вопросы (questions.yaml)

Файл `data/eval/questions.yaml` — 10 вопросов с ожиданиями и источниками.

```yaml
questions:
  - id: q01
    question: "Какие две стратегии чанкинга поддерживает пайплайн?"
    expected_answer: "fixed_size и structural"
    expected_sources: ["readme_guide.md"]
    notes: "раздел 'Стратегии чанкинга'"
```

Поля: `id`, `question`, `expected_answer`, `expected_sources` (опционально), `notes` (опционально).

### Генерация черновика вопросов

```bash
> /eval generate --index my_index --count 10
```

LLM получает выборку чанков (разные source/section) и формирует вопросы.
Если `data/eval/questions.yaml` уже существует и нет `--force` — результат пишется
в `data/eval/questions.generated.yaml`, чтобы не перезаписать ручные правки.

### Прогон и сравнение режимов

```bash
> /eval run my_index --top-k 5
> /eval report data/eval/results/20260101-120000.json --format console
```

`/eval run` прогоняет каждый вопрос в двух режимах (RAG и No-RAG) и сохраняет
результаты в `data/eval/results/<timestamp>.json` и `.md`.

### Эвристики сравнения (без LLM-as-judge)

Грубая автоматическая оценка (точное сравнение — вручную):

- **keyword_overlap** — пересечение ключевых слов (без стоп-слов) ответа и ожидания;
- **substring_match** — вхождение `expected_answer` как подстроки в ответе;
- **jaccard** — коэффициент Жаккара по множествам токенов;
- **hit** — «попал», если `substring_match` ИЛИ `keyword_overlap >= 0.5`;
- **source hit** (RAG) — пересечение `source`/`source#section` использованных чанков с `expected_sources`;
- сводка — hit rate RAG/No-RAG, среднее время, средние токены.

## Задел на Дни 23–25

- `src/reranking` — `rerank(chunks, query)` (День 23);
- `src/rag/citations.py` — `extract_citations(answer, chunks)` (День 24);
- `src/chat` — мини-чат с историей и памятью (День 25).

Модули — заглушки с интерфейсами, которые заполняются логикой на следующих этапах.

