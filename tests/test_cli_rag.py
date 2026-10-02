"""Тесты новых CLI-команд RAG (/ask, /ask-raw, /ask-compare, /eval report)."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli import main  # noqa: E402
from src.cli.main import App, handle_command  # noqa: E402
from src.config import Config  # noqa: E402
from tests.fakes import FakeEmbedder, FakeLLM, build_temp_index  # noqa: E402


class TestCLIRag(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.index_root = self.root / "index"
        build_temp_index(self.index_root, name="idx", n_chunks=3, dim=8)

        self.config = Config({
            "index": {"path": str(self.index_root)},
            "documents": {"path": str(self.root / "docs"), "extensions": [".txt"]},
            "rag": {"default_index": "idx", "top_k": 3},
            "llm": {"provider": "deepseek", "model": "deepseek-chat"},
            "evaluation": {
                "questions": str(self.root / "questions.yaml"),
                "results_dir": str(self.root / "results"),
            },
        })
        self.app = App(self.config)

        patcher_llm = mock.patch.object(main, "_make_llm", return_value=FakeLLM(answer="Ответ"))
        patcher_emb = mock.patch.object(main, "_make_embedder", return_value=FakeEmbedder(dim=8))
        patcher_llm.start()
        patcher_emb.start()
        self.addCleanup(patcher_llm.stop)
        self.addCleanup(patcher_emb.stop)

    def tearDown(self):
        self._tmp.cleanup()

    def test_help_lists_new_commands(self):
        out = main.build_help()
        self.assertIn("/ask", out)
        self.assertIn("/ask-raw", out)
        self.assertIn("/ask-compare", out)
        self.assertIn("/eval", out)

    def test_ask(self):
        _, out = handle_command('/ask "Какой вопрос?" --index idx', self.app)
        self.assertIn("--- Ответ (RAG) ---", out)
        self.assertIn("--- Источники ---", out)
        self.assertIn("Ответ", out)

    def test_ask_raw(self):
        _, out = handle_command('/ask-raw "Вопрос"', self.app)
        self.assertIn("--- Ответ (No-RAG) ---", out)

    def test_ask_compare(self):
        _, out = handle_command('/ask-compare "Вопрос" --index idx', self.app)
        self.assertIn("--- Ответ без RAG ---", out)
        self.assertIn("--- Ответ с RAG ---", out)

    def test_eval_report(self):
        from src.evaluation.questions import Question
        from src.evaluation.runner import save_results
        from src.models import Chunk
        from src.rag.models import RAGResult
        from src.retrieval.models import RetrievedChunk

        rc = RetrievedChunk(Chunk("c0", "a.md", "a", "t", 0, 4, "fixed", "s"), 0.9, 1)
        q = Question("q01", "Вопрос", "ответ", ["a.md"])
        rag = RAGResult("ответ", "rag", [rc], "", {"elapsed_ms": 1, "tokens_in": 2, "tokens_out": 1})
        no_rag = RAGResult("не знаю", "no_rag", [], "", {"elapsed_ms": 1, "tokens_in": 1, "tokens_out": 1})
        from src.evaluation.runner import QuestionResult
        json_path, _, _ = save_results([QuestionResult(q, rag, no_rag)], self.root / "results", "idx")

        _, out = handle_command(f'/eval report {json_path.as_posix()} --format console', self.app)
        self.assertIn("Отчёт об оценке", out)


if __name__ == "__main__":
    unittest.main()
