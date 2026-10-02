"""Тесты retrieval (VectorRetriever)."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embeddings.base import EMBEDDING_FORMAT_VERSION, SEARCH_QUERY_PREFIX  # noqa: E402
from src.retrieval.vector_retriever import VectorRetriever  # noqa: E402
from tests.fakes import FakeEmbedder, build_temp_index  # noqa: E402


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_search_returns_top_k_with_metadata_and_score(self):
        store = build_temp_index(self.root, name="idx", n_chunks=5, dim=8)
        embedder = FakeEmbedder(dim=8)
        retriever = VectorRetriever(store=store, embedder=embedder, index_name="idx")

        results = retriever.search("тестовый запрос", top_k=3)

        self.assertEqual(len(results), 3)
        self.assertEqual([r.rank for r in results], [1, 2, 3])
        for r in results:
            self.assertIn(r.chunk_id, {f"c{i}" for i in range(5)})
            self.assertTrue(r.source)
            self.assertIsInstance(r.score, float)

    def test_query_is_prefixed(self):
        store = build_temp_index(self.root, name="idx", n_chunks=3, dim=8)
        embedder = FakeEmbedder(dim=8)
        retriever = VectorRetriever(store=store, embedder=embedder, index_name="idx")

        retriever.search("Как дела?", top_k=2)

        self.assertTrue(any(call.startswith(SEARCH_QUERY_PREFIX) for call in embedder.calls))
        self.assertIn(SEARCH_QUERY_PREFIX + "Как дела?", embedder.calls)

    def test_search_missing_index_raises(self):
        store = build_temp_index(self.root, name="idx", n_chunks=2, dim=8)
        retriever = VectorRetriever(store=store, embedder=FakeEmbedder(dim=8))
        with self.assertRaises(ValueError):
            retriever.search("вопрос")


if __name__ == "__main__":
    unittest.main()
