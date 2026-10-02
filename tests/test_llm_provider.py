"""Тесты LLM-провайдера (DeepSeek)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.deepseek_provider import DeepSeekProvider  # noqa: E402


class TestLLMProvider(unittest.TestCase):
    def test_missing_api_key_raises_clear_error(self):
        provider = DeepSeekProvider(api_key="")
        with self.assertRaises(RuntimeError) as ctx:
            provider.complete([{"role": "user", "content": "привет"}])
        self.assertIn("DEEPSEEK_API_KEY", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
