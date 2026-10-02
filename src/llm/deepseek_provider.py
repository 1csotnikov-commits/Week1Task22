"""Реализация LLMProvider для DeepSeek (deepseek-chat).

Ключ API берётся из переменной окружения DEEPSEEK_API_KEY (в коде не хранится).
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

from src.llm.base import LLMProvider, LLMResponse

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"
API_KEY_ENV = "DEEPSEEK_API_KEY"


class DeepSeekProvider(LLMProvider):
    """LLM-провайдер DeepSeek (OpenAI-совместимый API)."""

    name = "deepseek"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ):
        self.api_key = api_key or os.environ.get(API_KEY_ENV, "")
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _client(self):
        if not self.api_key:
            raise RuntimeError(
                "Не задан ключ DeepSeek API. Установите переменную окружения DEEPSEEK_API_KEY. "
                "Пример (PowerShell): $env:DEEPSEEK_API_KEY=\"sk-...\""
            )
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError("Пакет openai не установлен. Выполните: pip install openai") from exc
        return OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> LLMResponse:
        client = self._client()
        started = time.time()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                timeout=self.timeout if timeout is None else timeout,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Ошибка вызова DeepSeek API: {exc}") from exc

        elapsed_ms = int((time.time() - started) * 1000)
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            text=(choice.message.content or "").strip(),
            model=self.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            elapsed_ms=elapsed_ms,
        )
