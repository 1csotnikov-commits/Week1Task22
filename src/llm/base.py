"""Абстракция LLM-провайдера."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class LLMResponse:
    """Ответ LLM с метаданными вызова."""

    text: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    elapsed_ms: int = 0


class LLMProvider(ABC):
    """Интерфейс LLM-провайдера.

    Реализации принимают список сообщений (роль + текст) и возвращают LLMResponse.
    """

    name: str = "base"

    @abstractmethod
    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> LLMResponse:
        """Отправить запрос к LLM и вернуть ответ."""
        raise NotImplementedError
