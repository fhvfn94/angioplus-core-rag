# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import ABC, abstractmethod

from app.llm.errors import LLMError
from app.llm.models import GateResult


class LLMProvider(ABC):
    """Provider-agnostic interface for text generation.

    Implementations translate provider-specific errors into a single
    `LLMError` type hierarchy. `app/main.py` is responsible for converting
    `LLMError` into the HTTP 200 / HTTP 500 contract.
    """

    @abstractmethod
    def generate_answer(
        self,
        question: str,
        context: str,
        system_prompt: str,
    ) -> str:
        """Generate the final grounded answer.

        Raises `LLMError` on any failure (quota/auth/timeout/network/5xx).
        """
        ...

    @abstractmethod
    def check_direct_answer(
        self,
        question: str,
        context: str,
    ) -> GateResult:
        """Decide whether the retrieved context directly answers the question.

        On success returns a strict `GateResult`. On any failure
        (API/auth/quota/timeout/network, invalid JSON or Pydantic validation)
        raises `LLMError`.
        """
        ...
