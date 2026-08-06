# -*- coding: utf-8 -*-
from __future__ import annotations

import os

from app.llm.base import LLMProvider
from app.llm.errors import LLMError, LLMErrorType
from app.llm.gemini_provider import DEFAULT_GEMINI_MODEL, GeminiProvider
from app.llm.gigachat_provider import GigaChatProvider
from app.llm.deepseek_provider import DeepSeekProvider


def build_llm() -> LLMProvider:
    """Build an LLM provider from the environment.

    Supports LLM_PROVIDER=gemini, gigachat and deepseek. This factory
    performs no network calls; it only constructs the in-memory provider
    handle (the SDK clients are created lazily inside each provider).
    """
    provider = (os.getenv("LLM_PROVIDER") or "gemini").strip().lower()

    if provider == "gemini":
        api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
        if not api_key:
            raise LLMError(
                LLMErrorType.AUTHENTICATION,
                "GEMINI_API_KEY is not set",
            )
        model = os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        return GeminiProvider(api_key=api_key, model=model)

    if provider == "gigachat":
        credentials = (os.getenv("GIGACHAT_CREDENTIALS") or "").strip()
        if not credentials:
            raise LLMError(
                LLMErrorType.AUTHENTICATION,
                "GIGACHAT_CREDENTIALS is not set",
            )
        model = (os.getenv("GIGACHAT_MODEL") or "").strip()
        if not model:
            raise LLMError(
                LLMErrorType.UNEXPECTED_INTERNAL,
                "LLM configuration error",
            )
        return GigaChatProvider(credentials=credentials, model=model)

    if provider == "deepseek":
        api_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
        if not api_key:
            raise LLMError(
                LLMErrorType.AUTHENTICATION,
                "DEEPSEEK_API_KEY is not set",
            )
        model = (os.getenv("DEEPSEEK_MODEL") or "").strip()
        if not model:
            raise LLMError(
                LLMErrorType.UNEXPECTED_INTERNAL,
                "LLM configuration error",
            )
        return DeepSeekProvider(api_key=api_key, model=model)

    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. Supported: gemini, gigachat, deepseek"
    )

