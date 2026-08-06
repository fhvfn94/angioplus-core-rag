# -*- coding: utf-8 -*-
"""LLM provider abstraction for the AngioPlus RAG pipeline.

Phase 1: only Gemini is supported via `build_llm()`. GigaChat is planned
for Phase 2 and must not contain any logic here yet.
"""
from app.llm.base import LLMProvider
from app.llm.errors import LLMError, LLMErrorType, is_transient_error
from app.llm.factory import build_llm
from app.llm.models import GateResult

__all__ = [
    "LLMProvider",
    "LLMError",
    "LLMErrorType",
    "GateResult",
    "build_llm",
    "is_transient_error",
]
