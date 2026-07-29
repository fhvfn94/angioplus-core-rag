# -*- coding: utf-8 -*-
"""Shared utilities for the AngioPlus Core RAG project.

These helpers are used by the FastAPI app (``app/``) and the CLI/dev
scripts (``scripts/``) so that retrieval, prompt building and Qdrant
access logic lives in a single place.
"""
from __future__ import annotations

from rag_common.config import (
    DEFAULT_COLLECTION,
    DEFAULT_GEMINI_EMBEDDING_MODEL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_SYSTEM_PROMPT_PATH,
    FALLBACK_SYSTEM_PROMPT,
)
from rag_common.prompts import build_context_block, load_system_prompt
from rag_common.retrieval import (
    build_context,
    iter_unique_chunks,
    search_qdrant,
    source_key,
)

__all__ = [
    "DEFAULT_COLLECTION",
    "DEFAULT_GEMINI_EMBEDDING_MODEL",
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_SYSTEM_PROMPT_PATH",
    "FALLBACK_SYSTEM_PROMPT",
    "build_context",
    "build_context_block",
    "iter_unique_chunks",
    "load_system_prompt",
    "search_qdrant",
    "source_key",
]
