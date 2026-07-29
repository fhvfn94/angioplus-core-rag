# -*- coding: utf-8 -*-
"""Shared configuration defaults for the RAG pipeline."""
from __future__ import annotations

import os

# Qdrant collection created by scripts/ingest_documents.py.
DEFAULT_COLLECTION = "angioplus_documents"

# Gemini embedding model. Must match the model used during ingestion.
DEFAULT_GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"

# Gemini generation model used to produce the final answer.
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"

# Location of the system prompt used by the answerer.
DEFAULT_SYSTEM_PROMPT_PATH = os.getenv(
    "SYSTEM_PROMPT_PATH",
    "app/prompts/system_prompt.md",
)

# Fallback used when the system prompt file is missing.
FALLBACK_SYSTEM_PROMPT = (
    "Ты — технический ассистент поддержки AngioPlus Core. "
    "Отвечай строго на основе контекста."
)
