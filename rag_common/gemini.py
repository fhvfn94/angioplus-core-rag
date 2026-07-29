# -*- coding: utf-8 -*-
"""Gemini embedding helper for the CLI scripts (google-generativeai SDK)."""
from __future__ import annotations

from rag_common.config import DEFAULT_GEMINI_EMBEDDING_MODEL


def embed_query(
    api_key: str,
    question: str,
    model_name: str = DEFAULT_GEMINI_EMBEDDING_MODEL,
) -> list[float]:
    """Embed a query with Gemini for retrieval (``retrieval_query`` task)."""
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise RuntimeError("Missing dependency: google-generativeai") from exc

    genai.configure(api_key=api_key)
    response = genai.embed_content(
        model=model_name,
        content=question,
        task_type="retrieval_query",
    )
    return response["embedding"]
