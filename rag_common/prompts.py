# -*- coding: utf-8 -*-
"""System prompt loading and RAG prompt assembly."""
from __future__ import annotations

from rag_common.config import DEFAULT_SYSTEM_PROMPT_PATH, FALLBACK_SYSTEM_PROMPT


def load_system_prompt(path: str = DEFAULT_SYSTEM_PROMPT_PATH) -> str:
    """Read the system prompt from ``path`` or return a safe fallback."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        return FALLBACK_SYSTEM_PROMPT


def build_context_block(context: str, question: str) -> str:
    """Build the shared retrieved-context / user-question block."""
    return f"""
--------------------
RETRIEVED CONTEXT / НАЙДЕННЫЙ КОНТЕКСТ
--------------------

{context}

--------------------
USER QUESTION / ВОПРОС ПОЛЬЗОВАТЕЛЯ
--------------------

{question}
""".strip()
