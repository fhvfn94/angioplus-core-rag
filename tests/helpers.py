# -*- coding: utf-8 -*-
"""Small factories shared between tests."""
from __future__ import annotations

from types import SimpleNamespace


def make_point(
    *,
    score: float | None = 0.9,
    text: str = "chunk text",
    file_name: str | None = "IFU.pdf",
    section: str | None = "1 Введение",
    page_start: int | None = 1,
    page_end: int | None = 2,
    payload: dict | None = None,
) -> SimpleNamespace:
    """Builds an object shaped like a Qdrant scored point."""
    if payload is None:
        payload = {
            "text": text,
            "file_name": file_name,
            "section": section,
            "page_start": page_start,
            "page_end": page_end,
        }
    return SimpleNamespace(score=score, payload=payload)
