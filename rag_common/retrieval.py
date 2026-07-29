# -*- coding: utf-8 -*-
"""Qdrant retrieval helpers shared across the app and scripts."""
from __future__ import annotations

from typing import Any, Iterator

from qdrant_client import QdrantClient

# Payload fields that together identify a unique document source.
SOURCE_FIELDS = ("file_name", "section", "page_start", "page_end")


def search_qdrant(
    client: QdrantClient,
    collection: str,
    vector: list[float],
    limit: int,
) -> list[Any]:
    """Return the top ``limit`` points for ``vector`` from ``collection``.

    Uses ``query_points`` (qdrant-client 1.10+); the legacy ``search`` was
    removed in 1.17.x.
    """
    response = client.query_points(
        collection_name=collection,
        query=vector,
        limit=limit,
        with_payload=True,
    )
    return list(response.points)


def build_context(chunks: list[Any]) -> str:
    """Render retrieved chunks into a numbered context block for the LLM."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        payload = chunk.payload or {}
        text = payload.get("text", "")
        parts.append(f"[Chunk {i}]\n{text}")
    return "\n\n".join(parts)


def source_key(payload: dict) -> tuple:
    """Deduplication key for a chunk payload based on its source fields."""
    return tuple(payload.get(field) for field in SOURCE_FIELDS)


def iter_unique_chunks(chunks: list[Any]) -> Iterator[Any]:
    """Yield chunks in order, skipping ones with a duplicate source key."""
    seen: set[tuple] = set()
    for chunk in chunks:
        key = source_key(chunk.payload or {})
        if key in seen:
            continue
        seen.add(key)
        yield chunk
