# -*- coding: utf-8 -*-
"""Environment configuration for the LiveKit voice component (Phase A).

Only the RAG API boundary configuration lives here for now. No secrets are
loaded on purpose: LiveKit / TTS keys belong in `.env` / `.env.local` outside
Git and are added in later phases.

The existing RAG pipeline is the single source of truth: this config only
tells the voice component WHERE that RAG API lives.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Local development default: the `rag` FastAPI service is published on
# localhost in docker-compose (127.0.0.1:8000:8000).
DEFAULT_RAG_API_URL = "http://127.0.0.1:8000/ask"

# Generous default matching the real RAG latency (embedding + Qdrant + gate +
# DeepSeek generation can take several seconds).
DEFAULT_RAG_TIMEOUT_SECONDS = 180


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid %s=%r, using default %s", name, raw, default
        )
        return default
    if value <= 0:
        logger.warning(
            "Invalid %s=%r (must be positive), using default %s",
            name,
            raw,
            default,
        )
        return default
    return value


class RAGConfig:
    """Resolved RAG API endpoint + timeout for the voice component."""

    __slots__ = ("url", "timeout_seconds")

    def __init__(self, url: str, timeout_seconds: int) -> None:
        self.url = url
        self.timeout_seconds = timeout_seconds

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"RAGConfig(url={self.url!r}, "
            f"timeout_seconds={self.timeout_seconds})"
        )


def load_rag_config() -> RAGConfig:
    """Build an immutable RAG boundary config from the environment."""
    url = (os.getenv("RAG_API_URL", "").strip() or DEFAULT_RAG_API_URL).rstrip(
        "/"
    )

    # Ensure the endpoint path is present. If only a scheme://host[:port]
    # is supplied, append the /ask endpoint.
    if not url.endswith("/ask"):
        url = f"{url}/ask"

    timeout_seconds = _read_int_env(
        "RAG_TIMEOUT_SECONDS", DEFAULT_RAG_TIMEOUT_SECONDS
    )

    return RAGConfig(url=url, timeout_seconds=timeout_seconds)
