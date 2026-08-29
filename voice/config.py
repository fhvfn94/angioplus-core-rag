# -*- coding: utf-8 -*-
"""Environment configuration for the LiveKit voice component (Phase A).

Only the RAG API boundary configuration lives here for now. No secrets are
loaded on purpose: LiveKit / TTS keys belong in `.env` / `.env.local` outside
Git and are added in later phases.

The existing RAG pipeline is the single source of truth: this config only
tells the voice component WHERE that RAG API lives.

Phase D2.3 adds the Qwen3-TTS HTTP boundary configuration. The TTS model is
NOT loaded locally: it runs on a separate GPU machine (e.g. a Vast.ai RTX
4090). This config only tells the local voice worker WHERE that TTS HTTP
service lives and which speaker/language to request. The service URL is never
hardcoded; it is read from the environment (TTS_BASE_URL).
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

# ---------------------------------------------------------------------------
# Qwen3-TTS HTTP boundary (Phase D2.3). The TTS model runs remotely on a GPU;
# the local worker only needs the HTTP endpoint + request parameters.
# ---------------------------------------------------------------------------

# TTS_BASE_URL is NOT set by default -> TTS is DISABLED until the operator
# provides the remote HTTP service URL. This keeps the existing text/data
# answer path (Phase D2.1) fully operational when no GPU TTS is reachable.
DEFAULT_TTS_BASE_URL: str | None = None
DEFAULT_TTS_SPEAKER = "Vivian"
DEFAULT_TTS_LANGUAGE = "Russian"
# Qwen3-TTS on a rented GPU (bf16, non-streaming) is fast for a single answer,
# but a cold/contended instance could still be slow. Generous default.
DEFAULT_TTS_TIMEOUT_SECONDS = 120
# LiveKit convention for audio sources is 48 kHz mono. The local TTS client
# resamples the model's native rate to this before publishing.
DEFAULT_TTS_SAMPLE_RATE = 48000


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


class TTSConfig:
    """Resolved Qwen3-TTS HTTP boundary configuration for the voice worker.

    ``enabled`` is False (and the voice worker skips TTS entirely) when no
    ``TTS_BASE_URL`` is provided, so the existing text/data-only answer path
    keeps working.
    """

    __slots__ = (
        "base_url",
        "speaker",
        "language",
        "timeout_seconds",
        "sample_rate",
    )

    def __init__(
        self,
        base_url: str | None,
        speaker: str,
        language: str,
        timeout_seconds: int,
        sample_rate: int,
    ) -> None:
        self.base_url = base_url
        self.speaker = speaker
        self.language = language
        self.timeout_seconds = timeout_seconds
        self.sample_rate = sample_rate

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"TTSConfig(base_url={self.base_url!r}, speaker={self.speaker!r}, "
            f"language={self.language!r}, timeout_seconds={self.timeout_seconds}, "
            f"sample_rate={self.sample_rate}, enabled={self.enabled})"
        )


def load_tts_config() -> TTSConfig:
    """Build an immutable Qwen3-TTS HTTP boundary config from the environment.

    Only ``TTS_BASE_URL`` (e.g. ``http://<vast-gpu-ip>:8001``) is required to
    enable TTS. speaker/language/timeout default to sensible values and can be
    overridden by env vars. The Vast IP is NEVER hardcoded here.
    """
    base_url = (os.getenv("TTS_BASE_URL", "").strip().rstrip("/")) or None

    speaker = (os.getenv("TTS_SPEAKER", "").strip() or DEFAULT_TTS_SPEAKER)
    language = (os.getenv("TTS_LANGUAGE", "").strip() or DEFAULT_TTS_LANGUAGE)
    timeout_seconds = _read_int_env(
        "TTS_TIMEOUT_SECONDS", DEFAULT_TTS_TIMEOUT_SECONDS
    )
    sample_rate = _read_int_env("TTS_SAMPLE_RATE", DEFAULT_TTS_SAMPLE_RATE)

    return TTSConfig(
        base_url=base_url,
        speaker=speaker,
        language=language,
        timeout_seconds=timeout_seconds,
        sample_rate=sample_rate,
    )
