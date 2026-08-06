# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import os
import re
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from time import perf_counter
from typing import Any

import logging
from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel
from qdrant_client import QdrantClient

from app.conversation_memory import ConversationMemory
from app.domain_terms import distinct_entities
from app.embeddings.sentence_transformer import SentenceTransformerEmbedder
from app.followup import rewrite_followup_question
from app.llm import (
    LLMProvider,
    LLMError,
    LLMErrorType,
    build_llm,
    is_transient_error,
)
from app.query_normalization import normalize_user_query

logger = logging.getLogger("uvicorn.error")

DEFAULT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "angioplus_documents",
)
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
DEFAULT_GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini")
SYSTEM_PROMPT_PATH = os.getenv(
    "SYSTEM_PROMPT_PATH",
    "app/prompts/system_prompt.md",
)


def _read_source_limit() -> int:
    raw = os.getenv("SOURCE_LIMIT", "3")
    try:
        value = int(raw)
        if value >= 1:
            return value
        raise ValueError(f"SOURCE_LIMIT must be >= 1, got {value}")
    except (TypeError, ValueError):
        logger.warning("Invalid SOURCE_LIMIT=%r, using default 3", raw)
        return 3


def _read_source_score_threshold() -> float:
    raw = os.getenv("SOURCE_SCORE_THRESHOLD", "0.0")
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid SOURCE_SCORE_THRESHOLD=%r, using default 0.0", raw
        )
        return 0.0


SOURCE_LIMIT = _read_source_limit()
SOURCE_SCORE_THRESHOLD = _read_source_score_threshold()


# ---------------------------------------------------------------------------
# Query context feature flags (strict boolean parsing)
# ---------------------------------------------------------------------------

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _strict_bool_env(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    if raw:
        logger.warning(
            "Invalid boolean env %s=%r, using default=%s (raw value is not logged)",
            name,
            raw,
            default,
        )
    return default


QUERY_NORMALIZATION_ENABLED = _strict_bool_env(
    "QUERY_NORMALIZATION_ENABLED", True
)
# Conversation memory/rewrite change the meaning of the retrieval query, so it
# must be enabled explicitly only after a smoke test confirms normalization is
# safe in production. Normalization is deterministic and stays enabled.
QUERY_CONTEXT_ENABLED = _strict_bool_env("QUERY_CONTEXT_ENABLED", False)

CONVERSATION_TTL_SECONDS = int(os.getenv("CONVERSATION_TTL_SECONDS", "1800"))
CONVERSATION_MAX_TURNS = int(os.getenv("CONVERSATION_MAX_TURNS", "3"))


def _read_low_score_log_threshold() -> float:
    raw = os.getenv("QUERY_LOW_SCORE_LOG_THRESHOLD", "0.45")
    try:
        value = float(raw)
        if 0.0 <= value <= 1.0:
            return value
        raise ValueError(f"must be in [0.0, 1.0], got {value}")
    except (TypeError, ValueError):
        logger.warning(
            "Invalid QUERY_LOW_SCORE_LOG_THRESHOLD=%r, using default 0.45",
            raw,
        )
        return 0.45


QUERY_LOW_SCORE_LOG_THRESHOLD = _read_low_score_log_threshold()

# Short in-memory conversation store (keyed by conversation_id/user_id).
conversation_memory = ConversationMemory(
    ttl_seconds=CONVERSATION_TTL_SECONDS,
    max_turns=CONVERSATION_MAX_TURNS,
)


def _conversation_id_hash(conversation_id: str | None) -> str:
    if not conversation_id:
        return "none"
    return hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()[:12]


def _hash_text(text: str) -> str:
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _last_standalone(snapshot) -> str | None:
    """Return the last standalone question from the snapshot, if any."""
    if not snapshot.turns:
        return None
    return snapshot.turns[-1].standalone_question


# Readiness state (set during lifespan startup; read by /health and /ask).
_readiness: dict[str, Any] = {
    "status": "starting",
    "embedding_model": None,
    "embedding_dimension": None,
    "last_startup_error": None,
}


def _embedding_provider() -> str:
    return (os.getenv("EMBEDDING_PROVIDER") or DEFAULT_EMBEDDING_PROVIDER).strip().lower()


@asynccontextmanager
async def lifespan(app: FastAPI):
    provider = _embedding_provider()
    _readiness["status"] = "starting"
    _readiness["last_startup_error"] = None

    if provider == "sentence_transformers":
        try:
            started = perf_counter()
            embedder = get_sentence_transformer_embedder()
            elapsed = perf_counter() - started
            dimension = embedder.dimension
            _readiness["embedding_model"] = embedder.model_label
            _readiness["embedding_dimension"] = dimension
            _readiness["status"] = "ready"
            _readiness["last_startup_error"] = None
            logger.info(
                "Local embedding model ready: model=%s device=%s dimension=%s load_time=%.2fs",
                embedder.model_label,
                getattr(embedder, "device", None),
                dimension,
                elapsed,
            )
        except Exception:
            logger.exception(
                "Failed to load local embedding model at startup"
            )
            _readiness["status"] = "error"
            _readiness["last_startup_error"] = (
                "Failed to load local embedding model"
            )
    elif provider == "gemini":
        # Non-local provider: no local model preload needed.
        _readiness["embedding_model"] = None
        _readiness["embedding_dimension"] = None
        _readiness["status"] = "ready"
        _readiness["last_startup_error"] = None
        logger.info(
            "Embedding provider ready (no local model required): provider=%s",
            provider,
        )
    else:
        # Unsupported provider: do not mark ready, keep /health available.
        logger.error(
            "Unsupported embedding provider set: provider=%r",
            provider,
        )
        _readiness["status"] = "error"
        _readiness["last_startup_error"] = "Unsupported embedding provider"

    try:
        yield
    finally:
        pass


app = FastAPI(title="AngioPlus Core RAG Assistant", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    conversation_id: str | None = None
    user_id: str | None = None


class Source(BaseModel):
    file_name: str | None
    section: str | None
    page_start: int | None
    page_end: int | None
    score: float | None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


def get_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set",
        )

    return api_key


def create_gemini_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def embed_query_gemini(question: str) -> list[float]:
    client = create_gemini_client(get_api_key())

    response = client.models.embed_content(
        model=DEFAULT_GEMINI_EMBEDDING_MODEL,
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
        ),
    )

    if not response.embeddings:
        raise RuntimeError("Gemini returned no query embedding")

    embedding = response.embeddings[0]

    if not embedding.values:
        raise RuntimeError("Gemini returned an empty query embedding")

    return list(embedding.values)


@lru_cache(maxsize=1)
def get_sentence_transformer_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()


def embed_query_sentence_transformer(question: str) -> list[float]:
    return get_sentence_transformer_embedder().embed_query(question)


def embed_query(question: str) -> list[float]:
    provider = _embedding_provider()

    if provider == "sentence_transformers":
        return embed_query_sentence_transformer(question)

    if provider != "gemini":
        raise RuntimeError(
            f"Unknown EMBEDDING_PROVIDER {provider!r}. Supported: gemini, sentence_transformers."
        )

    return embed_query_gemini(question)


def search_qdrant(vector: list[float], top_k: int):
    client = QdrantClient(url=DEFAULT_QDRANT_URL)

    response = client.query_points(
        collection_name=DEFAULT_COLLECTION,
        query=vector,
        limit=top_k,
        with_payload=True,
    )

    return list(response.points)


def build_context(chunks: list[Any]) -> str:
    parts = []

    for i, chunk in enumerate(chunks, 1):
        payload = chunk.payload or {}
        text = payload.get("text", "")
        parts.append(f"[Chunk {i}]\n{text}")

    return "\n\n".join(parts)


def load_system_prompt() -> str:
    try:
        with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        return (
            "Ты — технический ассистент поддержки AngioPlus Core. "
            "Отвечай строго на основе контекста."
        )


@lru_cache(maxsize=1)
def get_llm() -> "LLMProvider":
    """Build (once) and return the LLM provider.

    Lazy: only called after secret block, embedding, Qdrant and sanitization.
    Phase 1 supports only LLM_PROVIDER=gemini. GigaChat is planned for Phase 2.
    """
    return build_llm()


# ---------------------------------------------------------------------------
# Safety / grounding helpers
# ---------------------------------------------------------------------------

SECRET_REFUSAL = (
    "Я не могу предоставить учётные данные или секреты. "
    "Обратитесь к уполномоченному администратору или представителю Pulse Medical."
)
NOT_FOUND = "Такой информации нет в имеющейся документации."
REDACTED = "[REDACTED_SECRET]"
LLM_TEMPORARILY_UNAVAILABLE = (
    "Сервис проверки и генерации ответа временно недоступен. "
    "Попробуйте повторить запрос позже."
)

_SECRET_REVEAL_INTENTS = (
    "дай",
    "покажи",
    "скажи",
    "предоставь",
    "раскрой",
    "какой",
    "каков",
    "show",
    "give",
    "reveal",
    "provide",
    "what is",
)

_SECRET_OBJECTS = (
    "пароль",
    "токен",
    "api-ключ",
    "api ключ",
    "api key",
    "лицензионный ключ",
    "ключ активации",
    "credentials",
    "password",
    "token",
    "license key",
)


def question_requests_secret(question: str) -> bool:
    """Return True only for explicit requests to reveal a secret value."""
    normalized = question.casefold()
    has_reveal_intent = any(
        intent in normalized
        for intent in _SECRET_REVEAL_INTENTS
    )
    if not has_reveal_intent:
        return False
    return any(
        secret_object in normalized
        for secret_object in _SECRET_OBJECTS
    )


# Credential-pair sanitization: "login/user/admin ... password ..." or
# "password/password/Пароль: <value>" form.
_CRED_PAIR_RE = re.compile(
    r"(?i)(login|user|admin|пользователь|логин).{0,12}(password|пароль).{0,24}"
)

_PASSWORD_LABEL_RE = re.compile(
    r"(?i)(password|пароль)\s*[:=]\s*\S+"
)
_KNOWN_PASS_RE = re.compile(r"(?i)(pulse2015|pulse2023)")
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]+")
_SK_RE = re.compile(r"\bsk-[A-Za-z0-9]{12,}\b")

# Long token / API-key style strings (only clearly secret-like: high entropy).
_LONG_TOKEN_RE = re.compile(
    r"(?i)(api[_-]?key|token|secret|access[_-]?token)\s*[:=]\s*[A-Za-z0-9_\-\.]{16,}"
)
_KEY_VALUE_RE = re.compile(
    r"(?i)(license[_-]?key|activation[_-]?key|client[_-]?secret)\s*[:=]\s*[A-Za-z0-9_\-]{8,}"
)


def sanitize_secrets(text: str) -> str:
    """Replace sensitive credential-like values with [REDACTED_SECRET].

    Only secrets are removed. IP, port, AE Title, DICOM UID and ordinary
    technical identifiers are intentionally NOT redacted.
    """
    if not text:
        return text

    out = _KNOWN_PASS_RE.sub(REDACTED, text)
    out = _BEARER_RE.sub(REDACTED, out)
    out = _SK_RE.sub(REDACTED, out)
    out = _LONG_TOKEN_RE.sub(REDACTED, out)
    out = _KEY_VALUE_RE.sub(REDACTED, out)
    # "password: <value>" -> keep label, drop value
    out = _PASSWORD_LABEL_RE.sub(r"\1: " + REDACTED, out)
    # credential pairs login ... password ... (value after password label)
    out = _CRED_PAIR_RE.sub(REDACTED, out)
    return out


_OUTPUT_SECRET_PATTERNS = (
    re.compile(r"(?i)pulse2015"),
    re.compile(r"(?i)pulse2023"),
    re.compile(r"(?i)password\s*[:=]\s*(\S+)\b"),
    re.compile(r"(?i)пароль\s*[:=]\s*(\S+)\b"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{4,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{12,}\b"),
    re.compile(r"(?i)(api[_-]?key|token|license[_-]?key|activation[_-]?key)\s*[:=]\s*[A-Za-z0-9_\-\.]{8,}"),
)


def contains_secret(text: str) -> bool:
    """Return True if the generated answer appears to expose a secret."""
    if not text:
        return False
    for pattern in _OUTPUT_SECRET_PATTERNS:
        if pattern.search(text):
            return True
    return False


def build_sources(chunks: list[Any]) -> list[Source]:
    seen = set()
    sources: list[Source] = []

    threshold = SOURCE_SCORE_THRESHOLD
    limit = SOURCE_LIMIT

    for chunk in chunks:
        score = chunk.score if chunk.score is not None else None

        if threshold > 0 and score is None:
            continue
        if threshold > 0 and score < threshold:
            continue

        payload = chunk.payload or {}

        key = (
            payload.get("file_name"),
            payload.get("section"),
            payload.get("page_start"),
            payload.get("page_end"),
        )

        if key in seen:
            continue

        seen.add(key)
        sources.append(
            Source(
                file_name=payload.get("file_name"),
                section=payload.get("section"),
                page_start=payload.get("page_start"),
                page_end=payload.get("page_end"),
                score=round(float(score), 3) if score is not None else None,
            )
        )

        if len(sources) >= limit:
            break

    # Fallback: filtering removed everything -> keep the first chunk.
    if not sources and chunks:
        chunk = chunks[0]
        payload = chunk.payload or {}
        sources.append(
            Source(
                file_name=payload.get("file_name"),
                section=payload.get("section"),
                page_start=payload.get("page_start"),
                page_end=payload.get("page_end"),
                score=(
                    round(float(chunk.score), 3)
                    if chunk.score is not None
                    else None
                ),
            )
        )

    return sources


@app.get("/health")
def health():
    return {
        "status": _readiness["status"],
        "embedding_provider": _embedding_provider(),
        "qdrant_collection": DEFAULT_COLLECTION,
        "embedding_model": _readiness["embedding_model"],
        "embedding_dimension": _readiness["embedding_dimension"],
        "last_startup_error": _readiness["last_startup_error"],
    }


def _context_available(request: AskRequest) -> bool:
    return QUERY_CONTEXT_ENABLED and bool(request.conversation_id and request.user_id)


def _block_secret(
    request_id: str,
    total_started: float,
    top_k: int,
) -> AskResponse:
    """Log and return the secret refusal. Never persists a turn."""
    total_seconds = perf_counter() - total_started
    logger.info(
        "ask %s",
        {
            "event": "ask_secret_blocked",
            "request_id": request_id,
            "embedding_provider": _embedding_provider(),
            "collection": DEFAULT_COLLECTION,
            "embedding_seconds": 0.0000,
            "qdrant_seconds": 0.0000,
            "generation_seconds": 0.0000,
            "total_seconds": round(total_seconds, 4),
            "top_k": top_k,
            "result_count": 0,
            "readiness": _readiness["status"],
            "gate_seconds": 0.0000,
            "gate_result": "blocked",
            "secret_request_blocked": True,
            "secret_filter_triggered": False,
        },
    )
    return AskResponse(answer=SECRET_REFUSAL, sources=[])


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if _readiness["status"] != "ready":
        raise HTTPException(
            status_code=503,
            detail="RAG service is not ready",
        )

    request_id = uuid.uuid4().hex
    t_total = perf_counter()

    # Deterministic normalization first (STT term fixes). Safe even without
    # conversation context and independent of QUERY_CONTEXT_ENABLED.
    if QUERY_NORMALIZATION_ENABLED:
        normalized = normalize_user_query(request.question)
    else:
        normalized = request.question

    normalized = normalized.strip()
    if not normalized:
        raise HTTPException(
            status_code=400,
            detail="Question is empty",
        )

    # Secret-request check #1 on normalized current question. A blocked
    # secret is never persisted, last_standalone is never updated.
    if question_requests_secret(normalized):
        return _block_secret(
            request_id,
            t_total,
            request.top_k,
        )

    # Load conversation context (brief in-memory history) if available.
    context_enabled = _context_available(request)
    last_standalone: str | None = None
    standalone = normalized
    used_history = False

    if context_enabled:
        snapshot = conversation_memory.get_snapshot(
            request.conversation_id,
            request.user_id,
        )
        last_standalone = _last_standalone(snapshot)
        standalone, used_history = rewrite_followup_question(
            normalized,
            last_standalone,
        )
        standalone = standalone.strip()

    # Secret-request check #2 on the standalone question, so that rewrite
    # cannot accidentally produce a secret request. Never persisted.
    if question_requests_secret(standalone):
        return _block_secret(
            request_id,
            t_total,
            request.top_k,
        )

    conversation_id_hash = _conversation_id_hash(request.conversation_id)

    try:
        t_start = perf_counter()
        vector = embed_query(standalone)
        embedding_seconds = perf_counter() - t_start

        t_start = perf_counter()
        chunks = search_qdrant(vector, request.top_k)
        qdrant_seconds = perf_counter() - t_start

        top_score = (
            round(float(chunks[0].score), 4)
            if chunks and chunks[0].score is not None
            else None
        )

        if (
            top_score is not None
            and top_score < QUERY_LOW_SCORE_LOG_THRESHOLD
        ):
            logger.info(
                "low_score %s",
                {
                    "request_id": request_id,
                    "top_score": top_score,
                    "normalized_length": len(normalized),
                    "standalone_length": len(standalone),
                    "conversation_id_hash": conversation_id_hash,
                    "used_history": bool(used_history),
                    "entity_match_found": bool(distinct_entities(standalone)),
                },
            )

        if not chunks:
            total_seconds = perf_counter() - t_total
            logger.info(
                "ask %s",
                {
                    "event": "ask_empty",
                    "request_id": request_id,
                    "embedding_provider": _embedding_provider(),
                    "collection": DEFAULT_COLLECTION,
                    "embedding_seconds": round(embedding_seconds, 4),
                    "qdrant_seconds": round(qdrant_seconds, 4),
                    "generation_seconds": 0.0000,
                    "total_seconds": round(total_seconds, 4),
                    "top_k": request.top_k,
                    "result_count": 0,
                    "readiness": _readiness["status"],
                    "scores": [],
                    "gate_seconds": 0.0000,
                    "gate_result": "n/a",
                    "secret_request_blocked": False,
                    "secret_filter_triggered": False,
                    "context_enabled": bool(context_enabled),
                    "used_history": bool(used_history),
                    "rewritten": bool(used_history),
                    "conversation_id_hash": conversation_id_hash,
                    "original_length": len(request.question),
                    "normalized_length": len(normalized),
                    "standalone_length": len(standalone),
                },
            )
            if context_enabled:
                conversation_memory.append_turn(
                    request.conversation_id,
                    request.user_id,
                    normalized,
                    standalone,
                )
            return AskResponse(
                answer=NOT_FOUND,
                sources=[],
            )

        raw_context = build_context(chunks)
        # Sanitize before sending context to ANY external LLM call.
        context = sanitize_secrets(raw_context)
        sanitized_question = sanitize_secrets(standalone)

        sources = build_sources(chunks)

        # 2) Direct-answer gate (fail closed on error).
        #    The LLM provider is created lazily here, only after secret block,
        #    embedding, Qdrant retrieval and sanitization.
        llm = get_llm()
        t_start = perf_counter()
        try:
            gate = llm.check_direct_answer(sanitized_question, context)
        except LLMError as exc:
            gate_seconds = perf_counter() - t_start
            if is_transient_error(exc.error_type):
                total_seconds = perf_counter() - t_total
                logger.info(
                    "ask %s",
                    {
                        "event": "ask_gate_error",
                        "request_id": request_id,
                        "embedding_provider": _embedding_provider(),
                        "collection": DEFAULT_COLLECTION,
                        "embedding_seconds": round(embedding_seconds, 4),
                        "qdrant_seconds": round(qdrant_seconds, 4),
                        "generation_seconds": 0.0000,
                        "total_seconds": round(total_seconds, 4),
                        "top_k": request.top_k,
                        "result_count": len(chunks),
                        "readiness": _readiness["status"],
                        "gate_seconds": round(gate_seconds, 4),
                        "gate_result": "error",
                        "gate_error": True,
                        "secret_request_blocked": False,
                        "secret_filter_triggered": False,
                        "context_enabled": bool(context_enabled),
                        "used_history": bool(used_history),
                        "rewritten": bool(used_history),
                        "conversation_id_hash": conversation_id_hash,
                    },
                )
                if context_enabled:
                    conversation_memory.append_turn(
                        request.conversation_id,
                        request.user_id,
                        normalized,
                        standalone,
                    )
                return AskResponse(
                    answer=LLM_TEMPORARILY_UNAVAILABLE,
                    sources=sources,
                )
            raise
        gate_seconds = perf_counter() - t_start

        if not gate.direct_answer:
            total_seconds = perf_counter() - t_total
            logger.info(
                "ask %s",
                {
                    "event": "ask_gate_false",
                    "request_id": request_id,
                    "embedding_provider": _embedding_provider(),
                    "collection": DEFAULT_COLLECTION,
                    "embedding_seconds": round(embedding_seconds, 4),
                    "qdrant_seconds": round(qdrant_seconds, 4),
                    "generation_seconds": 0.0000,
                    "total_seconds": round(total_seconds, 4),
                    "top_k": request.top_k,
                    "result_count": len(chunks),
                    "readiness": _readiness["status"],
                    "gate_seconds": round(gate_seconds, 4),
                    "gate_result": False,
                    "gate_error": False,
                    "secret_request_blocked": False,
                    "secret_filter_triggered": False,
                    "context_enabled": bool(context_enabled),
                    "used_history": bool(used_history),
                    "rewritten": bool(used_history),
                    "conversation_id_hash": conversation_id_hash,
                },
            )
            if context_enabled:
                conversation_memory.append_turn(
                    request.conversation_id,
                    request.user_id,
                    normalized,
                    standalone,
                )
            return AskResponse(
                answer=NOT_FOUND,
                sources=sources,
            )

        # 3) Main generation using sanitized context only.
        t_start = perf_counter()
        try:
            answer = llm.generate_answer(
                sanitized_question,
                context,
                load_system_prompt(),
            )
        except LLMError as exc:
            if is_transient_error(exc.error_type):
                # The retrieval-standalone topic is already determined, so we
                # persist a safe standalone turn even on a transient LLM error.
                generation_seconds = perf_counter() - t_start
                status_code = exc.status_code
                total_seconds = perf_counter() - t_total
                logger.info(
                    "ask %s",
                    {
                        "event": "ask_generation_transient_error",
                        "request_id": request_id,
                        "embedding_provider": _embedding_provider(),
                        "collection": DEFAULT_COLLECTION,
                        "error_type": exc.error_type.value,
                        "status_code": status_code,
                        "gate_result": True,
                        "generation_seconds": round(generation_seconds, 4),
                        "total_seconds": round(total_seconds, 4),
                        "result_count": len(chunks),
                        "secret_request_blocked": False,
                        "secret_filter_triggered": False,
                        "context_enabled": bool(context_enabled),
                        "used_history": bool(used_history),
                        "rewritten": bool(used_history),
                        "conversation_id_hash": conversation_id_hash,
                    },
                )
                if context_enabled:
                    conversation_memory.append_turn(
                        request.conversation_id,
                        request.user_id,
                        normalized,
                        standalone,
                    )
                return AskResponse(
                    answer=(
                        "Сервис генерации ответа временно недоступен. "
                        "Попробуйте повторить запрос позже."
                    ),
                    sources=sources,
                )
            raise
        generation_seconds = perf_counter() - t_start

        # 4) Output secret filter (defence-in-depth after sanitized generation).
        secret_filter_triggered = contains_secret(answer)
        if secret_filter_triggered:
            answer = SECRET_REFUSAL

        total_seconds = perf_counter() - t_total
        logger.info(
            "ask %s",
            {
                "event": "ask_success",
                "request_id": request_id,
                "embedding_provider": _embedding_provider(),
                "collection": DEFAULT_COLLECTION,
                "embedding_seconds": round(embedding_seconds, 4),
                "qdrant_seconds": round(qdrant_seconds, 4),
                "generation_seconds": round(generation_seconds, 4),
                "total_seconds": round(total_seconds, 4),
                "top_k": request.top_k,
                "result_count": len(chunks),
                "readiness": _readiness["status"],
                "scores": [
                    round(float(c.score), 4) if c.score is not None else None
                    for c in chunks
                ],
                "gate_seconds": round(gate_seconds, 4),
                "gate_result": True,
                "gate_error": False,
                "secret_request_blocked": False,
                "secret_filter_triggered": bool(secret_filter_triggered),
                "context_enabled": bool(context_enabled),
                "used_history": bool(used_history),
                "rewritten": bool(used_history),
                "conversation_id_hash": conversation_id_hash,
                "original_length": len(request.question),
                "normalized_length": len(normalized),
                "standalone_length": len(standalone),
                "not_found_diagnostics": {
                    "not_found_in_system_prompt": NOT_FOUND in load_system_prompt(),
                    "not_found_in_generation_template": False,
                    "not_found_in_context": NOT_FOUND in context,
                    "raw_answer_starts_with_not_found": answer.startswith(NOT_FOUND),
                    "postprocess_prepends_not_found": False,
                },
            },
        )

        if context_enabled:
            conversation_memory.append_turn(
                request.conversation_id,
                request.user_id,
                normalized,
                standalone,
            )

        return AskResponse(
            answer=answer,
            sources=sources,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "RAG request failed request_id=%s readiness=%s",
            request_id,
            _readiness["status"],
        )
        raise HTTPException(
            status_code=500,
            detail="RAG request failed",
        ) from exc