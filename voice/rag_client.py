# -*- coding: utf-8 -*-
"""Async client for the existing AngioPlus RAG API (POST /ask).

This is the ONLY boundary between the future LiveKit voice component and the
existing RAG pipeline. It:

- calls the existing POST /ask endpoint with the exact AskRequest schema;
- parses the AskResponse (answer + sources);
- never accesses Qdrant;
- never imports app.main or any app.* RAG implementation.

The existing RAG pipeline remains the single source of truth for retrieval,
grounding, safety filtering and conversation memory. This client only forwards
the caller-provided conversation_id / user_id; it does NOT implement any
conversation memory of its own.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import httpx


class RagClientError(Exception):
    """Base error for all RAG client failures."""


class RagHttpError(RagClientError):
    """The RAG API returned a non-2xx HTTP status."""

    def __init__(self, status_code: int, method: str, url: str,
                 detail: str | None = None) -> None:
        self.status_code = status_code
        self.method = method
        self.url = url
        self.detail = detail
        super().__init__(
            f"RAG API returned HTTP {status_code} for {method} {url}"
            + (f": {detail}" if detail else "")
        )


class RagTimeoutError(RagClientError):
    """The RAG API request timed out."""


class RagInvalidJsonError(RagClientError):
    """The RAG API returned a body that is not valid JSON."""


class RagMalformedResponseError(RagClientError):
    """The RAG API response was valid JSON but missing/invalid fields."""


@dataclass(frozen=True)
class RagSource:
    """A single cited source in a RAG answer (mirrors the API Source)."""

    file_name: str | None
    section: str | None
    page_start: int | None
    page_end: int | None
    score: float | None


@dataclass(frozen=True)
class RagAnswer:
    """Parsed AskResponse from the existing RAG API."""

    answer: str
    sources: list[RagSource] = field(default_factory=list)


class RagClient:
    """Thin async HTTP client around the existing RAG /ask endpoint.

    One instance owns one httpx.AsyncClient. Close it with ``aclose()`` when
    done (or use it as an async context manager) to release pooled resources.
    """

    def __init__(
        self,
        url: str,
        timeout_seconds: int = 180,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        # Allow dependency injection of an httpx client for tests.
        self._http_client = http_client

    # -- lifecycle ----------------------------------------------------------

    def _client(self) -> httpx.AsyncClient:
        if self._http_client is not None:
            return self._http_client
        # httpx.Timeout with per-operation limits mirrors bot/main.py, with a
        # generous read timeout for the existing RAG latency.
        timeout = httpx.Timeout(
            connect=10.0,
            read=float(self.timeout_seconds),
            write=30.0,
            pool=10.0,
        )
        return httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        """Close the internally-owned httpx client (no-op if injected)."""
        if self._http_client is not None:
            # The caller owns an injected client; do not close it here.
            return
        # If we never created one lazily, there is nothing to close.
        client = getattr(self, "_owns_client", None)
        if client is not None:
            await client.aclose()

    async def __aenter__(self) -> "RagClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # -- request ------------------------------------------------------------

    def _build_payload(
        self,
        question: str,
        top_k: int,
        conversation_id: str | None,
        user_id: str | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question": question,
            "top_k": top_k,
        }
        if conversation_id is not None:
            payload["conversation_id"] = conversation_id
        if user_id is not None:
            payload["user_id"] = user_id
        return payload

    async def ask(
        self,
        question: str,
        top_k: int = 10,
        *,
        conversation_id: str | None = None,
        user_id: str | None = None,
    ) -> RagAnswer:
        """Send the question to the existing RAG /ask endpoint.

        The argument names match the existing AskRequest schema exactly:

        - question: str (required)
        - top_k: int (default 10)
        - conversation_id: str | None
        - user_id: str | None

        Conversation memory is owned by the RAG pipeline; this client only
        forwards the provided identifiers.
        """
        if not question or not str(question).strip():
            raise ValueError("question must be a non-empty string")

        payload = self._build_payload(
            question, top_k, conversation_id, user_id
        )

        client = self._client()
        self._owns_client = client

        try:
            response = await client.post(self.url, json=payload)
        except httpx.TimeoutException as exc:
            raise RagTimeoutError(
                f"RAG API request timed out after {self.timeout_seconds}s "
                f"(POST {self.url})"
            ) from exc
        except httpx.HTTPError as exc:
            raise RagClientError(
                f"RAG API request failed: {exc}"
            ) from exc

        if response.status_code >= 400:
            detail = _safe_error_detail(response)
            raise RagHttpError(
                status_code=response.status_code,
                method="POST",
                url=self.url,
                detail=detail,
            )

        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise RagInvalidJsonError(
                f"RAG API returned non-JSON body for POST {self.url}"
            ) from exc

        return _parse_answer(data)

    async def ask_dict(
        self,
        question: str,
        top_k: int = 10,
        *,
        conversation_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Like :meth:`ask` but returns the raw parsed response dict.

        Useful for callers that also need the raw payload; the answer/sources
        are still validated the same way.
        """
        ab = await self.ask(
            question,
            top_k=top_k,
            conversation_id=conversation_id,
            user_id=user_id,
        )
        return {
            "answer": ab.answer,
            "sources": [
                {
                    "file_name": s.file_name,
                    "section": s.section,
                    "page_start": s.page_start,
                    "page_end": s.page_end,
                    "score": s.score,
                }
                for s in ab.sources
            ],
        }


# -- helpers ----------------------------------------------------------------


def _safe_error_detail(response: httpx.Response) -> str | None:
    """Extract a short, safe error detail from an error response."""
    try:
        data = response.json()
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(data, dict):
        detail = data.get("detail")
        if isinstance(detail, str):
            return detail[:300]
    return None


def _parse_answer(data: Mapping[str, Any]) -> RagAnswer:
    """Validate and build a RagAnswer from a parsed /ask response.

    Raises RagMalformedResponseError on missing/invalid fields.
    """
    if not isinstance(data, dict):
        raise RagMalformedResponseError(
            f"RAG response must be a JSON object, got {type(data).__name__}"
        )

    answer = data.get("answer")
    if not isinstance(answer, str):
        raise RagMalformedResponseError(
            "RAG response is missing a string 'answer' field"
        )

    raw_sources = data.get("sources", [])
    if not isinstance(raw_sources, list):
        raise RagMalformedResponseError(
            "RAG response 'sources' must be a list"
        )

    sources: list[RagSource] = []
    for item in raw_sources:
        if not isinstance(item, dict):
            raise RagMalformedResponseError(
                "RAG response 'sources' items must be objects"
            )
        sources.append(
            RagSource(
                file_name=_as_optional_str(item.get("file_name")),
                section=_as_optional_str(item.get("section")),
                page_start=_as_optional_int(item.get("page_start")),
                page_end=_as_optional_int(item.get("page_end")),
                score=_as_optional_float(item.get("score")),
            )
        )

    return RagAnswer(answer=answer, sources=sources)


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
