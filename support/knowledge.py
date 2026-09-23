"""
Knowledge -- the provider-agnostic retrieval contract and execution core.

Infrastructure only. NOT imported by any production runtime path
(server.py, alice_support_agent.py, n8n, Tracker, DB). This slice (5A) is
the pure, provider-independent half of the Knowledge layer:

  * the immutable evidence contract (:class:`KnowledgeItem`,
    :class:`KnowledgeResult`),
  * the closed status vocabulary (:class:`KnowledgeStatus`),
  * the backend port (:class:`KnowledgeBackend`),
  * the execution core (:func:`_retrieve`) that maps a backend outcome to a
    status WITHOUT any provider, network, or LLM dependency.

This module is deliberately free of any workflow or provider vocabulary:
it imports no router, no procedure state, no analyzer, no Yandex SDK, and
no HTTP client. It carries no candidate action and no next-action verdict.
Turning evidence into an action belongs to later layers.

There is no I/O here, so the whole module is unit-testable in isolation
with an in-memory fake backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, Sequence


DEFAULT_MAX_RESULTS = 5


class _StrEnum(str, Enum):
    """Python 3.10 compatible str-enum (values are plain strings)."""

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return str(self.value)


class KnowledgeStatus(_StrEnum):
    """Outcome of one retrieval execution.

    A closed set of exactly three values:

      * ``FOUND``     -- at least one usable evidence item was returned.
      * ``NOT_FOUND`` -- the provider returned no usable evidence.
      * ``ERROR``     -- the provider/transport failed; no evidence is
                         present and the failure is described by
                         :attr:`KnowledgeResult.error`.

    ``ERROR`` is a fail-safe state, not a crash: the core never lets a
    backend exception escape.
    """

    FOUND = "found"
    NOT_FOUND = "not_found"
    ERROR = "error"


@dataclass(frozen=True)
class KnowledgeItem:
    """One immutable piece of retrieved evidence.

    Only fields supported by the existing retrieval architecture are kept:
    the vector-store client exposes text/score/file_id/filename/attributes,
    and the retrieval controller derives source/section/pages/source_type/
    authority plus a search-order rank. No workflow field is included.
    """

    text: str
    source: Optional[str] = None
    section: Optional[str] = None
    pages: Optional[str] = None
    filename: Optional[str] = None
    file_id: Optional[str] = None
    source_type: str = "unknown"
    authority: int = 0
    score: Optional[float] = None
    rank: int = 0


@dataclass(frozen=True)
class KnowledgeResult:
    """Immutable result of one retrieval execution.

    Invariants (enforced on construction):

      * ``FOUND``     -> at least one usable evidence item, and no error.
      * ``NOT_FOUND`` -> no usable evidence, and no error.
      * ``ERROR``     -> no evidence at all, and a non-empty internal error.

    ``error`` is INFRASTRUCTURE data only: it is never rendered into
    user-facing text here and never parsed for control flow by callers.
    """

    status: KnowledgeStatus
    query: str
    items: tuple[KnowledgeItem, ...] = ()
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status is KnowledgeStatus.FOUND:
            if not any(_is_usable(i) for i in self.items):
                raise ValueError(
                    "FOUND requires at least one usable evidence item"
                )
            if self.error is not None:
                raise ValueError("FOUND must not carry an error")
        elif self.status is KnowledgeStatus.NOT_FOUND:
            if any(_is_usable(i) for i in self.items):
                raise ValueError(
                    "NOT_FOUND must not contain usable evidence"
                )
            if self.error is not None:
                raise ValueError("NOT_FOUND must not carry an error")
        elif self.status is KnowledgeStatus.ERROR:
            if self.items:
                raise ValueError("ERROR must not contain evidence")
            if not self.error:
                raise ValueError(
                    "ERROR must carry internal error information"
                )
        else:  # pragma: no cover - exhaustive over KnowledgeStatus
            raise ValueError(f"unknown KnowledgeStatus: {self.status!r}")


class KnowledgeBackend(Protocol):
    """Provider-agnostic retrieval port.

    Implementations accept a query and a result limit and return the
    retrieved evidence items in backend order. A provider or transport
    failure MUST be surfaced by RAISING; the core translates any such
    exception into ``ERROR``.
    """

    def search(
        self,
        query: str,
        *,
        max_num_results: int,
    ) -> Sequence[KnowledgeItem]:
        ...


# ============================================================
# PURE HELPERS
# ============================================================

def _is_usable(item: object) -> bool:
    """True only for a KnowledgeItem that carries non-blank text."""
    return isinstance(item, KnowledgeItem) and bool(
        item.text and item.text.strip()
    )


def _describe_error(exc: BaseException) -> str:
    """Compact internal error string. Never user-facing."""
    return f"{type(exc).__name__}: {exc}"


def _normalize_query(query: str) -> str:
    """Minimal, provider-independent query normalization (strip + collapse)."""
    return " ".join((query or "").split())


# ============================================================
# EXECUTION CORE
# ============================================================

def _retrieve(
    query: str,
    *,
    backend: KnowledgeBackend,
    max_num_results: int = DEFAULT_MAX_RESULTS,
) -> KnowledgeResult:
    """Execute one retrieval against ``backend`` and map the outcome.

    Provider-independent core. Total and lossless on failures: no exception
    raised by the backend (or by a malformed response) ever escapes here.

      * blank query            -> NOT_FOUND (backend not called)
      * non-empty usable items -> FOUND
      * empty / unusable items -> NOT_FOUND
      * backend failure        -> ERROR
    """
    normalized = _normalize_query(query)

    if not normalized:
        return KnowledgeResult(status=KnowledgeStatus.NOT_FOUND, query="")

    try:
        raw = backend.search(normalized, max_num_results=max_num_results)
        candidates = tuple(raw)
    except Exception as exc:  # noqa: BLE001 - core boundary is total
        return KnowledgeResult(
            status=KnowledgeStatus.ERROR,
            query=normalized,
            error=_describe_error(exc),
        )

    usable = tuple(item for item in candidates if _is_usable(item))

    if usable:
        return KnowledgeResult(
            status=KnowledgeStatus.FOUND,
            query=normalized,
            items=usable,
        )

    return KnowledgeResult(status=KnowledgeStatus.NOT_FOUND, query=normalized)


__all__ = [
    "KnowledgeStatus",
    "KnowledgeItem",
    "KnowledgeResult",
    "KnowledgeBackend",
]
