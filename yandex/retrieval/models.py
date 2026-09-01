from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SearchResult:
    score: float
    text: str
    file_id: str | None = None
    filename: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchResponse:
    query: str
    results: list[SearchResult]
    latency_ms: float


@dataclass(slots=True)
class EvidenceChunk:
    rank: int
    score: float
    text: str

    source: str | None = None
    source_type: str = "unknown"
    authority: int = 0

    section: str | None = None
    pages: str | None = None

    file_id: str | None = None
    vector_store_filename: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalResponse:
    query: str
    evidence: list[EvidenceChunk]
    latency_ms: float
    total_candidates: int