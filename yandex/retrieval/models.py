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