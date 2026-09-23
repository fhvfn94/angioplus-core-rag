from __future__ import annotations

import re

from .client import YandexVectorStoreClient
from .models import EvidenceChunk, RetrievalResponse, SearchResult


DEFAULT_MAX_RESULTS = 5


SOURCE_AUTHORITY = {
    "ifu": 100,
    "training": 80,
    "faq": 70,
    "commercial": 40,
    "competitor": 30,
    "unknown": 10,
}


class RetrievalController:
    def __init__(
        self,
        *,
        client: YandexVectorStoreClient | None = None,
        max_num_results: int = DEFAULT_MAX_RESULTS,
    ) -> None:
        if max_num_results < 1:
            raise ValueError(
                "max_num_results must be greater than 0"
            )

        self.client = client or YandexVectorStoreClient()
        self.max_num_results = max_num_results

    def search(
        self,
        query: str,
    ) -> RetrievalResponse:
        query = query.strip()

        if not query:
            raise ValueError("query must not be empty")

        search_response = self.client.search(
            query,
            max_num_results=self.max_num_results,
            rewrite_query=False,
        )

        evidence = [
            self._build_evidence_chunk(
                result=result,
                rank=rank,
            )
            for rank, result in enumerate(
                search_response.results,
                start=1,
            )
        ]

        return RetrievalResponse(
            query=query,
            evidence=evidence,
            latency_ms=search_response.latency_ms,
            total_candidates=len(evidence),
        )

    def _build_evidence_chunk(
        self,
        *,
        result: SearchResult,
        rank: int,
    ) -> EvidenceChunk:
        # 5D: the source identity lives in the dedicated ``filename`` field
        # that the Vector Store API ALWAYS returns. The chunk text is the raw
        # document body and -- as audited against the live store -- carries no
        # in-text "Документ:" marker, so reading the source from the text alone
        # collapsed ``source`` to None and ``source_type`` to "unknown".
        # The in-text marker is kept ONLY as a fallback for stores that still
        # embed it (then a blank/absent filename must not shadow it).
        source = (result.filename or "").strip() or _extract_metadata_value(
            result.text,
            "Документ",
        )

        section = _extract_metadata_value(
            result.text,
            "Раздел",
        )

        pages = _extract_metadata_value(
            result.text,
            "Страницы",
        )

        source_type = _classify_source(
            source=source,
            text=result.text,
        )

        authority = SOURCE_AUTHORITY.get(
            source_type,
            SOURCE_AUTHORITY["unknown"],
        )

        return EvidenceChunk(
            rank=rank,
            score=result.score,
            text=result.text,
            source=source,
            source_type=source_type,
            authority=authority,
            section=section,
            pages=pages,
            file_id=result.file_id,
            vector_store_filename=result.filename,
            attributes=result.attributes,
        )


def search_angioplus_knowledge(
    query: str,
    *,
    controller: RetrievalController | None = None,
) -> RetrievalResponse:
    active_controller = controller or RetrievalController()

    return active_controller.search(query)


def _extract_metadata_value(
    text: str,
    key: str,
) -> str | None:
    pattern = rf"{re.escape(key)}:\s*(.+?)(?=\s+[А-ЯA-Z][^:\n]*:|$)"

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    value = match.group(1).strip()

    return value or None


def _classify_source(
    *,
    source: str | None,
    text: str,
) -> str:
    source_normalized = (source or "").lower()
    text_normalized = text.lower()

    if "ifu" in source_normalized:
        return "ifu"

    if "2025july" in source_normalized:
        return "training"

    if (
        "q&a list" in source_normalized
        or "source type / тип источника: faq"
        in text_normalized
    ):
        return "faq"

    if "common info to distributor" in source_normalized:
        return "commercial"

    if "competitor" in source_normalized:
        return "competitor"

    return "unknown"