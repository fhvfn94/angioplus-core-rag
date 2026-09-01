from __future__ import annotations

import os
import time
from typing import Any

import httpx

from .models import SearchResponse, SearchResult


VECTOR_STORE_SEARCH_URL = (
    "https://ai.api.cloud.yandex.net/v1/vector_stores/{vector_store_id}/search"
)


class YandexVectorStoreClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        folder_id: str | None = None,
        vector_store_id: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("YANDEX_API_KEY")
        self.folder_id = folder_id or os.environ.get("YANDEX_FOLDER_ID")
        self.vector_store_id = vector_store_id or os.environ.get(
            "YANDEX_VECTOR_STORE_ID"
        )

        if not self.api_key:
            raise RuntimeError("YANDEX_API_KEY is not set")

        if not self.folder_id:
            raise RuntimeError("YANDEX_FOLDER_ID is not set")

        if not self.vector_store_id:
            raise RuntimeError("YANDEX_VECTOR_STORE_ID is not set")

        self.timeout = timeout

    def search(
        self,
        query: str,
        *,
        max_num_results: int = 5,
        rewrite_query: bool = False,
    ) -> SearchResponse:
        url = VECTOR_STORE_SEARCH_URL.format(
            vector_store_id=self.vector_store_id
        )

        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "x-project": self.folder_id,
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "query": query,
            "rewrite_query": rewrite_query,
            "max_num_results": max_num_results,
            "ranking_options": {
                "ranker": "auto",
                "score_threshold": 0,
            },
        }

        started = time.perf_counter()

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                url,
                headers=headers,
                json=payload,
            )

        latency_ms = (time.perf_counter() - started) * 1000

        response.raise_for_status()

        data = response.json()

        results: list[SearchResult] = []

        for item in data.get("data", []):
            results.append(
                SearchResult(
                    score=float(item.get("score", 0.0)),
                    text=_extract_text(item),
                    file_id=_extract_file_id(item),
                    filename=_extract_filename(item),
                    attributes=_extract_attributes(item),
                    raw=item,
                )
            )

        return SearchResponse(
            query=query,
            results=results,
            latency_ms=latency_ms,
        )


def _extract_text(item: dict[str, Any]) -> str:
    content = item.get("content")

    if isinstance(content, list):
        parts: list[str] = []

        for part in content:
            if isinstance(part, dict):
                text = part.get("text")

                if isinstance(text, str):
                    parts.append(text)

        if parts:
            return "\n".join(parts)

    text = item.get("text")

    if isinstance(text, str):
        return text

    return ""


def _extract_file_id(item: dict[str, Any]) -> str | None:
    value = item.get("file_id")

    if isinstance(value, str):
        return value

    return None


def _extract_filename(item: dict[str, Any]) -> str | None:
    value = item.get("filename")

    if isinstance(value, str):
        return value

    return None


def _extract_attributes(item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("attributes")

    if isinstance(value, dict):
        return value

    return {}