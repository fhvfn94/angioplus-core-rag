# -*- coding: utf-8 -*-
"""Tests for the LiveKit voice <-> RAG API boundary client.

These tests use a hand-rolled fake httpx transport so they do NOT require a
real RAG server and do NOT require LiveKit to be installed.
"""
from __future__ import annotations

import json

import httpx
import pytest

from voice.rag_client import (
    RagAnswer,
    RagClient,
    RagClientError,
    RagHttpError,
    RagInvalidJsonError,
    RagMalformedResponseError,
    RagSource,
    RagTimeoutError,
)


class _FakeTransport(httpx.AsyncBaseTransport):
    """Configurable fake transport that returns a canned httpx.Response."""

    def __init__(self, status_code=200, json_body=None, bytes_body=None,
                 read_timeout=False):
        self.status_code = status_code
        self.json_body = json_body
        self.bytes_body = bytes_body
        self.read_timeout = read_timeout
        self.requests = []  # captured request artifacts for assertions

    async def handle_async_request(self, request):
        self.requests.append(
            {
                "method": request.method,
                "url": str(request.url),
                "content": request.content,
            }
        )
        if self.read_timeout:
            raise httpx.ReadTimeout("timed out", request=request)

        if self.json_body is not None:
            content = json.dumps(self.json_body).encode("utf-8")
        else:
            content = self.bytes_body or b""

        return httpx.Response(
            self.status_code,
            content=content,
            request=request,
            headers={"content-type": "application/json"},
        )


def _make_client(transport: _FakeTransport, url="http://127.0.0.1:8000/ask"):
    http_client = httpx.AsyncClient(transport=transport)
    return RagClient(
        url=url,
        timeout_seconds=5,
        http_client=http_client,
    ), http_client


# ---------------------------------------------------------------------------
# request payload
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_request_payload_matches_ask_request_schema():
    transport = _FakeTransport(
        json_body={"answer": "ok", "sources": []}
    )
    client, _ = _make_client(transport)

    await client.ask(
        "Какие системные требования?",
        conversation_id="chat-1",
        user_id="user-1",
    )

    assert len(transport.requests) == 1
    req = transport.requests[0]
    assert req["method"] == "POST"
    assert req["url"] == "http://127.0.0.1:8000/ask"

    payload = json.loads(req["content"])
    assert payload == {
        "question": "Какие системные требования?",
        "top_k": 10,
        "conversation_id": "chat-1",
        "user_id": "user-1",
    }


@pytest.mark.anyio
async def test_default_top_k_is_10_and_ids_optional():
    transport = _FakeTransport(json_body={"answer": "ok", "sources": []})
    client, _ = _make_client(transport)

    await client.ask("Просто вопрос")

    payload = json.loads(transport.requests[0]["content"])
    assert payload["top_k"] == 10
    assert "conversation_id" not in payload
    assert "user_id" not in payload


@pytest.mark.anyio
async def test_custom_top_k_is_forwarded():
    transport = _FakeTransport(json_body={"answer": "ok", "sources": []})
    client, _ = _make_client(transport)

    await client.ask("Вопрос", top_k=3)

    payload = json.loads(transport.requests[0]["content"])
    assert payload["top_k"] == 3


@pytest.mark.anyio
async def test_empty_question_raises_value_error():
    transport = _FakeTransport(json_body={"answer": "ok", "sources": []})
    client, _ = _make_client(transport)

    with pytest.raises(ValueError):
        await client.ask("   ")


# ---------------------------------------------------------------------------
# successful parsing
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_successful_answer_parsing():
    transport = _FakeTransport(
        json_body={"answer": "Сгенерированный ответ.", "sources": []}
    )
    client, http_client = _make_client(transport)
    try:
        result = await client.ask("Вопрос")
    finally:
        await http_client.aclose()

    assert isinstance(result, RagAnswer)
    assert result.answer == "Сгенерированный ответ."
    assert result.sources == []


@pytest.mark.anyio
async def test_sources_parsing():
    transport = _FakeTransport(
        json_body={
            "answer": "Ответ",
            "sources": [
                {
                    "file_name": "IFU.pdf",
                    "section": "12 Техподдержка",
                    "page_start": 10,
                    "page_end": 11,
                    "score": 0.91,
                },
                {
                    "file_name": "FAQ.xlsx",
                    "section": "Глава 2",
                    "page_start": 3,
                    "page_end": None,
                    "score": 0.77,
                },
            ],
        }
    )
    client, http_client = _make_client(transport)
    try:
        result = await client.ask("Вопрос")
    finally:
        await http_client.aclose()

    assert len(result.sources) == 2
    s0 = result.sources[0]
    assert isinstance(s0, RagSource)
    assert s0.file_name == "IFU.pdf"
    assert s0.section == "12 Техподдержка"
    assert s0.page_start == 10
    assert s0.page_end == 11
    assert s0.score == 0.91
    s1 = result.sources[1]
    assert s1.page_end is None
    assert s1.score == 0.77


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_http_error():
    transport = _FakeTransport(
        status_code=503,
        json_body={"detail": "RAG service is not ready"},
    )
    client, http_client = _make_client(transport)
    try:
        with pytest.raises(RagHttpError) as exc_info:
            await client.ask("Вопрос")
        err = exc_info.value
        assert err.status_code == 503
        assert err.method == "POST"
        assert err.detail == "RAG service is not ready"
        assert "503" in str(err)
    finally:
        await http_client.aclose()


@pytest.mark.anyio
async def test_http_error_without_json_detail():
    transport = _FakeTransport(status_code=500, bytes_body=b"boom")
    client, http_client = _make_client(transport)
    try:
        with pytest.raises(RagHttpError) as exc_info:
            await client.ask("Вопрос")
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail is None
    finally:
        await http_client.aclose()


@pytest.mark.anyio
async def test_timeout():
    transport = _FakeTransport(read_timeout=True)
    client, http_client = _make_client(transport)
    try:
        with pytest.raises(RagTimeoutError):
            await client.ask("Вопрос")
    finally:
        await http_client.aclose()


@pytest.mark.anyio
async def test_invalid_json_body():
    transport = _FakeTransport(bytes_body=b"<html>not json</html>")
    client, http_client = _make_client(transport)
    try:
        with pytest.raises(RagInvalidJsonError):
            await client.ask("Вопрос")
    finally:
        await http_client.aclose()


@pytest.mark.anyio
async def test_malformed_response_missing_answer():
    transport = _FakeTransport(json_body={"sources": []})
    client, http_client = _make_client(transport)
    try:
        with pytest.raises(RagMalformedResponseError) as exc_info:
            await client.ask("Вопрос")
        assert "answer" in str(exc_info.value)
    finally:
        await http_client.aclose()


@pytest.mark.anyio
async def test_malformed_response_answer_not_string():
    transport = _FakeTransport(json_body={"answer": 123, "sources": []})
    client, http_client = _make_client(transport)
    try:
        with pytest.raises(RagMalformedResponseError):
            await client.ask("Вопрос")
    finally:
        await http_client.aclose()


@pytest.mark.anyio
async def test_malformed_response_sources_not_a_list():
    transport = _FakeTransport(json_body={"answer": "ok", "sources": "x"})
    client, http_client = _make_client(transport)
    try:
        with pytest.raises(RagMalformedResponseError):
            await client.ask("Вопрос")
    finally:
        await http_client.aclose()


@pytest.mark.anyio
async def test_malformed_response_non_object_top_level():
    transport = _FakeTransport(json_body=["not", "an", "object"])
    client, http_client = _make_client(transport)
    try:
        with pytest.raises(RagMalformedResponseError):
            await client.ask("Вопрос")
    finally:
        await http_client.aclose()


# ---------------------------------------------------------------------------
# unknown / network errors wrap into RagClientError
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_network_error_wrapped():
    class _RaisingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            raise httpx.ConnectError("connection refused", request=request)

    http_client = httpx.AsyncClient(transport=_RaisingTransport())
    client = RagClient(
        url="http://127.0.0.1:8000/ask", http_client=http_client
    )
    try:
        with pytest.raises(RagClientError):
            await client.ask("Вопрос")
    finally:
        await http_client.aclose()
