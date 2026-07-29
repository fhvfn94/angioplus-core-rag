# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient
from google.genai import types

from app import main as rag


@dataclass
class FakeChunk:
    payload: dict
    score: float | None = 0.9


@dataclass
class FakePromptFeedback:
    block_reason: object | None


@dataclass
class FakeCandidate:
    finish_reason: object | None


@dataclass
class FakeResponse:
    text: str | None
    prompt_feedback: FakePromptFeedback | None = None
    candidates: list[FakeCandidate] = field(default_factory=list)


class FakeModels:
    def __init__(self, response: FakeResponse) -> None:
        self._response = response

    def generate_content(self, *, model, contents, config):
        return self._response


class FakeClient:
    def __init__(self, response: FakeResponse) -> None:
        self.models = FakeModels(response)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    return TestClient(rag.app)


def test_ask_maps_embedding_failure_to_502(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(api_key: str, question: str) -> list[float]:
        raise rag.EmbeddingError("boom")

    monkeypatch.setattr(rag, "embed_query", fail)

    response = client.post("/ask", json={"question": "как установить?"})

    assert response.status_code == 502
    assert response.json()["detail"] == "Embedding provider is unavailable"


def test_ask_maps_retrieval_failure_to_503(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rag, "embed_query", lambda api_key, question: [0.1])

    def fail(vector, top_k):
        raise rag.RetrievalError("qdrant is down")

    monkeypatch.setattr(rag, "search_qdrant", fail)

    response = client.post("/ask", json={"question": "как установить?"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Document store is unavailable"


def test_ask_maps_generation_failure_to_502(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rag, "embed_query", lambda api_key, question: [0.1])
    monkeypatch.setattr(
        rag,
        "search_qdrant",
        lambda vector, top_k: [FakeChunk(payload={"text": "chunk text"})],
    )

    def fail(api_key, question, context):
        raise rag.GenerationError("blocked")

    monkeypatch.setattr(rag, "generate_answer", fail)

    response = client.post("/ask", json={"question": "как установить?"})

    assert response.status_code == 502
    assert response.json()["detail"] == "Answer generation failed"


def test_ask_rejects_non_positive_top_k(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "вопрос", "top_k": 0})

    assert response.status_code == 422


def test_ask_does_not_call_llm_without_context(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rag, "embed_query", lambda api_key, question: [0.1])
    monkeypatch.setattr(
        rag,
        "search_qdrant",
        lambda vector, top_k: [FakeChunk(payload={"file_name": "ifu.pdf"})],
    )

    def unexpected(api_key, question, context):
        raise AssertionError("generate_answer must not be called")

    monkeypatch.setattr(rag, "generate_answer", unexpected)

    response = client.post("/ask", json={"question": "вопрос"})

    assert response.status_code == 200
    assert response.json()["answer"] == rag.NOT_FOUND_ANSWER


def test_load_system_prompt_warns_on_unreadable_path(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path,
) -> None:
    monkeypatch.setattr(rag, "SYSTEM_PROMPT_PATH", str(tmp_path / "missing.md"))

    with caplog.at_level(logging.WARNING, logger=rag.logger.name):
        prompt = rag.load_system_prompt()

    assert prompt == rag.FALLBACK_SYSTEM_PROMPT
    assert "could not be read" in caplog.text


def test_load_system_prompt_warns_on_empty_file(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path,
) -> None:
    prompt_path = tmp_path / "system_prompt.md"
    prompt_path.write_text("   \n", encoding="utf-8")
    monkeypatch.setattr(rag, "SYSTEM_PROMPT_PATH", str(prompt_path))

    with caplog.at_level(logging.WARNING, logger=rag.logger.name):
        prompt = rag.load_system_prompt()

    assert prompt == rag.FALLBACK_SYSTEM_PROMPT
    assert "is empty" in caplog.text


def test_generate_answer_raises_when_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = FakeResponse(
        text=None,
        prompt_feedback=FakePromptFeedback(block_reason=types.BlockedReason.SAFETY),
        candidates=[FakeCandidate(finish_reason=types.FinishReason.SAFETY)],
    )
    monkeypatch.setattr(rag, "create_gemini_client", lambda api_key: FakeClient(response))

    with pytest.raises(rag.GenerationError):
        rag.generate_answer("test-key", "вопрос", "context")


def test_generate_answer_falls_back_when_empty_without_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = FakeResponse(
        text="",
        candidates=[FakeCandidate(finish_reason=types.FinishReason.STOP)],
    )
    monkeypatch.setattr(rag, "create_gemini_client", lambda api_key: FakeClient(response))

    assert rag.generate_answer("test-key", "вопрос", "context") == rag.NOT_FOUND_ANSWER


def test_embed_query_wraps_provider_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingModels:
        def embed_content(self, *, model, contents, config):
            raise ConnectionError("no route to host")

    class FailingClient:
        models = FailingModels()

    monkeypatch.setattr(rag, "create_gemini_client", lambda api_key: FailingClient())

    with pytest.raises(rag.EmbeddingError):
        rag.embed_query("test-key", "вопрос")
