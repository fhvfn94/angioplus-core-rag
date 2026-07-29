# -*- coding: utf-8 -*-
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import main as rag
from tests.helpers import make_point


class FakeModels:
    def __init__(self, embeddings=None, text=None):
        self._embeddings = embeddings
        self._text = text
        self.embed_calls: list[dict] = []
        self.generate_calls: list[dict] = []

    def embed_content(self, *, model, contents, config):
        self.embed_calls.append(
            {"model": model, "contents": contents, "config": config}
        )
        return SimpleNamespace(embeddings=self._embeddings)

    def generate_content(self, *, model, contents, config):
        self.generate_calls.append(
            {"model": model, "contents": contents, "config": config}
        )
        return SimpleNamespace(text=self._text)


class FakeGeminiClient:
    def __init__(self, embeddings=None, text=None):
        self.models = FakeModels(embeddings=embeddings, text=text)


@pytest.fixture
def client():
    return TestClient(rag.app)


def test_get_api_key_returns_stripped_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "  secret-key  ")

    assert rag.get_api_key() == "secret-key"


@pytest.mark.parametrize("value", ["", "   "])
def test_get_api_key_raises_when_missing(monkeypatch, value):
    monkeypatch.setenv("GEMINI_API_KEY", value)

    with pytest.raises(HTTPException) as exc_info:
        rag.get_api_key()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "GEMINI_API_KEY is not set"


def test_create_gemini_client_passes_api_key(monkeypatch):
    created: dict = {}

    class FakeClient:
        def __init__(self, api_key):
            created["api_key"] = api_key

    monkeypatch.setattr(rag.genai, "Client", FakeClient)

    client = rag.create_gemini_client("key")

    assert isinstance(client, FakeClient)
    assert created["api_key"] == "key"


def test_embed_query_returns_embedding_values(monkeypatch):
    fake = FakeGeminiClient(embeddings=[SimpleNamespace(values=[0.1, 0.2])])
    monkeypatch.setattr(rag, "create_gemini_client", lambda api_key: fake)

    assert rag.embed_query("key", "вопрос") == [0.1, 0.2]

    call = fake.models.embed_calls[0]
    assert call["model"] == rag.DEFAULT_GEMINI_EMBEDDING_MODEL
    assert call["contents"] == "вопрос"
    assert call["config"].task_type == "RETRIEVAL_QUERY"


@pytest.mark.parametrize(
    "embeddings, message",
    [
        ([], "Gemini returned no query embedding"),
        ([SimpleNamespace(values=[])], "Gemini returned an empty query embedding"),
    ],
)
def test_embed_query_rejects_missing_embeddings(monkeypatch, embeddings, message):
    fake = FakeGeminiClient(embeddings=embeddings)
    monkeypatch.setattr(rag, "create_gemini_client", lambda api_key: fake)

    with pytest.raises(RuntimeError, match=message):
        rag.embed_query("key", "вопрос")


def test_search_qdrant_queries_default_collection(monkeypatch):
    points = [make_point(), make_point()]
    captured: dict = {}

    class FakeQdrantClient:
        def __init__(self, url):
            captured["url"] = url

        def query_points(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(points=points)

    monkeypatch.setattr(rag, "QdrantClient", FakeQdrantClient)

    assert rag.search_qdrant([0.1], top_k=3) == points
    assert captured["url"] == rag.DEFAULT_QDRANT_URL
    assert captured["collection_name"] == rag.DEFAULT_COLLECTION
    assert captured["query"] == [0.1]
    assert captured["limit"] == 3
    assert captured["with_payload"] is True


def test_build_context_numbers_chunks_and_tolerates_missing_payload():
    chunks = [
        make_point(text="первый"),
        SimpleNamespace(score=0.5, payload=None),
    ]

    assert rag.build_context(chunks) == "[Chunk 1]\nпервый\n\n[Chunk 2]\n"


def test_build_context_with_no_chunks_is_empty():
    assert rag.build_context([]) == ""


def test_load_system_prompt_reads_file(monkeypatch, tmp_path):
    prompt_file = tmp_path / "system_prompt.md"
    prompt_file.write_text("  Системный промпт  \n", encoding="utf-8")
    monkeypatch.setattr(rag, "SYSTEM_PROMPT_PATH", str(prompt_file))

    assert rag.load_system_prompt() == "Системный промпт"


def test_load_system_prompt_falls_back_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "SYSTEM_PROMPT_PATH", str(tmp_path / "missing.md"))

    assert "AngioPlus Core" in rag.load_system_prompt()


def test_generate_answer_uses_system_prompt_and_context(monkeypatch):
    fake = FakeGeminiClient(text="  Ответ  ")
    monkeypatch.setattr(rag, "create_gemini_client", lambda api_key: fake)
    monkeypatch.setattr(rag, "load_system_prompt", lambda: "SYSTEM")

    answer = rag.generate_answer("key", "вопрос", "контекст")

    assert answer == "Ответ"
    call = fake.models.generate_calls[0]
    assert call["model"] == rag.DEFAULT_GEMINI_MODEL
    assert "контекст" in call["contents"]
    assert "вопрос" in call["contents"]
    assert call["config"].system_instruction == "SYSTEM"


@pytest.mark.parametrize("text", [None, ""])
def test_generate_answer_falls_back_on_empty_response(monkeypatch, text):
    monkeypatch.setattr(
        rag,
        "create_gemini_client",
        lambda api_key: FakeGeminiClient(text=text),
    )
    monkeypatch.setattr(rag, "load_system_prompt", lambda: "SYSTEM")

    assert rag.generate_answer("key", "вопрос", "контекст") == (
        "Такой информации нет в имеющейся документации."
    )


def test_build_sources_keeps_only_relevant_chunks():
    chunks = [
        make_point(score=0.95, file_name="a.pdf"),
        make_point(score=0.5, file_name="b.pdf"),
    ]

    sources = rag.build_sources(chunks)

    assert [source.file_name for source in sources] == ["a.pdf"]
    assert sources[0].score == 0.95


def test_build_sources_deduplicates_identical_metadata():
    chunks = [
        make_point(score=0.9123, text="один"),
        make_point(score=0.91, text="два"),
    ]

    sources = rag.build_sources(chunks)

    assert len(sources) == 1
    assert sources[0].score == 0.912


def test_build_sources_falls_back_to_best_chunk_below_threshold():
    chunks = [
        make_point(score=0.4, file_name="a.pdf"),
        make_point(score=0.3, file_name="b.pdf"),
    ]

    sources = rag.build_sources(chunks)

    assert [source.file_name for source in sources] == ["a.pdf"]


def test_build_sources_handles_missing_score_and_payload():
    chunks = [SimpleNamespace(score=None, payload=None)]

    sources = rag.build_sources(chunks)

    assert len(sources) == 1
    assert sources[0].score is None
    assert sources[0].file_name is None


def test_build_sources_with_no_chunks():
    assert rag.build_sources([]) == []


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("question", ["", "   "])
def test_ask_rejects_empty_question(client, question):
    response = client.post("/ask", json={"question": question})

    assert response.status_code == 400
    assert response.json()["detail"] == "Question is empty"


def test_ask_returns_answer_and_sources(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "key")
    monkeypatch.setattr(rag, "embed_query", lambda api_key, question: [0.1])
    monkeypatch.setattr(
        rag,
        "search_qdrant",
        lambda vector, top_k: [make_point(score=0.9, file_name="IFU.pdf")],
    )
    monkeypatch.setattr(
        rag,
        "generate_answer",
        lambda api_key, question, context: "Ответ",
    )

    response = client.post("/ask", json={"question": " Как установить? ", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Ответ"
    assert body["sources"][0]["file_name"] == "IFU.pdf"


def test_ask_returns_placeholder_when_no_chunks(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "key")
    monkeypatch.setattr(rag, "embed_query", lambda api_key, question: [0.1])
    monkeypatch.setattr(rag, "search_qdrant", lambda vector, top_k: [])

    response = client.post("/ask", json={"question": "вопрос"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Not found in documentation.",
        "sources": [],
    }


def test_ask_requires_api_key(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")

    response = client.post("/ask", json={"question": "вопрос"})

    assert response.status_code == 500
    assert response.json()["detail"] == "GEMINI_API_KEY is not set"


def test_ask_propagates_http_exceptions_unchanged(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "key")

    def raise_http_error(api_key, question):
        raise HTTPException(status_code=429, detail="quota exceeded")

    monkeypatch.setattr(rag, "embed_query", raise_http_error)

    response = client.post("/ask", json={"question": "вопрос"})

    assert response.status_code == 429
    assert response.json()["detail"] == "quota exceeded"


def test_ask_wraps_unexpected_errors(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "key")

    def boom(api_key, question):
        raise RuntimeError("gemini down")

    monkeypatch.setattr(rag, "embed_query", boom)

    response = client.post("/ask", json={"question": "вопрос"})

    assert response.status_code == 500
    assert response.json()["detail"] == "RAG request failed: gemini down"
