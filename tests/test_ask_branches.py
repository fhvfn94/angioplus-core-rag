# -*- coding: utf-8 -*-
"""Structural tests that exercise every logging branch of ask().

Each test asserts the exact expected outcome of a branch. If any logger call
had a placeholder/argument mismatch, logging would raise TypeError and the
call would either surface as an HTTP 500 or leak an unexpected exception,
failing the assertion and exposing the defect.
"""

import pytest
from fastapi import HTTPException

import app.main as rag


class _FakeChunk:
    def __init__(self, score, text):
        self.score = score
        self.payload = {
            "file_name": "test.pdf",
            "section": "section",
            "page_start": 1,
            "page_end": 2,
            "text": text,
        }


class _Gate:
    def __init__(self, direct):
        self.direct_answer = direct


class _FakeLLM:
    def __init__(self, gate=True, answer="Генерация успешна.", gen_error=None, gate_error=None):
        self._gate = gate
        self._answer = answer
        self._gen_error = gen_error
        self._gate_error = gate_error

    def check_direct_answer(self, question, context):
        if self._gate_error is not None:
            raise self._gate_error
        return _Gate(self._gate)

    def generate_answer(self, question, context, system_prompt):
        if self._gen_error is not None:
            raise self._gen_error
        return self._answer


def _chunks(n=1, score=0.9):
    return [_FakeChunk(score=score, text=f"Chunk {i} про AngioPlus Core.") for i in range(n)]


def _setup(monkeypatch, chunks, llm, embed_raises=None):
    monkeypatch.setattr(rag, "QUERY_NORMALIZATION_ENABLED", True)
    monkeypatch.setattr(rag, "QUERY_CONTEXT_ENABLED", False)
    rag._readiness["status"] = "ready"
    rag.conversation_memory._store.clear()

    def _embed(q):
        if embed_raises is not None:
            raise embed_raises
        return [0.1, 0.2]

    monkeypatch.setattr(rag, "embed_query", _embed)
    monkeypatch.setattr(rag, "search_qdrant", lambda v, k: chunks)
    monkeypatch.setattr(rag, "get_llm", lambda: llm)


def _req(question="Кто такие Пульс Мьюзикл?"):
    return rag.AskRequest(question=question, top_k=5)


def test_branch_success(monkeypatch):
    _setup(monkeypatch, _chunks(), _FakeLLM(answer="SOME_ANSWER"))
    resp = rag.ask(_req())
    assert resp.answer == "SOME_ANSWER"
    assert resp.sources


def test_branch_retrieval_empty(monkeypatch):
    _setup(monkeypatch, [], _FakeLLM())
    resp = rag.ask(_req())
    assert resp.answer == rag.NOT_FOUND
    assert resp.sources == []


def test_branch_gate_false(monkeypatch):
    _setup(monkeypatch, _chunks(), _FakeLLM(gate=False))
    resp = rag.ask(_req())
    assert resp.answer == rag.NOT_FOUND
    # Intentional: on gate=false no unrelated sources are shown.
    assert resp.sources == []


def test_branch_secret_blocked(monkeypatch):
    _setup(monkeypatch, _chunks(), _FakeLLM())
    resp = rag.ask(_req(question="А какой пароль?"))
    assert resp.answer == rag.SECRET_REFUSAL
    assert resp.sources == []


def test_branch_gate_error_transient(monkeypatch):
    err = rag.LLMError(rag.LLMErrorType.TEMPORARY_UNAVAILABLE, "boom", 503)
    _setup(monkeypatch, _chunks(), _FakeLLM(gate_error=err))
    resp = rag.ask(_req())
    assert resp.answer == rag.LLM_TEMPORARILY_UNAVAILABLE


def test_branch_generation_transient_error(monkeypatch):
    err = rag.LLMError(rag.LLMErrorType.INVALID_RESPONSE, "boom", None)
    _setup(monkeypatch, _chunks(), _FakeLLM(gen_error=err))
    resp = rag.ask(_req())
    assert resp.answer == ("Сервис генерации ответа временно недоступен. "
                           "Попробуйте повторить запрос позже.")


def test_branch_output_secret_filter(monkeypatch):
    _setup(monkeypatch, _chunks(), _FakeLLM(answer="секрет pulse2015 1234"))
    resp = rag.ask(_req())
    assert resp.answer == rag.SECRET_REFUSAL


def test_branch_unexpected_500(monkeypatch):
    _setup(monkeypatch, _chunks(), _FakeLLM(), embed_raises=RuntimeError("boom"))
    with pytest.raises(HTTPException) as excinfo:
        rag.ask(_req())
    assert excinfo.value.status_code == 500


def test_low_score_branch_logging(monkeypatch):
    _setup(monkeypatch, _chunks(score=0.3), _FakeLLM())
    resp = rag.ask(_req())
    assert resp.answer == "Генерация успешна."
