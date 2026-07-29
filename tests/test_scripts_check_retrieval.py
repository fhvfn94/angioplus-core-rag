# -*- coding: utf-8 -*-
from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import check_retrieval
from tests.helpers import make_point


def test_main_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY not found"):
        check_retrieval.main()


def test_main_prints_retrieved_chunks(monkeypatch, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "key")
    captured: dict = {}

    def fake_embed_query(api_key, question):
        captured["api_key"] = api_key
        captured["question"] = question
        return [0.1]

    monkeypatch.setattr(check_retrieval.rag, "embed_query", fake_embed_query)
    monkeypatch.setattr(
        check_retrieval.rag,
        "search_qdrant",
        lambda vector, top_k: [make_point(score=0.912345, text="текст чанка")],
    )

    check_retrieval.main()

    out = capsys.readouterr().out
    assert captured["api_key"] == "key"
    assert captured["question"] == check_retrieval.QUESTION
    assert "RANK     : 1" in out
    assert "SCORE    : 0.912345" in out
    assert "текст чанка" in out


def test_main_reports_no_results(monkeypatch, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "key")
    monkeypatch.setattr(check_retrieval.rag, "embed_query", lambda *a, **k: [0.1])
    monkeypatch.setattr(check_retrieval.rag, "search_qdrant", lambda *a, **k: [])

    check_retrieval.main()

    assert "No retrieval results." in capsys.readouterr().out


def test_main_handles_missing_payload(monkeypatch, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "key")
    monkeypatch.setattr(check_retrieval.rag, "embed_query", lambda *a, **k: [0.1])
    monkeypatch.setattr(
        check_retrieval.rag,
        "search_qdrant",
        lambda *a, **k: [SimpleNamespace(score=0.5, payload=None)],
    )

    check_retrieval.main()

    assert "FILE     : None" in capsys.readouterr().out
