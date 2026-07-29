# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from scripts import ask
from tests.helpers import make_point


class FakeQdrantClient:
    def __init__(self, points=None, error: Exception | None = None, **kwargs):
        self.points = points or []
        self.error = error
        self.kwargs = kwargs
        self.calls: list[dict] = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(points=self.points)


@pytest.fixture()
def fake_genai(monkeypatch):
    import google.generativeai as genai

    calls: dict = {}

    monkeypatch.setattr(genai, "configure", lambda api_key: calls.update(key=api_key))
    monkeypatch.setattr(
        genai,
        "embed_content",
        lambda model, content, task_type: calls.update(
            model=model, content=content, task_type=task_type
        )
        or {"embedding": [0.1, 0.2]},
    )
    return calls


def test_embed_query_uses_retrieval_query_task(fake_genai):
    vector = ask.embed_query("key", "вопрос", "models/gemini-embedding-001")

    assert vector == [0.1, 0.2]
    assert fake_genai["key"] == "key"
    assert fake_genai["model"] == "models/gemini-embedding-001"
    assert fake_genai["task_type"] == "retrieval_query"


def test_search_qdrant_passes_query_parameters():
    points = [make_point()]
    client = FakeQdrantClient(points=points)

    result = ask.search_qdrant(
        client=client,
        collection="col",
        query_vector=[0.1],
        limit=7,
    )

    assert result == points
    assert client.calls[0] == {
        "collection_name": "col",
        "query": [0.1],
        "limit": 7,
        "with_payload": True,
    }


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", ""),
        (None, ""),
        ("  короткий текст  ", "короткий текст"),
    ],
)
def test_format_preview_keeps_short_text(text, expected):
    assert ask.format_preview(text) == expected


def test_format_preview_truncates_long_text():
    preview = ask.format_preview("a" * 100, max_chars=10)

    assert preview == "a" * 7 + "..."
    assert len(preview) == 10


def test_resolve_question_prefers_flag():
    args = argparse.Namespace(question="positional", question_flag="  from flag  ")

    assert ask.resolve_question(args) == "from flag"


def test_resolve_question_falls_back_to_positional():
    args = argparse.Namespace(question="  positional  ", question_flag=None)

    assert ask.resolve_question(args) == "positional"


def test_resolve_question_reads_stdin(monkeypatch):
    args = argparse.Namespace(question=None, question_flag=None)
    monkeypatch.setattr(
        ask.sys, "stdin", SimpleNamespace(isatty=lambda: False, read=lambda: " из stdin ")
    )

    assert ask.resolve_question(args) == "из stdin"


def test_resolve_question_exits_when_interactive_without_question(monkeypatch, capsys):
    args = argparse.Namespace(question=None, question_flag=None)
    monkeypatch.setattr(ask.sys, "stdin", SimpleNamespace(isatty=lambda: True))

    with pytest.raises(SystemExit) as exc_info:
        ask.resolve_question(args)

    assert exc_info.value.code == 2
    assert "provide a question" in capsys.readouterr().err


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(ask.sys, "argv", ["ask.py", "вопрос"])

    args = ask.parse_args()

    assert args.question == "вопрос"
    assert args.top_k == 5
    assert args.collection == ask.DEFAULT_COLLECTION
    assert args.qdrant_url == ask.DEFAULT_QDRANT_URL


def test_parse_args_overrides(monkeypatch):
    monkeypatch.setattr(
        ask.sys,
        "argv",
        [
            "ask.py",
            "-q",
            "вопрос",
            "--top-k",
            "2",
            "--collection",
            "other",
            "--qdrant-url",
            "http://qdrant:6333",
        ],
    )

    args = ask.parse_args()

    assert args.question_flag == "вопрос"
    assert args.top_k == 2
    assert args.collection == "other"
    assert args.qdrant_url == "http://qdrant:6333"


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(
        question="вопрос",
        question_flag=None,
        qdrant_url="http://localhost:6333",
        qdrant_api_key=None,
        collection="col",
        top_k=2,
        gemini_api_key="key",
        gemini_embedding_model="models/gemini-embedding-001",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_main_prints_hits(monkeypatch, capsys):
    monkeypatch.setattr(ask, "parse_args", _args)
    monkeypatch.setattr(ask, "embed_query", lambda *a, **k: [0.1])
    monkeypatch.setattr(
        ask,
        "QdrantClient",
        lambda url, api_key: FakeQdrantClient(
            points=[make_point(score=0.9, text="длинный текст" * 60)]
        ),
    )

    ask.main()

    out = capsys.readouterr().out
    assert "Matched chunks: 1 (top_k=2)" in out
    assert "IFU.pdf" in out
    assert "..." in out


def test_main_reports_empty_result(monkeypatch, capsys):
    monkeypatch.setattr(ask, "parse_args", _args)
    monkeypatch.setattr(ask, "embed_query", lambda *a, **k: [0.1])
    monkeypatch.setattr(
        ask, "QdrantClient", lambda url, api_key: FakeQdrantClient(points=[])
    )

    ask.main()

    assert "(No hits" in capsys.readouterr().out


def test_main_handles_chunk_without_payload(monkeypatch, capsys):
    monkeypatch.setattr(ask, "parse_args", _args)
    monkeypatch.setattr(ask, "embed_query", lambda *a, **k: [0.1])
    monkeypatch.setattr(
        ask,
        "QdrantClient",
        lambda url, api_key: FakeQdrantClient(
            points=[SimpleNamespace(score=0.5, payload=None)]
        ),
    )

    ask.main()

    assert "<empty>" in capsys.readouterr().out


def test_main_exits_on_empty_question(monkeypatch, capsys):
    monkeypatch.setattr(
        ask, "parse_args", lambda: _args(question=None, question_flag="   ")
    )

    with pytest.raises(SystemExit) as exc_info:
        ask.main()

    assert exc_info.value.code == 2
    assert "empty question" in capsys.readouterr().err


def test_main_exits_without_api_key(monkeypatch, capsys):
    monkeypatch.setattr(ask, "parse_args", lambda: _args(gemini_api_key=" "))

    with pytest.raises(SystemExit) as exc_info:
        ask.main()

    assert exc_info.value.code == 2
    assert "GEMINI_API_KEY is required" in capsys.readouterr().err


def test_main_exits_when_embedding_fails(monkeypatch, capsys):
    monkeypatch.setattr(ask, "parse_args", _args)

    def boom(*args, **kwargs):
        raise RuntimeError("no quota")

    monkeypatch.setattr(ask, "embed_query", boom)

    with pytest.raises(SystemExit) as exc_info:
        ask.main()

    assert exc_info.value.code == 2
    assert "failed to embed query: no quota" in capsys.readouterr().err


def test_main_exits_when_search_fails(monkeypatch, capsys):
    monkeypatch.setattr(ask, "parse_args", _args)
    monkeypatch.setattr(ask, "embed_query", lambda *a, **k: [0.1])
    monkeypatch.setattr(
        ask,
        "QdrantClient",
        lambda url, api_key: FakeQdrantClient(error=RuntimeError("qdrant down")),
    )

    with pytest.raises(SystemExit) as exc_info:
        ask.main()

    assert exc_info.value.code == 2
    assert "Qdrant search failed: qdrant down" in capsys.readouterr().err


def test_embed_query_reports_missing_dependency(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "google.generativeai":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="Missing dependency: google-generativeai"):
        ask.embed_query("key", "вопрос", "model")
