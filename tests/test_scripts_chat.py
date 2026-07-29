# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from scripts import chat
from tests.helpers import make_point


class FakeQdrantClient:
    def __init__(self, points=None, **kwargs):
        self.points = points or []
        self.kwargs = kwargs
        self.calls: list[dict] = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
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
        or {"embedding": [0.3]},
    )

    class FakeModel:
        def __init__(self, model_name):
            calls["generative_model"] = model_name

        def generate_content(self, prompt):
            calls["prompt"] = prompt
            return SimpleNamespace(text="Ответ модели")

    monkeypatch.setattr(genai, "GenerativeModel", FakeModel)
    return calls


def test_embed_query(fake_genai):
    assert chat.embed_query("key", "вопрос", "model") == [0.3]
    assert fake_genai["task_type"] == "retrieval_query"


def test_search_qdrant_passes_parameters():
    points = [make_point()]
    client = FakeQdrantClient(points=points)

    assert chat.search_qdrant(client, "col", [0.1], 3) == points
    assert client.calls[0]["collection_name"] == "col"
    assert client.calls[0]["limit"] == 3


def test_build_context_numbers_chunks():
    chunks = [make_point(text="один"), SimpleNamespace(score=0.1, payload=None)]

    assert chat.build_context(chunks) == "[Chunk 1]\nодин\n\n[Chunk 2]\n"


def test_load_system_prompt_reads_file(monkeypatch, tmp_path):
    prompt = tmp_path / "prompt.md"
    prompt.write_text(" промпт ", encoding="utf-8")
    monkeypatch.setattr(chat, "SYSTEM_PROMPT_PATH", str(prompt))

    assert chat.load_system_prompt() == "промпт"


def test_load_system_prompt_falls_back(monkeypatch, tmp_path):
    monkeypatch.setattr(chat, "SYSTEM_PROMPT_PATH", str(tmp_path / "nope.md"))

    assert "AngioPlus Core" in chat.load_system_prompt()


def test_generate_answer_includes_prompt_parts(monkeypatch, fake_genai):
    monkeypatch.setattr(chat, "load_system_prompt", lambda: "SYSTEM")

    answer = chat.generate_answer("key", "вопрос", "контекст")

    assert answer == "Ответ модели"
    assert fake_genai["generative_model"] == chat.DEFAULT_GEMINI_MODEL
    assert "SYSTEM" in fake_genai["prompt"]
    assert "контекст" in fake_genai["prompt"]
    assert "вопрос" in fake_genai["prompt"]


def test_resolve_question_prefers_flag():
    args = argparse.Namespace(question="positional", question_flag="flag")

    assert chat.resolve_question(args) == "flag"


def test_resolve_question_uses_positional():
    args = argparse.Namespace(question="positional", question_flag=None)

    assert chat.resolve_question(args) == "positional"


def test_resolve_question_exits_without_question(capsys):
    args = argparse.Namespace(question=None, question_flag=None)

    with pytest.raises(SystemExit) as exc_info:
        chat.resolve_question(args)

    assert exc_info.value.code == 2
    assert "No question provided" in capsys.readouterr().err


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(chat.sys, "argv", ["chat.py", "вопрос"])

    args = chat.parse_args()

    assert args.question == "вопрос"
    assert args.top_k == 5
    assert args.collection == chat.DEFAULT_COLLECTION


def _args(**overrides) -> argparse.Namespace:
    defaults = {
        "question": "вопрос",
        "question_flag": None,
        "top_k": 3,
        "qdrant_url": "http://localhost:6333",
        "collection": "col",
        "gemini_api_key": "key",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_main_prints_answer_and_deduplicated_sources(monkeypatch, capsys):
    monkeypatch.setattr(chat, "parse_args", _args)
    monkeypatch.setattr(chat, "embed_query", lambda *a, **k: [0.1])
    points = [
        make_point(score=0.91, file_name="IFU.pdf"),
        make_point(score=0.90, file_name="IFU.pdf"),
        make_point(score=0.80, file_name="FAQ.xlsx"),
    ]
    monkeypatch.setattr(chat, "QdrantClient", lambda url: FakeQdrantClient(points))
    monkeypatch.setattr(chat, "generate_answer", lambda *a, **k: "Ответ")

    chat.main()

    out = capsys.readouterr().out
    assert "=== ANSWER ===" in out
    assert "Ответ" in out
    assert out.count("IFU.pdf") == 1
    assert "score 0.800" in out


def test_main_limits_sources_to_three(monkeypatch, capsys):
    monkeypatch.setattr(chat, "parse_args", _args)
    monkeypatch.setattr(chat, "embed_query", lambda *a, **k: [0.1])
    points = [
        make_point(score=0.9, file_name=f"doc{i}.pdf", section=f"section {i}")
        for i in range(5)
    ]
    monkeypatch.setattr(chat, "QdrantClient", lambda url: FakeQdrantClient(points))
    monkeypatch.setattr(chat, "generate_answer", lambda *a, **k: "Ответ")

    chat.main()

    out = capsys.readouterr().out
    assert sum(f"doc{i}.pdf" in out for i in range(5)) == 3


def test_main_exits_without_api_key(monkeypatch, capsys):
    monkeypatch.setattr(chat, "parse_args", lambda: _args(gemini_api_key=None))

    with pytest.raises(SystemExit) as exc_info:
        chat.main()

    assert exc_info.value.code == 2
    assert "GEMINI_API_KEY required" in capsys.readouterr().err
