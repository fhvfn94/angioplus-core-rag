# -*- coding: utf-8 -*-
import types as _types

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


class _FakeGate:
    direct_answer = True


class _FakeLLM:
    def check_direct_answer(self, question, context):
        return _FakeGate()

    def generate_answer(self, question, context, system_prompt):
        return "Сгенерированный ответ."


def _search(vector, top_k):
    return [_FakeChunk(score=0.9, text="Контекстный фрагмент про AngioPlus Core.")]


def _run_ask(monkeypatch, request):
    # Make rag service appear ready and stub all external calls.
    rag._readiness["status"] = "ready"
    monkeypatch.setattr(rag, "embed_query", lambda q: [0.1, 0.2, 0.3])
    monkeypatch.setattr(rag, "search_qdrant", _search)
    monkeypatch.setattr(rag, "get_llm", lambda: _FakeLLM())
    return rag.ask(request)


def _make_request(question, conversation_id=None, user_id=None):
    return rag.AskRequest(
        question=question,
        top_k=5,
        conversation_id=conversation_id,
        user_id=user_id,
    )


def test_ask_without_ids_works_and_does_not_rewrite(monkeypatch):
    rag.conversation_memory._store.clear()
    request = _make_request("Какие системные требования AngioPlus Core?")
    resp = _run_ask(monkeypatch, request)
    assert resp.answer == "Сгенерированный ответ."
    assert resp.sources  # non-empty sources built from fake chunk


def test_follow_up_rewrite_uses_history(monkeypatch):
    monkeypatch.setattr(rag, "QUERY_CONTEXT_ENABLED", True)
    rag.conversation_memory._store.clear()
    req1 = _make_request(
        "Какие системные требования у AngioPlus Core?",
        conversation_id="c1",
        user_id="u1",
    )
    _run_ask(monkeypatch, req1)
    # Second, follow-up question.
    req2 = _make_request(
        "А как его у становить?",
        conversation_id="c1",
        user_id="u1",
    )
    resp = _run_ask(monkeypatch, req2)
    assert resp.answer == "Сгенерированный ответ."

    # History must have been used: the standalone question written to memory
    # for turn #2 is the rewritten one.
    snap = rag.conversation_memory.get_snapshot("c1", "u1")
    second = snap.turns[-1].standalone_question
    assert second == "Как установить AngioPlus Core?"


def test_secret_request_not_persisted(monkeypatch):
    rag.conversation_memory._store.clear()
    rag.conversation_memory.append_turn("c2", "u2", "Как войти в систему?", "Как войти в систему?")
    req = _make_request(
        "А какой пароль?",
        conversation_id="c2",
        user_id="u2",
    )
    resp = _run_ask(monkeypatch, req)
    assert resp.answer == rag.SECRET_REFUSAL
    assert resp.sources == []
    # Secret request must not have been stored.
    snap = rag.conversation_memory.get_snapshot("c2", "u2")
    assert len(snap.turns) == 1  # only the earlier safe turn remains
    assert snap.turns[-1].standalone_question == "Как войти в систему?"


def test_context_enabled_false_disables_rewrite(monkeypatch):
    rag.conversation_memory._store.clear()
    rag.conversation_memory.append_turn("c3", "u3", "Какие системные требования у AngioPlus Core?", "Какие системные требования у AngioPlus Core?")
    monkeypatch.setattr(rag, "QUERY_CONTEXT_ENABLED", False)
    monkeypatch.setattr(rag, "QUERY_NORMALIZATION_ENABLED", True)
    req = _make_request(
        "А как его у становить?",
        conversation_id="c3",
        user_id="u3",
    )
    resp = _run_ask(monkeypatch, req)
    assert resp.answer == "Сгенерированный ответ."
    # With context disabled, no new turn is appended.
    snap = rag.conversation_memory.get_snapshot("c3", "u3")
    assert len(snap.turns) == 1


def test_normalization_enabled_false_disables_stt_fix(monkeypatch):
    rag.conversation_memory._store.clear()
    monkeypatch.setattr(rag, "QUERY_NORMALIZATION_ENABLED", False)
    monkeypatch.setattr(rag, "QUERY_CONTEXT_ENABLED", False)
    req = _make_request("Кто такие Пульс Мьюзикл?")
    resp = _run_ask(monkeypatch, req)
    # It should still produce a normal answer (raw remains), no crash.
    assert resp.answer == "Сгенерированный ответ."


def test_default_flags_context_off_normalization_on():
    """Without env vars, context/rewrite are off, normalization is on.

    Spawn a clean subprocess with QUERY_* env removed so the module-level
    defaults are exercised, independent of the outer test environment.
    """
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items()
           if k not in ("QUERY_CONTEXT_ENABLED", "QUERY_NORMALIZATION_ENABLED")}
    code = (
        "import app.main as m;"
        "print(m.QUERY_NORMALIZATION_ENABLED);"
        "print(m.QUERY_CONTEXT_ENABLED)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines == ["True", "False"]


def test_normaization_reaches_embed_query(monkeypatch):
    """Prove normalization happens inside /ask: embed receives canonical text.

    Context must be disabled here, so only normalization (not rewrite) runs.
    """
    monkeypatch.setattr(rag, "QUERY_NORMALIZATION_ENABLED", True)
    monkeypatch.setattr(rag, "QUERY_CONTEXT_ENABLED", False)
    rag._readiness["status"] = "ready"
    rag.conversation_memory._store.clear()

    captured = {}

    def _embed(q):
        captured["query"] = q
        return [0.1, 0.2]

    monkeypatch.setattr(rag, "embed_query", _embed)
    monkeypatch.setattr(
        rag, "search_qdrant",
        lambda v, k: [_FakeChunk(score=0.9, text="Контекст про Pulse Medical.")],
    )
    monkeypatch.setattr(rag, "get_llm", lambda: _FakeLLM())

    # STT variant -> canonical domain term must reach embedding.
    rag.ask(_make_request("Кто такие Пульс Мьюзикл?"))
    assert captured["query"] == "Кто такие Pulse Medical?"

    # Word-gap fix on a short follow-up-shaped question must also reach
    # embedding (rewrite is disabled because context=false).
    captured.clear()
    rag.ask(_make_request("А как его у становить?"))
    assert captured["query"] == "А как его установить?"


def test_ambiguity_disables_rewrite(monkeypatch):
    monkeypatch.setattr(rag, "QUERY_CONTEXT_ENABLED", True)
    rag.conversation_memory._store.clear()
    rag.conversation_memory.append_turn(
        "c4", "u4",
        "Расскажи про AngioPlus Core и PStation",
        "Расскажи про AngioPlus Core и PStation",
    )
    req = _make_request(
        "А как его настроить?",
        conversation_id="c4",
        user_id="u4",
    )
    resp = _run_ask(monkeypatch, req)
    assert resp.answer == "Сгенерированный ответ."
    snap = rag.conversation_memory.get_snapshot("c4", "u4")
    # No rewrite: last stored standalone equals current (normalized) input.
    assert snap.turns[-1].standalone_question == "А как его настроить?"
