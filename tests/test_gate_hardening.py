# -*- coding: utf-8 -*-
"""Regression tests for DeepSeek direct-answer gate hardening.

Covers envelope normalization, strict schema enforcement, and the single
immediate retry on INVALID_RESPONSE (bounded by elapsed time). No real API
call is made; the OpenAI-compatible client is stubbed.
"""
from __future__ import annotations

import json

import pytest

import app.llm.deepseek_provider as prov
from app.llm.deepseek_provider import DeepSeekProvider
from app.llm.errors import LLMError, LLMErrorType
from app.llm.models import GateResult


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = _FakeMessage(content)
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, content, finish_reason="stop"):
        self.choices = [_FakeChoice(content, finish_reason)]


class _FakeCompletions:
    """Returns queued outcomes per create() call and records kwargs."""

    def __init__(self, queue):
        self.queue = list(queue)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.queue.pop(0)
        if item["kind"] == "raise":
            raise item["error"]
        return _FakeResponse(
            item["content"], item.get("finish_reason", "stop")
        )


class _FakeChat:
    def __init__(self, queue):
        self.completions = _FakeCompletions(queue)


class _FakeClient:
    def __init__(self, queue):
        self.chat = _FakeChat(queue)


# ---- response builders ---------------------------------------------------

def _ok(direct=True, reason="ok"):
    return {
        "kind": "ok",
        "content": json.dumps({"direct_answer": direct, "reason": reason}),
    }


def _fenced(direct=True, reason="ok"):
    body = json.dumps({"direct_answer": direct, "reason": reason})
    return {"kind": "ok", "content": "```json\n" + body + "\n```"}


def _whitespaced(direct=True, reason="ok"):
    body = json.dumps({"direct_answer": direct, "reason": reason})
    return {"kind": "ok", "content": "\n  " + body + "  \n"}


def _invalid(content):
    return {"kind": "ok", "content": content}


def _empty():
    return {"kind": "ok", "content": ""}


def _truncated():
    return {
        "kind": "ok",
        "content": '{"direct_answer": true',
        "finish_reason": "length",
    }


# ---- mocks ---------------------------------------------------------------

def _mock_provider(queue):
    provider = DeepSeekProvider(api_key="test-key", model="test-model")
    provider._client = _FakeClient(queue)  # override cached_property, no network
    return provider


def _gate_calls(provider):
    return provider._client.chat.completions.calls


def _slow_clock(monkeypatch):
    # First gate attempt appears to take 7.5 s (> 3.0 s budget) -> no retry.
    t = iter([0.0, 7.5])
    monkeypatch.setattr(prov, "perf_counter", lambda: next(t))


def _fast_clock(monkeypatch):
    # First gate attempt appears to take 1.5 s (<= 3.0 s budget) -> retryable.
    t = iter([0.0, 1.5])
    monkeypatch.setattr(prov, "perf_counter", lambda: next(t))


@pytest.fixture
def ask_args():
    return (
        "Какая это компания?",
        "Контекст: Shanghai Pulse Medical Technology, Inc.",
    )


def _run_gate(provider, question, context):
    return provider.check_direct_answer(question, context)


# ---- envelope normalization ----------------------------------------------

def test_valid_json_single_call(ask_args):
    provider = _mock_provider([_ok(True)])
    result = _run_gate(provider, *ask_args)
    assert isinstance(result, GateResult)
    assert result.direct_answer is True
    assert len(_gate_calls(provider)) == 1


def test_gate_false_single_call(ask_args):
    provider = _mock_provider([_ok(False)])
    result = _run_gate(provider, *ask_args)
    assert isinstance(result, GateResult)
    assert result.direct_answer is False
    assert len(_gate_calls(provider)) == 1


def test_fenced_json_single_call(ask_args):
    provider = _mock_provider([_fenced(True)])
    result = _run_gate(provider, *ask_args)
    assert result.direct_answer is True
    assert len(_gate_calls(provider)) == 1


def test_whitespace_json_single_call(ask_args):
    provider = _mock_provider([_whitespaced(False)])
    result = _run_gate(provider, *ask_args)
    assert result.direct_answer is False
    assert len(_gate_calls(provider)) == 1


# ---- strict schema still enforced (slow clock => no retry, 1 call) -------

def test_string_bool_invalid(ask_args, monkeypatch):
    _slow_clock(monkeypatch)
    provider = _mock_provider(
        [_invalid('{"direct_answer": "true", "reason": "ok"}')]
    )
    with pytest.raises(LLMError) as ei:
        _run_gate(provider, *ask_args)
    assert ei.value.error_type is LLMErrorType.INVALID_RESPONSE
    assert ei.value.message == "LLM returned a schema-invalid response"
    assert len(_gate_calls(provider)) == 1


def test_missing_reason_invalid(ask_args, monkeypatch):
    _slow_clock(monkeypatch)
    provider = _mock_provider([_invalid('{"direct_answer": true}')])
    with pytest.raises(LLMError) as ei:
        _run_gate(provider, *ask_args)
    assert ei.value.error_type is LLMErrorType.INVALID_RESPONSE
    assert len(_gate_calls(provider)) == 1


def test_extra_field_invalid(ask_args, monkeypatch):
    _slow_clock(monkeypatch)
    provider = _mock_provider(
        [_invalid('{"direct_answer": true, "reason": "ok", "x": 1}')]
    )
    with pytest.raises(LLMError) as ei:
        _run_gate(provider, *ask_args)
    assert ei.value.error_type is LLMErrorType.INVALID_RESPONSE
    assert len(_gate_calls(provider)) == 1


def test_malformed_json_invalid(ask_args, monkeypatch):
    _slow_clock(monkeypatch)
    provider = _mock_provider([_invalid("not json at all")])
    with pytest.raises(LLMError) as ei:
        _run_gate(provider, *ask_args)
    assert ei.value.error_type is LLMErrorType.INVALID_RESPONSE
    assert ei.value.message == "LLM returned malformed JSON"
    assert len(_gate_calls(provider)) == 1


def test_empty_response_invalid(ask_args, monkeypatch):
    _slow_clock(monkeypatch)
    provider = _mock_provider([_empty()])
    with pytest.raises(LLMError) as ei:
        _run_gate(provider, *ask_args)
    assert ei.value.error_type is LLMErrorType.INVALID_RESPONSE
    assert ei.value.message == "LLM returned an empty response"
    assert len(_gate_calls(provider)) == 1


def test_truncated_response_invalid(ask_args, monkeypatch):
    _slow_clock(monkeypatch)
    provider = _mock_provider([_truncated()])
    with pytest.raises(LLMError) as ei:
        _run_gate(provider, *ask_args)
    assert ei.value.error_type is LLMErrorType.INVALID_RESPONSE
    assert ei.value.message == "LLM response was truncated or filtered"
    assert len(_gate_calls(provider)) == 1


# ---- single immediate retry on INVALID_RESPONSE --------------------------

def test_fast_invalid_then_valid_retries_two_calls(ask_args, monkeypatch):
    _fast_clock(monkeypatch)
    provider = _mock_provider([_invalid("{bad json"), _ok(True)])
    result = _run_gate(provider, *ask_args)
    assert result.direct_answer is True
    assert len(_gate_calls(provider)) == 2


def test_fast_invalid_then_invalid_fails_two_calls(ask_args, monkeypatch):
    _fast_clock(monkeypatch)
    provider = _mock_provider([_invalid("{bad json"), _invalid("{also bad")])
    with pytest.raises(LLMError) as ei:
        _run_gate(provider, *ask_args)
    assert ei.value.error_type is LLMErrorType.INVALID_RESPONSE
    assert len(_gate_calls(provider)) == 2


def test_slow_invalid_does_not_retry(ask_args, monkeypatch):
    _slow_clock(monkeypatch)
    provider = _mock_provider([_invalid("{bad json")])
    with pytest.raises(LLMError) as ei:
        _run_gate(provider, *ask_args)
    assert ei.value.error_type is LLMErrorType.INVALID_RESPONSE
    assert len(_gate_calls(provider)) == 1


def test_non_invalid_response_does_not_retry(ask_args):
    err = LLMError(LLMErrorType.TIMEOUT_OR_NETWORK, "timeout", None)
    provider = _mock_provider([{"kind": "raise", "error": err}])
    with pytest.raises(LLMError) as ei:
        _run_gate(provider, *ask_args)
    assert ei.value.error_type is LLMErrorType.TIMEOUT_OR_NETWORK
    assert len(_gate_calls(provider)) == 1


# ---- thinking mode is preserved ------------------------------------------

def test_gate_still_sends_thinking_disabled(ask_args):
    provider = _mock_provider([_ok(True)])
    _run_gate(provider, *ask_args)
    captured = _gate_calls(provider)[0]
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}


def test_generation_still_sends_thinking_disabled():
    provider = _mock_provider([{"kind": "ok", "content": "ответ"}])
    provider.generate_answer(
        question="Вопрос",
        context="Контекст",
        system_prompt="SYSTEM",
    )
    captured = _gate_calls(provider)[0]
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
