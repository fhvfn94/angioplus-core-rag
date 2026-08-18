# -*- coding: utf-8 -*-
"""Regression tests for DeepSeek thinking-mode configuration.

Goal: explicitly disable DeepSeek thinking mode for normal answer generation
so that a hidden reasoning phase cannot stall /ask beyond the TWIN 10-second
timeout. The direct-answer gate already disables thinking; these tests prove:

- generate_answer sends extra_body={"thinking": {"type": "disabled"}}
- check_direct_answer (the gate) still sends the same disabled setting

No real API call is made; the OpenAI-compatible client is stubbed.
"""
from __future__ import annotations

from app.llm.deepseek_provider import DeepSeekProvider
from app.llm.models import GateResult

_THINKING_DISABLED = {"thinking": {"type": "disabled"}}


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
    def __init__(self):
        self.captured = None

    def create(self, **kwargs):
        self.captured = kwargs
        # generate_answer path returns a natural-language string.
        if any(
            m.get("role") == "system"
            for m in kwargs.get("messages", [])
        ):
            return _FakeResponse("Сгенерированный ответ.")
        # gate path returns valid GateResult JSON.
        return _FakeResponse('{"direct_answer": true, "reason": "ok"}')


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self):
        self.chat = _FakeChat()


def _provider() -> tuple[DeepSeekProvider, _FakeClient]:
    provider = DeepSeekProvider(api_key="test-key", model="test-model")
    fake = _FakeClient()
    provider._client = fake  # override cached_property, no network
    return provider, fake


def test_generate_answer_disables_thinking():
    provider, fake = _provider()

    provider.generate_answer(
        question="Какие системные требования?",
        context="Контекст про AngioPlus Core.",
        system_prompt="SYSTEM",
    )

    captured = fake.chat.completions.captured
    assert captured is not None
    # Only one LLM call must have been issued by generate_answer.
    assert captured["extra_body"] == _THINKING_DISABLED


def test_check_direct_answer_still_disables_thinking():
    provider, fake = _provider()

    result = provider.check_direct_answer(
        question="Как установить AngioPlus Core?",
        context="Установка выполняется сертифицированным инженером.",
    )

    assert isinstance(result, GateResult)
    captured = fake.chat.completions.captured
    assert captured is not None
    assert captured["extra_body"] == _THINKING_DISABLED
