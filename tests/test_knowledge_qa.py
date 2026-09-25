# -*- coding: utf-8 -*-
"""Slice 11B.1b.3 -- Knowledge QA engine (PURE, offline).

Pins the small, architecture-grounded QA contract:

    retrieve_general -> [no evidence => NOT_ANSWERED, provider NOT called]
                     -> provider(question + evidence)
                     -> strict {"answer": "..."} parse
                     -> ANSWERED | NOT_ANSWERED
    infra failure    -> INTERNAL_ERROR

No lexical grounding engine, no regex policy lists, no scoring. No network,
no DB, no LLM, no cloud.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from support import knowledge_qa as kq
from support.knowledge import KnowledgeItem
from support.knowledge_qa import (
    KnowledgeQAStatus,
    answer_question,
    parse_qa_output,
)


# ----------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------

def _item(text="AngioPlus Core supports Windows 10 and Windows 11."):
    return KnowledgeItem(text=text)


class FakeBackend:
    """In-memory KnowledgeBackend. Records calls; optional canned failure."""

    def __init__(self, items=(), exc=None):
        self._items = list(items)
        self._exc = exc
        self.calls: list[tuple[str, int]] = []

    def search(self, query, *, max_num_results):
        self.calls.append((query, max_num_results))
        if self._exc is not None:
            raise self._exc
        return list(self._items)


class FakeProvider:
    """Records prompts; returns canned output or raises."""

    def __init__(self, output="", exc=None):
        self._output = output
        self._exc = exc
        self.prompts: list[str] = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if self._exc is not None:
            raise self._exc
        return self._output


def _json(answer):
    return json.dumps({"answer": answer})


# ================================================================
# 1. CLOSED VOCABULARY + RESULT INVARIANTS
# ================================================================

def test_status_is_closed_three_value_enum():
    assert {s.value for s in KnowledgeQAStatus} == {
        "answered",
        "not_answered",
        "internal_error",
    }


def test_result_requires_non_blank_text_and_is_immutable():
    with pytest.raises(ValueError):
        kq.KnowledgeQAResult(status=KnowledgeQAStatus.ANSWERED, text="   ")
    res = kq.KnowledgeQAResult(status=KnowledgeQAStatus.ANSWERED, text="ok")
    with pytest.raises(FrozenInstanceError):
        res.text = "x"  # type: ignore[misc]


# ================================================================
# 2. ANSWERED PATH
# ================================================================

def test_found_evidence_yields_answered():
    backend = FakeBackend(items=[_item()])
    provider = FakeProvider(output=_json("Windows 10 и Windows 11."))

    result = answer_question(
        "Какие версии Windows поддерживаются?", backend=backend, provider=provider
    )

    assert result.status is KnowledgeQAStatus.ANSWERED
    assert result.text == "Windows 10 и Windows 11."
    assert len(provider.prompts) == 1


def test_provider_receives_only_question_and_evidence():
    backend = FakeBackend(items=[_item("Evidence marker: PACS DICOM.")])
    provider = FakeProvider(output=_json("PACS — это система."))

    answer_question("Что такое PACS?", backend=backend, provider=provider)

    prompt = provider.prompts[0]
    assert "Что такое PACS?" in prompt
    assert "Evidence marker: PACS DICOM." in prompt
    assert '"answer"' in prompt


# ================================================================
# 3. NO EVIDENCE -> NOT_ANSWERED, PROVIDER NOT CALLED
# ================================================================

def test_no_evidence_is_not_answered_and_provider_not_called():
    backend = FakeBackend(items=[])
    provider = FakeProvider(output=_json("should never be used"))

    result = answer_question("Что такое PACS?", backend=backend, provider=provider)

    assert result.status is KnowledgeQAStatus.NOT_ANSWERED
    assert result.reason == "no_usable_evidence"
    assert provider.prompts == []


def test_blank_question_short_circuits_without_backend_or_provider_call():
    backend = FakeBackend(items=[_item()])
    provider = FakeProvider(output=_json("x"))

    result = answer_question("   ", backend=backend, provider=provider)

    assert result.status is KnowledgeQAStatus.NOT_ANSWERED
    assert backend.calls == []
    assert provider.prompts == []


# ================================================================
# 4. INFRASTRUCTURE FAILURE -> INTERNAL_ERROR
# ================================================================

def test_retrieval_error_is_internal_error_and_provider_not_called():
    backend = FakeBackend(exc=RuntimeError("transport down"))
    provider = FakeProvider(output=_json("x"))

    result = answer_question("q", backend=backend, provider=provider)

    assert result.status is KnowledgeQAStatus.INTERNAL_ERROR
    assert result.reason == "retrieval_error"
    assert provider.prompts == []


def test_provider_exception_is_internal_error():
    backend = FakeBackend(items=[_item()])
    provider = FakeProvider(exc=RuntimeError("429"))

    result = answer_question("q", backend=backend, provider=provider)

    assert result.status is KnowledgeQAStatus.INTERNAL_ERROR
    assert result.reason == "responder_error"


# ================================================================
# 5. STRICT OUTPUT CONTRACT -> NOT_ANSWERED
# ================================================================

@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "not json at all",
        "{broken",
        "[]",
        '"just a string"',
        '{"other": "x"}',
        '{"answer": 42}',
        '{"answer": ""}',
        '{"answer": "  "}',
        '{"answer": "ok", "extra": 1}',
    ],
)
def test_malformed_or_blank_output_is_not_answered(raw):
    backend = FakeBackend(items=[_item()])
    provider = FakeProvider(output=raw)

    result = answer_question("q", backend=backend, provider=provider)

    assert result.status is KnowledgeQAStatus.NOT_ANSWERED
    assert result.reason == "answer_rejected"


def test_parse_accepts_exactly_one_answer_field():
    assert parse_qa_output('{"answer": "ok"}') == "ok"
    assert parse_qa_output('{"answer": "ok", "x": 1}') is None
    assert parse_qa_output("[1,2,3]") is None


# ================================================================
# 6. NEUTRAL NON-ANSWERS (no escalation / handoff claim)
# ================================================================

def test_non_answered_texts_are_neutral_and_non_blank():
    backend_none = FakeBackend(items=[])
    backend_err = FakeBackend(exc=RuntimeError("x"))
    provider = FakeProvider(output="")

    no_ev = answer_question("q", backend=backend_none, provider=provider)
    err = answer_question("q", backend=backend_err, provider=provider)

    for res in (no_ev, err):
        assert res.text.strip()
        low = res.text.lower()
        for forbidden in ("передаю", "специалист", "поддержк", "contact", "escalat"):
            assert forbidden not in low, forbidden


# ================================================================
# 7. DETERMINISM
# ================================================================

def test_repeated_execution_is_deterministic():
    backend = FakeBackend(items=[_item()])
    provider = FakeProvider(output=_json("Windows 10 и Windows 11."))

    first = answer_question("q", backend=backend, provider=provider)
    second = answer_question("q", backend=backend, provider=provider)

    assert (first.status, first.text) == (second.status, second.text)
