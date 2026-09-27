# -*- coding: utf-8 -*-
"""Slice 11B.1b.4b -- deterministic CLARIFICATION responder (PURE, offline).

The responder owns exactly ONE neutral prompt: ask the user to clarify.
No LLM/provider, no network, no DB, no procedure/knowledge/router, no
TurnFamily. It does not decide WHEN it is called.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from support import clarification as cl
from support.clarification import ClarificationResult, clarify


def test_returns_non_blank_user_facing_text():
    result = clarify()
    assert isinstance(result, ClarificationResult)
    assert isinstance(result.text, str)
    assert result.text.strip()


def test_is_deterministic_and_has_exactly_one_field():
    a = clarify()
    b = clarify()
    assert a == b
    assert a.text == b.text
    # One field only: no status / reason / provider_called.
    assert set(ClarificationResult.__dataclass_fields__) == {"text"}


def test_result_rejects_blank_text():
    for bad in ("", "   ", None):
        with pytest.raises(ValueError):
            ClarificationResult(text=bad)


def test_wording_makes_no_escalation_or_error_claim():
    text = clarify().text.lower()
    forbidden = (
        "специалист",
        "оператор",
        "переда",
        "обращени",
        "заявк",
        "эскал",
        "не могу",
        "не удалось",
        "вне ",
        "ошибк",
        "сбой",
        "operator",
        "handoff",
        "escalat",
        "ticket",
    )
    for token in forbidden:
        assert token not in text, token


def test_module_has_no_external_dependency():
    tree = ast.parse(Path(cl.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported <= {"__future__", "dataclasses"}
    assert set(cl.__all__) == {"ClarificationResult", "clarify"}


def test_responder_is_wired_only_into_the_application_boundary():
    root = Path(cl.__file__).resolve().parents[1]

    # Slice 11B.2: the dispatcher CALLS ``clarify()``; SupportCore references
    # the result type while projecting the CLARIFICATION family outcome.
    dispatch = (root / "support/turn_dispatch.py").read_text(encoding="utf-8")
    assert "support.clarification" in dispatch
    assert "ClarificationResult" in dispatch

    core = (root / "support/support_core.py").read_text(encoding="utf-8")
    assert "support.clarification" in core
    assert "ClarificationResult" in core

    # Procedure-family siblings and EVERY runtime path stay UNWIRED.
    for rel in (
        "support/turn_family.py",
        "support/orchestrator.py",
        "support/turn_executor.py",
        "support/router.py",
        "support/knowledge_qa.py",
        "yandex/chat/server.py",
        "yandex/chat/alice_support_agent.py",
    ):
        text = (root / rel).read_text(encoding="utf-8")
        assert "support.clarification" not in text, rel
        assert "ClarificationResult" not in text, rel
