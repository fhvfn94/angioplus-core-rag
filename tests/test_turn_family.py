# -*- coding: utf-8 -*-
"""Slice 11B.1b.2 -- deterministic TurnFamily selector (PURE, offline).

    RequestKind.PROBLEM      -> TurnFamily.PROCEDURE_SUPPORT
    RequestKind.INFORMATION  -> TurnFamily.KNOWLEDGE_QA
    RequestKind.UNCLEAR      -> TurnFamily.CLARIFICATION

The selector reads exactly ONE input dimension (observation.request_kind).
No network, no DB, no LLM, no cloud.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from support import turn_family as tf
from support.observation import (
    ObservationScope,
    ProblemStatus,
    ReportedAction,
    RequestKind,
    SemanticObservation,
    StepOutcome,
)
from support.turn_family import TurnFamily, select_turn_family


def _obs(request_kind, **overrides) -> SemanticObservation:
    fields = dict(
        scope=ObservationScope.IN_SCOPE,
        reported_actions=(),
        problem_status=None,
        human_requested=False,
        clarification_facts=(),
        new_issue=None,
    )
    fields.update(overrides)
    return SemanticObservation(request_kind=request_kind, **fields)


_EXPECTED = {
    RequestKind.PROBLEM: TurnFamily.PROCEDURE_SUPPORT,
    RequestKind.INFORMATION: TurnFamily.KNOWLEDGE_QA,
    RequestKind.UNCLEAR: TurnFamily.CLARIFICATION,
}


def test_enum_is_exactly_three_values():
    assert set(TurnFamily) == {
        TurnFamily.PROCEDURE_SUPPORT,
        TurnFamily.KNOWLEDGE_QA,
        TurnFamily.CLARIFICATION,
    }
    assert {m.value for m in TurnFamily} == {
        "procedure_support",
        "knowledge_qa",
        "clarification",
    }


def test_mapping_table_is_exhaustive():
    assert set(_EXPECTED) == set(RequestKind)


@pytest.mark.parametrize("request_kind, expected", list(_EXPECTED.items()))
def test_mapping_is_exact(request_kind, expected):
    assert select_turn_family(_obs(request_kind)) is expected


def test_other_observation_fields_do_not_affect_the_result():
    a = _obs(RequestKind.INFORMATION)
    b = _obs(
        RequestKind.INFORMATION,
        scope=ObservationScope.OUT_OF_SCOPE,
        reported_actions=(
            ReportedAction(step_id="STEP_A", outcome=StepOutcome.FAILED),
        ),
        problem_status=ProblemStatus.ONGOING,
        human_requested=True,
        clarification_facts=("model=X",),
        new_issue="other problem",
    )
    assert select_turn_family(a) is TurnFamily.KNOWLEDGE_QA
    assert select_turn_family(b) is TurnFamily.KNOWLEDGE_QA


def test_unclear_and_analyzer_fail_safe_map_to_clarification():
    from support import alice_analyzer as az

    assert (
        select_turn_family(_obs(RequestKind.UNCLEAR))
        is TurnFamily.CLARIFICATION
    )
    assert (
        select_turn_family(az._fail_safe_observation())
        is TurnFamily.CLARIFICATION
    )
    assert (
        select_turn_family(az.parse_semantic_output("{ not json"))
        is TurnFamily.CLARIFICATION
    )


def test_module_imports_only_enum_and_observation():
    tree = ast.parse(Path(tf.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported <= {"__future__", "enum", "support.observation"}
    assert not hasattr(tf, "TurnFamilyDecision")
    assert set(tf.__all__) == {"TurnFamily", "select_turn_family"}


def test_selector_is_wired_only_into_the_application_boundary():
    root = Path(tf.__file__).resolve().parents[1]

    # Slice 11B.2: the dispatcher OWNS family selection
    # (``select_turn_family``); SupportCore references the closed family
    # vocabulary while projecting the dispatcher's typed results.
    for rel in ("support/turn_dispatch.py", "support/support_core.py"):
        text = (root / rel).read_text(encoding="utf-8")
        assert "turn_family" in text, rel
        assert "TurnFamily" in text, rel

    # EVERY runtime path stays UNWIRED: this is a later slice.
    for rel in (
        "yandex/chat/server.py",
        "yandex/chat/alice_support_agent.py",
    ):
        text = (root / rel).read_text(encoding="utf-8")
        assert "turn_family" not in text, rel
        assert "TurnFamily" not in text, rel
