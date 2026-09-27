# -*- coding: utf-8 -*-
"""Slice 11B.1b.4d -- offline proofs for the pure Global Semantic Policy.

PURE: no network, no DB, no LLM, no Procedure, no Knowledge, no dispatcher.
Compact behavioral coverage of ``support/global_policy.py``:

  * closed 3-value action vocabulary;
  * precedence OUT_OF_SCOPE -> human_requested -> CONTINUE;
  * only ``scope`` + ``human_requested`` are read;
  * RESOLVED / new_issue / request_kind never produce a global terminal;
  * the fail-safe observation (scope=UNCLEAR) still CONTINUEs;
  * no I/O / engine imports and no runtime wiring.

The ``RESOLVED`` + ``procedure is None`` interaction is deliberately NOT
tested here: it is an 11B.2 concern (see the module docstring).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from support.global_policy import GlobalAction, evaluate_global_policy
from support.observation import (
    ObservationScope,
    ProblemStatus,
    ReportedAction,
    RequestKind,
    SemanticObservation,
)


# ============================================================
# HELPERS
# ============================================================

def _obs(
    *,
    scope: ObservationScope = ObservationScope.IN_SCOPE,
    human_requested: bool = False,
    problem_status=None,
    request_kind: RequestKind = RequestKind.PROBLEM,
    new_issue=None,
    actions: tuple = (),
) -> SemanticObservation:
    return SemanticObservation(
        scope=scope,
        reported_actions=tuple(actions),
        problem_status=problem_status,
        human_requested=human_requested,
        clarification_facts=(),
        new_issue=new_issue,
        request_kind=request_kind,
    )


def _fail_safe() -> SemanticObservation:
    return _obs(
        scope=ObservationScope.UNCLEAR,
        request_kind=RequestKind.UNCLEAR,
        actions=(ReportedAction(step_id=None, outcome=None),),
    )


# ============================================================
# 1. CLOSED VOCABULARY
# ============================================================

def test_global_action_is_closed_to_exactly_three_values():
    assert set(GlobalAction) == {
        GlobalAction.CONTINUE,
        GlobalAction.ESCALATE,
        GlobalAction.OUT_OF_SCOPE,
    }
    assert {a.value for a in GlobalAction} == {
        "continue",
        "escalate",
        "out_of_scope",
    }


# ============================================================
# 2. PRECEDENCE: OUT_OF_SCOPE -> human_requested -> CONTINUE
# ============================================================

def test_out_of_scope_is_terminal():
    assert (
        evaluate_global_policy(_obs(scope=ObservationScope.OUT_OF_SCOPE))
        is GlobalAction.OUT_OF_SCOPE
    )


def test_out_of_scope_wins_over_human_request():
    assert (
        evaluate_global_policy(
            _obs(scope=ObservationScope.OUT_OF_SCOPE, human_requested=True)
        )
        is GlobalAction.OUT_OF_SCOPE
    )


def test_human_request_is_terminal():
    assert (
        evaluate_global_policy(_obs(human_requested=True))
        is GlobalAction.ESCALATE
    )


@pytest.mark.parametrize(
    "request_kind",
    [RequestKind.PROBLEM, RequestKind.INFORMATION, RequestKind.UNCLEAR],
)
def test_human_request_escalates_regardless_of_request_kind(request_kind):
    # Cannot fall through to Knowledge QA / Clarification.
    assert (
        evaluate_global_policy(
            _obs(human_requested=True, request_kind=request_kind)
        )
        is GlobalAction.ESCALATE
    )


@pytest.mark.parametrize(
    "request_kind",
    [RequestKind.PROBLEM, RequestKind.INFORMATION, RequestKind.UNCLEAR],
)
def test_out_of_scope_declines_regardless_of_request_kind(request_kind):
    # Cannot fall through to family dispatch.
    assert (
        evaluate_global_policy(
            _obs(scope=ObservationScope.OUT_OF_SCOPE, request_kind=request_kind)
        )
        is GlobalAction.OUT_OF_SCOPE
    )


# ============================================================
# 3. ORDINARY TURNS -> CONTINUE
# ============================================================

@pytest.mark.parametrize(
    "scope, request_kind",
    [
        (ObservationScope.IN_SCOPE, RequestKind.PROBLEM),
        (ObservationScope.IN_SCOPE, RequestKind.INFORMATION),
        (ObservationScope.UNCLEAR, RequestKind.UNCLEAR),
    ],
)
def test_ordinary_turns_continue(scope, request_kind):
    assert (
        evaluate_global_policy(_obs(scope=scope, request_kind=request_kind))
        is GlobalAction.CONTINUE
    )


def test_unclear_scope_is_not_out_of_scope_and_continues():
    # The analyzer fail-safe observation must still reach family dispatch.
    assert evaluate_global_policy(_fail_safe()) is GlobalAction.CONTINUE


# ============================================================
# 4. PROCEDURE-SPECIFIC FACTS NEVER TERMINATE GLOBALLY
# ============================================================

@pytest.mark.parametrize("request_kind", [RequestKind.INFORMATION, RequestKind.PROBLEM])
def test_resolved_does_not_terminate_globally(request_kind):
    # "Спасибо, теперь всё работает. А какая версия Windows?" must be answered,
    # not globally resolved. RESOLVED stays a procedure-family concern.
    assert (
        evaluate_global_policy(
            _obs(problem_status=ProblemStatus.RESOLVED, request_kind=request_kind)
        )
        is GlobalAction.CONTINUE
    )


def test_new_issue_does_not_terminate_globally():
    assert (
        evaluate_global_policy(
            _obs(new_issue="screen flicker", request_kind=RequestKind.INFORMATION)
        )
        is GlobalAction.CONTINUE
    )


# ============================================================
# 5. READS ONLY scope + human_requested
# ============================================================

def test_policy_reads_only_scope_and_human_requested():
    names = set(evaluate_global_policy.__code__.co_names)
    assert names == {
        "ObservationScope",
        "OUT_OF_SCOPE",
        "scope",
        "human_requested",
        "GlobalAction",
        "ESCALATE",
        "CONTINUE",
    }


def test_policy_is_deterministic_and_pure():
    observation = _obs(human_requested=True)
    first = evaluate_global_policy(observation)
    second = evaluate_global_policy(observation)
    assert first == second is GlobalAction.ESCALATE
    assert observation.human_requested is True  # not mutated


# ============================================================
# 6. NO I/O / ENGINE IMPORTS, NOT WIRED REQUIRED
# ============================================================

def test_module_imports_only_observation_and_stdlib():
    import support.global_policy as gp

    tree = ast.parse(Path(gp.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported == {"__future__", "enum", "support.observation"}


def test_module_is_wired_only_into_the_dispatch_boundary():
    root = Path(__file__).resolve().parents[1]

    # Slice 11B.2: the dispatcher APPLIES global policy (after the ONE
    # analysis, before family selection); SupportCore references the closed
    # action vocabulary while projecting the dispatcher's typed results. These
    # are the ONLY two sanctioned, non-runtime integration points.
    dispatch = (root / "support/turn_dispatch.py").read_text(encoding="utf-8")
    assert "global_policy" in dispatch
    assert "GlobalAction" in dispatch
    assert "evaluate_global_policy" in dispatch

    core = (root / "support/support_core.py").read_text(encoding="utf-8")
    assert "GlobalAction" in core

    # Procedure-family siblings, the legacy Router and EVERY runtime path stay
    # UNWIRED: the legacy Router gates 1-2 are retired only when the legacy
    # ``execute_persisted_turn`` / ``process_support_turn`` path is retired.
    for relative in (
        "support/router.py",
        "support/orchestrator.py",
        "support/turn_executor.py",
        "support/turn_family.py",
        "yandex/chat/server.py",
        "yandex/chat/alice_support_agent.py",
    ):
        text = (root / relative).read_text(encoding="utf-8")
        assert "global_policy" not in text, relative
        assert "GlobalAction" not in text, relative
        assert "evaluate_global_policy" not in text, relative
