# -*- coding: utf-8 -*-
"""Slice 11B.1b.1 -- RequestKind semantic-fact extension (PURE, offline).

Pins the ONE new semantic dimension added to the canonical observation
contract: ``RequestKind`` -- is the current user message primarily
reporting/seeking help with a problem, asking for factual/product/
documentation information, or semantically unclear?

Scope guards (this is a deliberately SMALL contract extension):
  * ``request_kind`` is a CLOSED semantic FACT, not a routing decision;
  * the analyzer parser/schema accept it, fail closed to ``UNCLEAR``;
  * the Yandex structured schema requires it as a closed 3-value enum;
  * NO TurnFamily / Knowledge-QA / runtime wiring is introduced.

No network, no DB, no LLM, no cloud.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from support import alice_analyzer as az
from support import alice_analyzer_yandex as ay
from support.observation import (
    RequestKind,
    SemanticObservation,
)


# ================================================================
# 1. CLOSED ENUM
# ================================================================

def test_request_kind_is_closed_to_exactly_three_values():
    assert set(RequestKind) == {
        RequestKind.PROBLEM,
        RequestKind.INFORMATION,
        RequestKind.UNCLEAR,
    }
    assert {k.value for k in RequestKind} == {"problem", "information", "unclear"}


def test_request_kind_is_present_on_the_observation_contract():
    assert "request_kind" in SemanticObservation.__dataclass_fields__


# ================================================================
# 2. PARSER -- each valid value round-trips
# ================================================================

def _raw(request_kind, **overrides) -> str:
    payload = {
        "scope": "in_scope",
        "reported_actions": [],
        "problem_status": None,
        "human_requested": False,
        "clarification_facts": [],
        "new_issue": None,
        "request_kind": request_kind,
    }
    payload.update(overrides)
    return json.dumps(payload)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("problem", RequestKind.PROBLEM),
        ("information", RequestKind.INFORMATION),
        ("unclear", RequestKind.UNCLEAR),
    ],
)
def test_valid_request_kind_parses(value, expected):
    obs = az.parse_semantic_output(_raw(value))
    assert obs.request_kind is expected


# ================================================================
# 3. PARSER -- strict rejection + fail-safe
# ================================================================

def _assert_fail_safe(obs):
    assert obs == az._fail_safe_observation()


def test_missing_request_kind_is_rejected():
    raw = json.dumps(
        {
            "scope": "in_scope",
            "reported_actions": [],
            "problem_status": None,
            "human_requested": False,
            "clarification_facts": [],
            "new_issue": None,
            # request_kind absent
        }
    )
    _assert_fail_safe(az.parse_semantic_output(raw))


@pytest.mark.parametrize("bad", ["question", "PROBLEM", "info", "", "help"])
def test_unknown_request_kind_is_rejected(bad):
    _assert_fail_safe(az.parse_semantic_output(_raw(bad)))


@pytest.mark.parametrize("bad", [3, True, None, ["problem"], {"k": "problem"}])
def test_wrong_type_request_kind_is_rejected(bad):
    _assert_fail_safe(az.parse_semantic_output(_raw(bad)))


def test_malformed_json_sets_request_kind_unclear():
    obs = az.parse_semantic_output("{ not json at all")
    assert obs.request_kind is RequestKind.UNCLEAR
    _assert_fail_safe(obs)


def test_provider_failure_sets_request_kind_unclear():
    def _boom(prompt, history):
        raise RuntimeError("transport down")

    obs = az.analyze(message="x", history=[], procedure=None, provider=_boom)
    assert obs.request_kind is RequestKind.UNCLEAR
    _assert_fail_safe(obs)


def test_information_is_never_inferred_from_absent_problem_status():
    # An explicit "problem" fact with NO problem_status must stay PROBLEM.
    obs = az.parse_semantic_output(_raw("problem", problem_status=None))
    assert obs.request_kind is RequestKind.PROBLEM
    assert obs.problem_status is None


# ================================================================
# 4. REPRESENTATIVE MESSAGE -> FACT (wire-level, injected provider)
# ================================================================
# The semantic classification itself is model-owned (Alice); the offline
# contract is that the model's closed fact is surfaced verbatim. The
# real-model semantic assertions live in the opt-in cloud suites.

_INFORMATION_QUESTION = "Какие версии Windows поддерживает AngioPlus?"
_PROBLEM_REPORTS = (
    "PACS не подключается",
    "AngioPlus не запускается на Windows 11",
)


def _provider_returning(payload: str):
    def _p(prompt, history):
        return payload
    return _p


def test_information_question_surfaces_information():
    # INFORMATION: "Какие версии Windows поддерживает AngioPlus?"
    obs = az.analyze(
        message=_INFORMATION_QUESTION,
        history=[],
        procedure=None,
        provider=_provider_returning(_raw("information")),
    )
    assert obs.request_kind is RequestKind.INFORMATION


@pytest.mark.parametrize("message", _PROBLEM_REPORTS)
def test_problem_reports_surface_problem(message):
    obs = az.analyze(
        message=message,
        history=[],
        procedure=None,
        provider=_provider_returning(_raw("problem")),
    )
    assert obs.request_kind is RequestKind.PROBLEM


def test_ambiguous_input_can_surface_unclear():
    obs = az.analyze(
        message="Не работает",
        history=[],
        procedure=None,
        provider=_provider_returning(_raw("unclear")),
    )
    assert obs.request_kind is RequestKind.UNCLEAR


def test_prompt_states_request_kind_is_a_fact_not_a_decision():
    prompt = az.build_analyzer_prompt(message="PACS не подключается")
    assert "request_kind" in prompt
    # The three closed values are named in the rules.
    assert "problem" in prompt and "information" in prompt and "unclear" in prompt
    # The examples the model must follow are present.
    assert "Какие версии Windows поддерживает AngioPlus?" in prompt
    # It is explicitly forbidden to encode routing/engine choices in it.
    assert "НЕ решение о маршрутизации" in prompt


def test_prompt_does_not_introduce_new_vocabulary():
    prompt = az.build_analyzer_prompt(message="x")
    for forbidden in ("TurnFamily", "knowledge_qa", "answerable", "route_action"):
        assert forbidden not in prompt


# ================================================================
# 5. YANDEX STRUCTURED SCHEMA -- required + exactly three enum values
# ================================================================

def test_yandex_schema_requires_request_kind_with_exactly_three_values():
    fmt = ay._semantic_format()
    schema = fmt["schema"]

    assert schema["additionalProperties"] is False
    assert "request_kind" in schema["required"]

    request_kind = schema["properties"]["request_kind"]
    assert request_kind["type"] == "string"
    assert request_kind["enum"] == ["problem", "information", "unclear"]
    assert len(request_kind["enum"]) == 3

    # The required set is exactly the closed field set (no drift).
    assert set(schema["required"]) == set(schema["properties"])
    assert "request_kind" in set(schema["properties"])


# ================================================================
# 6. SCOPE GUARDS -- no TurnFamily / Knowledge-QA / wiring in this slice
# ================================================================

def test_no_turn_family_or_knowledge_qa_symbols_introduced():
    for module in (az, ay):
        source = Path(module.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "TurnFamily",
            "KnowledgeQA",
            "KnowledgeQa",
            "RuntimeEngine",
        ):
            assert forbidden not in source, (module.__name__, forbidden)


def test_observation_module_has_no_routing_symbols():
    import ast

    import support.observation as observation_mod

    tree = ast.parse(
        Path(observation_mod.__file__).read_text(encoding="utf-8")
    )

    # Scan CODE only: identifiers / attribute names / string literals.
    parts: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            parts.append(node.id)
        elif isinstance(node, ast.Attribute):
            parts.append(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            parts.append(node.value)
    code = " ".join(parts)

    for forbidden in (
        "TurnFamily",
        "RouteAction",
        "KnowledgeQA",
        "RoutingDecision",
    ):
        assert forbidden not in code, forbidden
