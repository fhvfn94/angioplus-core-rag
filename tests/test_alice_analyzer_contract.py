# -*- coding: utf-8 -*-
"""Contract tests for support.alice_analyzer (PURE: no DB, no network, no LLM).

Covers the approved Slice 4 matrix, including the two final corrections:

  * CORRECTION 1 -- the parser does NO semantic attribution. An ambiguous
    reference not bound by the model is preserved verbatim as
    ReportedAction(step_id=None, ...); the parser never binds "не помогло"
    to last_recommended_step itself.
  * CORRECTION 2 -- STRICT schema. Malformed JSON, missing fields, EXTRA
    fields, wrong types and invalid enums are all hard rejections (no
    silent drift), converging on the single fail-safe observation.

It also pins the cross-layer invariants against the FROZEN Slice 2 reducer
and Slice 3 router, plus the anti-regression guards (no legacy cascade
vocabulary, no File Search, no production/LLM imports).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from support import alice_analyzer as az
from support.observation import (
    ObservationScope,
    ProblemStatus,
    ReportedAction,
    StepOutcome,
)
from support.procedure import load_procedure
from support.procedure_state import initial_state, update_procedure_state
from support.router import ClarifyReason, RouteAction, route


FIXTURE = (
    Path(__file__).parent / "fixtures" / "procedures" / "test_procedure.yaml"
)
BLACK_SCREEN = (
    Path(__file__).parent / "fixtures" / "procedures" / "black_screen.yaml"
)


@pytest.fixture
def procedure():
    return load_procedure(FIXTURE)


@pytest.fixture
def allowed(procedure):
    return az.build_allowed_steps(procedure)


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def make_raw(
    *,
    scope="in_scope",
    actions=(),
    problem_status=None,
    human_requested=False,
    facts=(),
    new_issue=None,
    **extra,
) -> str:
    payload = {
        "scope": scope,
        "reported_actions": list(actions),
        "problem_status": problem_status,
        "human_requested": human_requested,
        "clarification_facts": list(facts),
        "new_issue": new_issue,
    }
    payload.update(extra)
    return json.dumps(payload)


def action(step_id, outcome):
    return {"step_id": step_id, "outcome": outcome}


# ================================================================
# 0. ALLOWLIST: minimal vocabulary only
# ================================================================

def test_allowed_steps_only_step_id_and_action(allowed):
    assert allowed  # non-empty
    for entry in allowed:
        assert set(entry) == {"step_id", "action"}
        # Explicitly: no graph / evidence leakage.
        assert "transitions" not in entry
        assert "terminators" not in entry
        assert "evidence_query" not in entry


def test_allowed_steps_none_procedure_is_empty():
    assert az.build_allowed_steps(None) == ()


def test_allowed_steps_matches_procedure_ids(procedure, allowed):
    assert {e["step_id"] for e in allowed} == set(procedure.steps)


# ================================================================
# 1. FAIL-SAFE OBSERVATION SHAPE
# ================================================================

def test_fail_safe_observation_shape():
    obs = az._fail_safe_observation()
    assert obs.scope == ObservationScope.UNCLEAR
    assert obs.problem_status is None
    assert obs.human_requested is False
    assert obs.new_issue is None
    assert obs.clarification_facts == ()
    assert len(obs.reported_actions) == 1
    assert obs.reported_actions[0].step_id is None
    assert obs.reported_actions[0].outcome is None


# ================================================================
# 2. HAPPY PATHS -- parsing
# ================================================================

def test_single_mappable_action(allowed):
    obs = az.parse_semantic_output(
        make_raw(actions=(action("STEP_A", "failed"),),
                 problem_status="ongoing"),
        allowed_steps=allowed,
    )
    assert obs.scope == ObservationScope.IN_SCOPE
    assert obs.reported_actions == (
        ReportedAction("STEP_A", StepOutcome.FAILED),
    )
    assert obs.problem_status == ProblemStatus.ONGOING


def test_multi_action_preserved_in_order(allowed):
    obs = az.parse_semantic_output(
        make_raw(
            actions=(
                action("STEP_A", "failed"),
                action("STEP_B", "skipped"),
            ),
            problem_status="ongoing",
        ),
        allowed_steps=allowed,
    )
    assert obs.reported_actions == (
        ReportedAction("STEP_A", StepOutcome.FAILED),
        ReportedAction("STEP_B", StepOutcome.SKIPPED),
    )


def test_failed_outcome(allowed):
    obs = az.parse_semantic_output(
        make_raw(actions=(action("STEP_A", "failed"),)), allowed_steps=allowed
    )
    assert obs.reported_actions[0].outcome == StepOutcome.FAILED


def test_succeeded_outcome(allowed):
    obs = az.parse_semantic_output(
        make_raw(actions=(action("STEP_A", "succeeded"),),
                 problem_status="resolved"),
        allowed_steps=allowed,
    )
    assert obs.reported_actions[0].outcome == StepOutcome.SUCCEEDED
    assert obs.problem_status == ProblemStatus.RESOLVED


def test_skipped_outcome_exact(allowed):
    obs = az.parse_semantic_output(
        make_raw(actions=(action("STEP_A", "skipped"),)),
        allowed_steps=allowed,
    )
    assert obs.reported_actions[0] == ReportedAction(
        "STEP_A", StepOutcome.SKIPPED
    )


def test_done_unknown_outcome_exact(allowed):
    obs = az.parse_semantic_output(
        make_raw(actions=(action("STEP_A", "done_unknown"),)),
        allowed_steps=allowed,
    )
    assert obs.reported_actions[0] == ReportedAction(
        "STEP_A", StepOutcome.DONE_UNKNOWN
    )


def test_human_requested(allowed):
    obs = az.parse_semantic_output(
        make_raw(human_requested=True), allowed_steps=allowed
    )
    assert obs.human_requested is True


def test_out_of_scope(allowed):
    obs = az.parse_semantic_output(
        make_raw(scope="out_of_scope"), allowed_steps=allowed
    )
    assert obs.scope == ObservationScope.OUT_OF_SCOPE


def test_out_of_scope_with_human_request(allowed):
    obs = az.parse_semantic_output(
        make_raw(scope="out_of_scope", human_requested=True),
        allowed_steps=allowed,
    )
    assert obs.scope == ObservationScope.OUT_OF_SCOPE
    assert obs.human_requested is True


def test_new_issue_present(allowed):
    obs = az.parse_semantic_output(
        make_raw(new_issue="screen flicker"), allowed_steps=allowed
    )
    assert obs.new_issue == "screen flicker"


def test_blank_new_issue_normalized_to_none(allowed):
    obs = az.parse_semantic_output(
        make_raw(new_issue="   "), allowed_steps=allowed
    )
    assert obs.new_issue is None


def test_continuation_is_not_new_issue(allowed):
    # "не помогло" about the SAME problem must NOT be reported as new_issue.
    obs = az.parse_semantic_output(
        make_raw(actions=(action("STEP_A", "failed"),), new_issue=None),
        allowed_steps=allowed,
    )
    assert obs.new_issue is None


def test_clarification_facts_preserved(allowed):
    obs = az.parse_semantic_output(
        make_raw(facts=("model=X", "led=red")), allowed_steps=allowed
    )
    assert obs.clarification_facts == ("model=X", "led=red")


def test_unmappable_failed_preserved_verbatim(allowed):
    # CORRECTION 1: parser does NOT bind it to any step.
    obs = az.parse_semantic_output(
        make_raw(actions=(action(None, "failed"),),
                 problem_status="ongoing"),
        allowed_steps=allowed,
    )
    assert obs.reported_actions == (
        ReportedAction(None, StepOutcome.FAILED),
    )


def test_null_outcome_preserved(allowed):
    obs = az.parse_semantic_output(
        make_raw(actions=(action("STEP_A", None),)), allowed_steps=allowed
    )
    assert obs.reported_actions == (ReportedAction("STEP_A", None),)


# ================================================================
# 3. CORRECTION 2 -- STRICT REJECTIONS -> fail-safe
# ================================================================

def _assert_fail_safe(obs):
    assert obs == az._fail_safe_observation()


def test_invalid_json_rejected(allowed):
    _assert_fail_safe(az.parse_semantic_output("{not json", allowed_steps=allowed))


def test_non_object_root_rejected(allowed):
    _assert_fail_safe(az.parse_semantic_output("[1, 2, 3]", allowed_steps=allowed))


def test_missing_field_rejected(allowed):
    raw = json.dumps(
        {
            "scope": "in_scope",
            "reported_actions": [],
            # problem_status missing
            "human_requested": False,
            "clarification_facts": [],
            "new_issue": None,
        }
    )
    _assert_fail_safe(az.parse_semantic_output(raw, allowed_steps=allowed))


def test_extra_field_rejected(allowed):
    # CORRECTION 2: EXTRA field is a hard rejection (was previously "ignored").
    raw = make_raw(bogus_field="surprise")
    _assert_fail_safe(az.parse_semantic_output(raw, allowed_steps=allowed))


def test_extra_per_action_field_rejected(allowed):
    raw = make_raw(actions=({"step_id": "STEP_A", "outcome": "failed",
                             "note": "x"},))
    _assert_fail_safe(az.parse_semantic_output(raw, allowed_steps=allowed))


def test_forbidden_legacy_top_level_field_rejected(allowed):
    raw = make_raw(candidate_l1_action="restart")
    _assert_fail_safe(az.parse_semantic_output(raw, allowed_steps=allowed))


def test_invalid_scope_enum_rejected(allowed):
    _assert_fail_safe(
        az.parse_semantic_output(make_raw(scope="banana"), allowed_steps=allowed)
    )


def test_invalid_outcome_enum_rejected(allowed):
    _assert_fail_safe(
        az.parse_semantic_output(
            make_raw(actions=(action("STEP_A", "exploded"),)),
            allowed_steps=allowed,
        )
    )


def test_invalid_problem_status_enum_rejected(allowed):
    _assert_fail_safe(
        az.parse_semantic_output(
            make_raw(problem_status="maybe"), allowed_steps=allowed
        )
    )


def test_wrong_type_human_requested_rejected(allowed):
    _assert_fail_safe(
        az.parse_semantic_output(make_raw(human_requested="yes"),
                                 allowed_steps=allowed)
    )


def test_wrong_type_reported_actions_rejected(allowed):
    _assert_fail_safe(
        az.parse_semantic_output(
            make_raw(actions="STEP_A"), allowed_steps=allowed
        )
    )


def test_wrong_type_action_entry_rejected(allowed):
    _assert_fail_safe(
        az.parse_semantic_output(make_raw(actions=("STEP_A",)),
                                 allowed_steps=allowed)
    )


def test_invented_step_id_downgraded_to_unmappable(allowed):
    # An invented id must NOT be trusted as a step; it becomes step_id=None,
    # which the reducer turns into pending_clarification.
    obs = az.parse_semantic_output(
        make_raw(actions=(action("GHOST_STEP", "failed"),)),
        allowed_steps=allowed,
    )
    assert obs.reported_actions == (ReportedAction(None, StepOutcome.FAILED),)


def test_invented_step_id_no_allowlist_becomes_unmappable():
    obs = az.parse_semantic_output(
        make_raw(actions=(action("STEP_A", "failed"),)),
        allowed_steps=(),  # empty allowlist (e.g. no procedure)
    )
    assert obs.reported_actions == (ReportedAction(None, StepOutcome.FAILED),)


# ================================================================
# 4. PROVIDER BOUNDARY (analyze)
# ================================================================

def _provider_returning(raw):
    def _p(prompt, history):
        return raw
    return _p


def _provider_raising(exc):
    def _p(prompt, history):
        raise exc
    return _p


def test_analyze_happy_path(procedure):
    raw = make_raw(actions=(action("STEP_A", "failed"),),
                   problem_status="ongoing")
    obs = az.analyze(
        message="уже перезапускал, не помогло",
        history=[],
        procedure=procedure,
        last_recommendation="STEP_A",
        provider=_provider_returning(raw),
    )
    assert obs.reported_actions == (
        ReportedAction("STEP_A", StepOutcome.FAILED),
    )


def test_analyze_provider_exception_is_fail_safe(procedure):
    obs = az.analyze(
        message="x",
        history=[],
        procedure=procedure,
        provider=_provider_raising(RuntimeError("boom")),
    )
    _assert_fail_safe(obs)


def test_analyze_no_procedure_is_fail_safe_on_invented_id():
    obs = az.analyze(
        message="x",
        history=[],
        procedure=None,
        provider=_provider_returning(
            make_raw(actions=(action("STEP_A", "failed"),),
                     problem_status="ongoing")
        ),
    )
    # Empty allowlist -> step downgraded to None -> unmappable.
    assert obs.reported_actions == (ReportedAction(None, StepOutcome.FAILED),)


def test_analyze_prompt_contains_allowed_steps_only(procedure):
    captured = {}

    def _p(prompt, history):
        captured["prompt"] = prompt
        return make_raw()

    az.analyze(message="x", history=[], procedure=procedure, provider=_p)
    prompt = captured["prompt"]
    # Allowed step ids appear; the declarative GRAPH does NOT leak.
    # (The words ANSWER/CLARIFY/ESCALATE ARE allowed: the rules explicitly
    # FORBID choosing them. What must never leak is the transition table.)
    assert "STEP_A" in prompt
    assert "attempted_failed" not in prompt
    assert "transitions" not in prompt
    assert "terminators" not in prompt
    assert "RESOLVED" not in prompt  # no terminator target leak


def test_analyze_prompt_includes_readonly_last_recommendation(procedure):
    captured = {}

    def _p(prompt, history):
        captured["prompt"] = prompt
        return make_raw()

    az.analyze(
        message="не помогло",
        history=[],
        procedure=procedure,
        last_recommendation="STEP_B",
        provider=_p,
    )
    assert "STEP_B" in captured["prompt"]


# ================================================================
# 5. CROSS-LAYER: analyzer -> reducer -> router
# ================================================================

def test_cross_layer_failure_yields_clarify(procedure):
    # Provider fails -> fail-safe observation -> reducer flags pending ->
    # router CLARIFY(PENDING_CLARIFICATION).
    obs = az.analyze(
        message="x",
        history=[],
        procedure=procedure,
        provider=_provider_raising(RuntimeError("boom")),
    )
    s1 = update_procedure_state(initial_state(), obs, procedure)
    assert s1.pending_clarification is True
    d = route(s1, obs, procedure)
    assert d.action == RouteAction.CLARIFY
    assert d.clarify_reason == ClarifyReason.PENDING_CLARIFICATION


def test_cross_layer_invented_id_yields_clarify(procedure):
    # Invented id -> parser downgrades -> unmappable -> reducer flags
    # pending -> router CLARIFY(PENDING_CLARIFICATION).
    obs = az.parse_semantic_output(
        make_raw(actions=(action("GHOST_STEP", "failed"),),
                 problem_status="ongoing"),
        allowed_steps=az.build_allowed_steps(procedure),
    )
    s1 = update_procedure_state(initial_state(), obs, procedure)
    assert s1.pending_clarification is True
    d = route(s1, obs, procedure)
    assert d.action == RouteAction.CLARIFY
    assert d.clarify_reason == ClarifyReason.PENDING_CLARIFICATION


def test_cross_layer_invalid_json_yields_clarify(procedure):
    obs = az.parse_semantic_output("{broken", allowed_steps=az.build_allowed_steps(procedure))
    s1 = update_procedure_state(initial_state(), obs, procedure)
    d = route(s1, obs, procedure)
    assert d.action == RouteAction.CLARIFY
    assert d.clarify_reason == ClarifyReason.PENDING_CLARIFICATION


def test_cross_layer_failed_action_walks(procedure):
    # A clean FAILED of the entry step walks the declarative edge.
    obs = az.parse_semantic_output(
        make_raw(actions=(action("STEP_A", "failed"),),
                 problem_status="ongoing"),
        allowed_steps=az.build_allowed_steps(procedure),
    )
    from dataclasses import replace

    s0 = replace(initial_state(), last_recommended_step="STEP_A")
    s1 = update_procedure_state(s0, obs, procedure)
    assert s1.pending_clarification is False
    d = route(s1, obs, procedure)
    assert d.action == RouteAction.RECOMMEND_STEP
    assert d.next_step == "STEP_B"
    assert d.transition_type == "attempted_failed"


def test_cross_layer_black_screen_full(black_screen=None):
    procedure = load_procedure(BLACK_SCREEN)
    allowed = az.build_allowed_steps(procedure)

    obs0 = az.parse_semantic_output(make_raw(), allowed_steps=allowed)
    s0 = update_procedure_state(initial_state(), obs0, procedure)
    d0 = route(s0, obs0, procedure)
    assert d0.action == RouteAction.RECOMMEND_STEP
    assert d0.next_step == "CHECK_POWER"

    from dataclasses import replace

    obs1 = az.parse_semantic_output(
        make_raw(actions=(action("CHECK_POWER", "failed"),),
                 problem_status="ongoing"),
        allowed_steps=allowed,
    )
    s1 = update_procedure_state(
        replace(s0, last_recommended_step="CHECK_POWER"), obs1, procedure
    )
    d1 = route(s1, obs1, procedure)
    assert d1.action == RouteAction.RECOMMEND_STEP
    assert d1.next_step == "CHECK_CABLE"


# ================================================================
# 6. ANTI-REGRESSION GUARDS
# ================================================================

def _module_ast():
    return ast.parse(Path(az.__file__).read_text(encoding="utf-8"))


def test_no_production_or_llm_imports():
    tree = _module_ast()
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for forbidden in (
        "yandex",
        "openai",
        "requests",
        "httpx",
        "psycopg",
        "psycopg2",
        "sqlalchemy",
    ):
        assert forbidden not in imported


def test_no_file_search_vocabulary():
    source = Path(az.__file__).read_text(encoding="utf-8")
    # No File Search tool / vector-store call exists in this slice.
    for forbidden in (
        "file_search",
        "vector_store",
        "responses.create",
        "chat.completions",
    ):
        assert forbidden not in source


def test_no_legacy_policy_callable_used():
    # Scan CODE only (identifiers / string literals), NOT comments or
    # docstrings -- the module docstring may NAME the legacy anti-pattern
    # to explain its deliberate absence.
    tree = _module_ast()

    # Strip docstrings so prose mentioning the anti-pattern is not flagged.
    for node in ast.walk(tree):
        if isinstance(
            node,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:]

    code = ast.dump(tree)
    assert "apply_support_policy" not in code


def test_legacy_forbidden_fields_constant():
    assert "candidate_l1_action" in az.LEGACY_FORBIDDEN_FIELDS
    assert "candidate_action_already_attempted" in az.LEGACY_FORBIDDEN_FIELDS


def test_public_api_surface():
    assert set(az.__all__) == {
        "LEGACY_FORBIDDEN_FIELDS",
        "AnalyzerProvider",
        "build_allowed_steps",
        "build_analyzer_prompt",
        "parse_semantic_output",
        "analyze",
    }
