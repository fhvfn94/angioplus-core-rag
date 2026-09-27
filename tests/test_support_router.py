"""Deterministic Support Router tests (pure: no DB, no network, no LLM).

Covers the approved Slice 3 unit-test matrix, including the two final
corrections:

  * SKIPPED / DONE_UNKNOWN -> CLARIFY(STEP_OUTCOME_UNKNOWN) (inconclusive,
    never coerced into FAILED/SUCCEEDED, never follow an edge).
  * PROCEDURE_TERMINAL (normal graph edge to ESCALATE terminator) and
    STEPS_EXHAUSTED (structural dead-end, no edge) are mutually exclusive.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from support.observation import (
    ObservationScope,
    ProblemStatus,
    ReportedAction,
    RequestKind,
    SemanticObservation,
    StepOutcome,
)
from support.procedure import Procedure, load_procedure
from support.procedure_state import (
    ProcedureState,
    StepAttempt,
    initial_state,
    update_procedure_state,
)
from support.router import (
    ClarifyReason,
    EscalateReason,
    RouteAction,
    RoutingDecision,
    authoritative_outcome,
    route,
)


FIXTURE = (
    Path(__file__).parent / "fixtures" / "procedures" / "test_procedure.yaml"
)
BLACK_SCREEN = (
    Path(__file__).parent / "fixtures" / "procedures" / "black_screen.yaml"
)


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

@pytest.fixture
def procedure() -> Procedure:
    return load_procedure(FIXTURE)


@pytest.fixture
def black_screen() -> Procedure:
    return load_procedure(BLACK_SCREEN)


def obs(
    *,
    scope: ObservationScope = ObservationScope.IN_SCOPE,
    actions: tuple[ReportedAction, ...] = (),
    problem_status: ProblemStatus | None = None,
    human_requested: bool = False,
    facts: tuple[str, ...] = (),
    new_issue: str | None = None,
    request_kind: RequestKind = RequestKind.PROBLEM,
) -> SemanticObservation:
    return SemanticObservation(
        scope=scope,
        reported_actions=actions,
        problem_status=problem_status,
        human_requested=human_requested,
        clarification_facts=facts,
        new_issue=new_issue,
        request_kind=request_kind,
    )


def state_with(
    *,
    issue: str | None = None,
    procedure_id: str | None = None,
    procedure_version: int | None = None,
    last_recommended_step: str | None = None,
    attempts: tuple[StepAttempt, ...] = (),
    problem_status: ProblemStatus = ProblemStatus.UNKNOWN,
    pending_clarification: bool = False,
    clarification_topic: str | None = None,
) -> ProcedureState:
    return ProcedureState(
        issue=issue,
        procedure_id=procedure_id,
        procedure_version=procedure_version,
        last_recommended_step=last_recommended_step,
        step_attempts=attempts,
        problem_status=problem_status,
        pending_clarification=pending_clarification,
        clarification_topic=clarification_topic,
    )


def in_scope(**kw) -> SemanticObservation:
    kw.setdefault("scope", ObservationScope.IN_SCOPE)
    return obs(**kw)


# ----------------------------------------------------------------
# 0. Vocabulary / shape
# ----------------------------------------------------------------

def test_route_action_members():
    assert set(RouteAction) == {
        RouteAction.RECOMMEND_STEP,
        RouteAction.CLARIFY,
        RouteAction.ESCALATE,
        RouteAction.RESOLVE,
        RouteAction.OUT_OF_SCOPE,
    }


def test_escalate_reason_members():
    assert set(EscalateReason) == {
        EscalateReason.USER_REQUESTED_HUMAN,
        EscalateReason.PROCEDURE_TERMINAL,
        EscalateReason.STEPS_EXHAUSTED,
        EscalateReason.INTERNAL_INCONSISTENCY,
    }


def test_clarify_reason_members():
    assert set(ClarifyReason) == {
        ClarifyReason.PENDING_CLARIFICATION,
        ClarifyReason.NO_PROCEDURE,
        ClarifyReason.NEW_ISSUE_RESEELECT,
        ClarifyReason.STEP_OUTCOME_UNKNOWN,
    }


def test_decision_is_frozen():
    d = RoutingDecision(action=RouteAction.RESOLVE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.action = RouteAction.ESCALATE  # type: ignore[misc]


def test_decision_field_set():
    assert {f.name for f in dataclasses.fields(RoutingDecision)} == {
        "action",
        "next_step",
        "transition_type",
        "escalate_reason",
        "clarify_reason",
    }


# ----------------------------------------------------------------
# 1-2. Purity / totality
# ----------------------------------------------------------------

def test_deterministic_and_pure(procedure):
    s0 = initial_state()
    o = in_scope(
        actions=(ReportedAction("STEP_A", StepOutcome.FAILED),),
        problem_status=ProblemStatus.ONGOING,
    )
    a = route(s0, o, procedure)
    b = route(s0, o, procedure)
    assert a == b
    assert s0.step_attempts == ()  # state not mutated


def test_total_on_none_procedure():
    d = route(initial_state(), in_scope(), None)
    assert d.action == RouteAction.CLARIFY
    assert d.clarify_reason == ClarifyReason.NO_PROCEDURE


# ----------------------------------------------------------------
# 3. Gate 1: scope
# ----------------------------------------------------------------

def test_out_of_scope_wins(procedure):
    d = route(
        initial_state(),
        obs(scope=ObservationScope.OUT_OF_SCOPE),
        procedure,
    )
    assert d.action == RouteAction.OUT_OF_SCOPE
    assert d.escalate_reason is None
    assert d.clarify_reason is None


def test_out_of_scope_wins_over_human_request(procedure):
    d = route(
        initial_state(),
        obs(scope=ObservationScope.OUT_OF_SCOPE, human_requested=True),
        procedure,
    )
    assert d.action == RouteAction.OUT_OF_SCOPE


# ----------------------------------------------------------------
# 4. Gate 2: explicit human request (global gate)
# ----------------------------------------------------------------

def test_human_request_escalates(procedure):
    d = route(initial_state(), in_scope(human_requested=True), procedure)
    assert d.action == RouteAction.ESCALATE
    assert d.escalate_reason == EscalateReason.USER_REQUESTED_HUMAN


def test_human_request_does_not_need_procedure_edge(procedure):
    # Mid-walk human request still escalates; not a CLARIFY.
    prior = state_with(
        last_recommended_step="STEP_A",
        attempts=(StepAttempt("STEP_A", StepOutcome.FAILED, 0),),
    )
    d = route(prior, in_scope(human_requested=True), procedure)
    assert d.action == RouteAction.ESCALATE
    assert d.escalate_reason == EscalateReason.USER_REQUESTED_HUMAN


def test_human_request_without_procedure_still_escalates():
    d = route(initial_state(), in_scope(human_requested=True), None)
    assert d.action == RouteAction.ESCALATE
    assert d.escalate_reason == EscalateReason.USER_REQUESTED_HUMAN


# ----------------------------------------------------------------
# 5. Gate 3: resolved
# ----------------------------------------------------------------

def test_problem_resolved_in_observation(procedure):
    d = route(
        initial_state(),
        in_scope(problem_status=ProblemStatus.RESOLVED),
        procedure,
    )
    assert d.action == RouteAction.RESOLVE


def test_problem_resolved_in_state(procedure):
    prior = state_with(problem_status=ProblemStatus.RESOLVED)
    d = route(prior, in_scope(), procedure)
    assert d.action == RouteAction.RESOLVE


def test_resolved_wins_over_no_procedure():
    d = route(
        state_with(problem_status=ProblemStatus.RESOLVED),
        in_scope(),
        None,
    )
    assert d.action == RouteAction.RESOLVE


# ----------------------------------------------------------------
# 6. Gate 5: identity / dangling cursor
# ----------------------------------------------------------------

def test_procedure_id_mismatch(procedure):
    prior = state_with(procedure_id="OTHER_PROCEDURE")
    d = route(prior, in_scope(), procedure)
    assert d.action == RouteAction.ESCALATE
    assert d.escalate_reason == EscalateReason.INTERNAL_INCONSISTENCY


def test_procedure_version_mismatch(procedure):
    prior = state_with(procedure_version=99)
    d = route(prior, in_scope(), procedure)
    assert d.action == RouteAction.ESCALATE
    assert d.escalate_reason == EscalateReason.INTERNAL_INCONSISTENCY


def test_dangling_last_recommended_step(procedure):
    prior = state_with(last_recommended_step="GHOST_STEP")
    d = route(prior, in_scope(), procedure)
    assert d.action == RouteAction.ESCALATE
    assert d.escalate_reason == EscalateReason.INTERNAL_INCONSISTENCY


# ----------------------------------------------------------------
# 7. Gate 6: new issue
# ----------------------------------------------------------------

def test_new_issue_clarifies(procedure):
    d = route(initial_state(), in_scope(new_issue="screen flicker"), procedure)
    assert d.action == RouteAction.CLARIFY
    assert d.clarify_reason == ClarifyReason.NEW_ISSUE_RESEELECT


def test_blank_new_issue_is_ignored(procedure):
    # Blank string must not divert the walk into a reselect.
    prior = state_with(
        last_recommended_step="STEP_A",
        attempts=(StepAttempt("STEP_A", StepOutcome.FAILED, 0),),
    )
    d = route(prior, in_scope(new_issue="   "), procedure)
    assert d.action == RouteAction.RECOMMEND_STEP
    assert d.next_step == "STEP_B"


# ----------------------------------------------------------------
# 8. Gate 7: pending clarification
# ----------------------------------------------------------------

def test_pending_clarification_clarifies(procedure):
    prior = state_with(pending_clarification=True)
    d = route(prior, in_scope(), procedure)
    assert d.action == RouteAction.CLARIFY
    assert d.clarify_reason == ClarifyReason.PENDING_CLARIFICATION


# ----------------------------------------------------------------
# 9. Gate 8: initial entry
# ----------------------------------------------------------------

def test_initial_entry_recommends_entry_step(procedure):
    d = route(initial_state(), in_scope(), procedure)
    assert d.action == RouteAction.RECOMMEND_STEP
    assert d.next_step == procedure.entry_step
    assert d.transition_type is None
    assert d.escalate_reason is None
    assert d.clarify_reason is None


# ----------------------------------------------------------------
# 10. Gate 9: deterministic walk
# ----------------------------------------------------------------

def test_walk_no_attempt_recommends_current(procedure):
    prior = state_with(last_recommended_step="STEP_A")
    d = route(prior, in_scope(), procedure)
    assert d.action == RouteAction.RECOMMEND_STEP
    assert d.next_step == "STEP_A"


def test_walk_failed_follows_attempted_failed(procedure):
    prior = state_with(
        last_recommended_step="STEP_A",
        attempts=(StepAttempt("STEP_A", StepOutcome.FAILED, 0),),
    )
    d = route(prior, in_scope(), procedure)
    assert d.action == RouteAction.RECOMMEND_STEP
    assert d.next_step == "STEP_B"
    assert d.transition_type == "attempted_failed"


def test_walk_succeeded_follows_resolved(procedure):
    prior = state_with(
        last_recommended_step="STEP_A",
        attempts=(StepAttempt("STEP_A", StepOutcome.SUCCEEDED, 0),),
    )
    d = route(prior, in_scope(), procedure)
    assert d.action == RouteAction.RESOLVE


def test_walk_to_escalate_terminator_is_procedure_terminal(procedure):
    # STEP_A failed -> STEP_B; STEP_B failed -> ESCALATE terminator.
    prior = state_with(
        last_recommended_step="STEP_B",
        attempts=(
            StepAttempt("STEP_A", StepOutcome.FAILED, 0),
            StepAttempt("STEP_B", StepOutcome.FAILED, 1),
        ),
    )
    d = route(prior, in_scope(), procedure)
    assert d.action == RouteAction.ESCALATE
    assert d.escalate_reason == EscalateReason.PROCEDURE_TERMINAL


def test_walk_to_resolved_terminator(procedure):
    prior = state_with(
        last_recommended_step="STEP_B",
        attempts=(
            StepAttempt("STEP_A", StepOutcome.FAILED, 0),
            StepAttempt("STEP_B", StepOutcome.SUCCEEDED, 1),
        ),
    )
    d = route(prior, in_scope(), procedure)
    assert d.action == RouteAction.RESOLVE


def test_walk_anchored_on_entry_when_no_attempts_or_cursor(procedure):
    # No attempts, no cursor -> entry step (same as gate 8).
    d = route(initial_state(), in_scope(), procedure)
    assert d.next_step == "STEP_A"


def test_walk_never_re_recommends_attempted_step(procedure):
    # STEP_A already attempted -> walk continues from its outcome, never
    # re-recommends STEP_A.
    prior = state_with(
        last_recommended_step="STEP_A",
        attempts=(StepAttempt("STEP_A", StepOutcome.FAILED, 0),),
    )
    d = route(prior, in_scope(), procedure)
    assert d.next_step != "STEP_A"


def test_walk_authoritative_outcome_is_latest(procedure):
    # Latest attempt wins: FAILED then SUCCEEDED -> resolved edge -> RESOLVE.
    prior = state_with(
        last_recommended_step="STEP_A",
        attempts=(
            StepAttempt("STEP_A", StepOutcome.FAILED, 0),
            StepAttempt("STEP_A", StepOutcome.SUCCEEDED, 1),
        ),
    )
    d = route(prior, in_scope(), procedure)
    assert d.action == RouteAction.RESOLVE


def test_authoritative_outcome_helper(procedure):
    prior = state_with(
        attempts=(
            StepAttempt("STEP_A", StepOutcome.FAILED, 0),
            StepAttempt("STEP_A", StepOutcome.SUCCEEDED, 1),
        ),
    )
    assert authoritative_outcome(prior, "STEP_A") == StepOutcome.SUCCEEDED
    assert authoritative_outcome(prior, "STEP_B") is None


# ----------------------------------------------------------------
# 11. CORRECTION 1: SKIPPED / DONE_UNKNOWN are inconclusive
# ----------------------------------------------------------------

def test_skipped_step_clarifies_outcome_unknown(procedure):
    prior = state_with(
        last_recommended_step="STEP_A",
        attempts=(StepAttempt("STEP_A", StepOutcome.SKIPPED, 0),),
    )
    d = route(prior, in_scope(), procedure)
    assert d.action == RouteAction.CLARIFY
    assert d.clarify_reason == ClarifyReason.STEP_OUTCOME_UNKNOWN
    assert d.next_step is None


def test_done_unknown_step_clarifies_outcome_unknown(procedure):
    prior = state_with(
        last_recommended_step="STEP_A",
        attempts=(StepAttempt("STEP_A", StepOutcome.DONE_UNKNOWN, 0),),
    )
    d = route(prior, in_scope(), procedure)
    assert d.action == RouteAction.CLARIFY
    assert d.clarify_reason == ClarifyReason.STEP_OUTCOME_UNKNOWN


def test_skipped_does_not_follow_attempted_failed(procedure):
    # If SKIPPED wrongly followed attempted_failed it would reach STEP_B.
    prior = state_with(
        last_recommended_step="STEP_A",
        attempts=(StepAttempt("STEP_A", StepOutcome.SKIPPED, 0),),
    )
    d = route(prior, in_scope(), procedure)
    assert d.next_step is None  # no edge followed at all


def test_done_unknown_does_not_follow_resolved(procedure):
    prior = state_with(
        last_recommended_step="STEP_A",
        attempts=(StepAttempt("STEP_A", StepOutcome.DONE_UNKNOWN, 0),),
    )
    d = route(prior, in_scope(), procedure)
    assert d.action == RouteAction.CLARIFY  # not RESOLVE


def test_skipped_on_final_step_does_not_escalate_terminal(procedure):
    # Even on the last step, SKIPPED must NOT become PROCEDURE_TERMINAL.
    prior = state_with(
        last_recommended_step="STEP_B",
        attempts=(
            StepAttempt("STEP_A", StepOutcome.FAILED, 0),
            StepAttempt("STEP_B", StepOutcome.SKIPPED, 1),
        ),
    )
    d = route(prior, in_scope(), procedure)
    assert d.action == RouteAction.CLARIFY
    assert d.clarify_reason == ClarifyReason.STEP_OUTCOME_UNKNOWN
    assert d.escalate_reason is None


def test_definitive_outcome_follows_edge_after_inconclusive_history(procedure):
    # STEP_A: SKIPPED then later FAILED -> latest FAILED is authoritative.
    prior = state_with(
        last_recommended_step="STEP_A",
        attempts=(
            StepAttempt("STEP_A", StepOutcome.SKIPPED, 0),
            StepAttempt("STEP_A", StepOutcome.FAILED, 1),
        ),
    )
    d = route(prior, in_scope(), procedure)
    assert d.action == RouteAction.RECOMMEND_STEP
    assert d.next_step == "STEP_B"


# ----------------------------------------------------------------
# 12. CORRECTION 2: PROCEDURE_TERMINAL vs STEPS_EXHAUSTED
# ----------------------------------------------------------------

def test_procedure_terminal_reason_on_declared_edge(procedure):
    prior = state_with(
        last_recommended_step="STEP_B",
        attempts=(StepAttempt("STEP_B", StepOutcome.FAILED, 0),),
    )
    d = route(prior, in_scope(), procedure)
    assert d.action == RouteAction.ESCALATE
    assert d.escalate_reason == EscalateReason.PROCEDURE_TERMINAL


def test_steps_exhausted_on_structural_dead_end():
    # A procedure whose FAILED outcome has no attempted_failed edge.
    from support.procedure import load_procedure_from_dict

    dead_end = load_procedure_from_dict(
        {
            "procedure_id": "DEAD_END",
            "version": 1,
            "issue": "dead end",
            "entry_step": "ONLY",
            "steps": [
                {
                    "id": "ONLY",
                    "action": "the only step",
                    "transitions": {"resolved": "RESOLVED"},
                }
            ],
            "terminators": ["RESOLVED"],
        }
    )
    prior = state_with(
        procedure_id="DEAD_END",
        procedure_version=1,
        last_recommended_step="ONLY",
        attempts=(StepAttempt("ONLY", StepOutcome.FAILED, 0),),
    )
    d = route(prior, in_scope(), dead_end)
    assert d.action == RouteAction.ESCALATE
    assert d.escalate_reason == EscalateReason.STEPS_EXHAUSTED


def test_terminal_and_exhausted_are_distinct_reasons():
    assert (
        EscalateReason.PROCEDURE_TERMINAL
        != EscalateReason.STEPS_EXHAUSTED
    )
    assert EscalateReason.PROCEDURE_TERMINAL.value == "procedure_terminal"
    assert EscalateReason.STEPS_EXHAUSTED.value == "steps_exhausted"


def test_no_overlapping_support_reason_remains():
    # The old overlapping PROCEDURE_REQUIRES_SUPPORT member is gone.
    assert not hasattr(EscalateReason, "PROCEDURE_REQUIRES_SUPPORT")


# ----------------------------------------------------------------
# 13. Invariant: exactly one reason field is populated
# ----------------------------------------------------------------

def test_recommend_step_populates_only_step_fields(procedure):
    d = route(initial_state(), in_scope(), procedure)
    assert d.next_step is not None
    assert d.escalate_reason is None
    assert d.clarify_reason is None


def test_escalate_populates_only_escalate_reason(procedure):
    d = route(initial_state(), in_scope(human_requested=True), procedure)
    assert d.escalate_reason is not None
    assert d.next_step is None
    assert d.transition_type is None
    assert d.clarify_reason is None


def test_clarify_populates_only_clarify_reason(procedure):
    d = route(initial_state(), in_scope(new_issue="other"), procedure)
    assert d.clarify_reason is not None
    assert d.next_step is None
    assert d.transition_type is None
    assert d.escalate_reason is None


def test_resolve_populates_no_reason(procedure):
    d = route(
        initial_state(),
        in_scope(problem_status=ProblemStatus.RESOLVED),
        procedure,
    )
    assert d.escalate_reason is None
    assert d.clarify_reason is None


# ----------------------------------------------------------------
# 14. Router IS the transition reader; no LLM/routing leakage
# ----------------------------------------------------------------

def test_router_reads_transitions(procedure):
    """The router MUST read transitions (complement to the reducer guard)."""
    original = procedure.steps

    class _CountingStep:
        def __init__(self, real):
            self.id = real.id
            self.action = real.action
            self.evidence_query = real.evidence_query
            self._real = real
            self.transitions_read = 0

        @property
        def transitions(self):
            self.transitions_read += 1
            return self._real.transitions

    class _CountingProcedure:
        def __init__(self, real):
            self.procedure_id = real.procedure_id
            self.version = real.version
            self.issue = real.issue
            self.entry_step = real.entry_step
            self.terminators = real.terminators
            self.steps = {
                sid: _CountingStep(step) for sid, step in original.items()
            }

        def step_ids(self):
            return frozenset(self.steps)

    guarded = _CountingProcedure(procedure)
    prior = state_with(
        last_recommended_step="STEP_A",
        attempts=(StepAttempt("STEP_A", StepOutcome.FAILED, 0),),
    )
    route(prior, in_scope(), guarded)
    assert guarded.steps["STEP_A"].transitions_read >= 1


def test_router_has_no_candidate_policy_vocabulary():
    """No legacy string-priority cascade vocabulary in CODE (identifiers /
    names / string literals). The module docstring may *name* these
    anti-patterns to explain their deliberate absence, so docstrings are
    excluded from the scan."""
    import ast

    import support.router as router_mod

    tree = ast.parse(Path(router_mod.__file__).read_text(encoding="utf-8"))

    # Drop docstrings so prose mentioning the anti-pattern is not flagged.
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
    for forbidden in (
        "candidate_l1_action",
        "candidate_action_already_attempted",
        "apply_support_policy",
    ):
        assert forbidden not in code


def test_router_imports_no_production_or_llm():
    """No production-runtime / network / DB import may reach the router."""
    import ast

    import support.router as router_mod

    tree = ast.parse(Path(router_mod.__file__).read_text(encoding="utf-8"))
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
        "psycopg",
        "psycopg2",
        "sqlalchemy",
    ):
        assert forbidden not in imported


# ----------------------------------------------------------------
# 15. End-to-end reducer -> router (Slice 2 + Slice 3 integration)
# ----------------------------------------------------------------

def test_reducer_then_router_initial_then_failed(procedure):
    s0 = initial_state()
    d0 = route(s0, in_scope(), procedure)
    assert d0.action == RouteAction.RECOMMEND_STEP
    assert d0.next_step == "STEP_A"

    # Simulate: assistant recommended STEP_A, user reports it failed.
    s1 = dataclasses.replace(s0, last_recommended_step="STEP_A")
    s1 = update_procedure_state(
        s1,
        in_scope(
            actions=(ReportedAction("STEP_A", StepOutcome.FAILED),),
            problem_status=ProblemStatus.ONGOING,
        ),
        procedure,
    )
    d1 = route(s1, in_scope(), procedure)
    assert d1.action == RouteAction.RECOMMEND_STEP
    assert d1.next_step == "STEP_B"


def test_reducer_then_router_resolution(procedure):
    s0 = initial_state()
    s1 = dataclasses.replace(s0, last_recommended_step="STEP_A")
    obs_resolved = in_scope(
        actions=(ReportedAction("STEP_A", StepOutcome.SUCCEEDED),),
        problem_status=ProblemStatus.RESOLVED,
    )
    s1 = update_procedure_state(s1, obs_resolved, procedure)
    d1 = route(s1, obs_resolved, procedure)
    assert d1.action == RouteAction.RESOLVE


# ----------------------------------------------------------------
# 16. REGRESSION: "black screen" -- deterministic, ordered, no LLM
# ----------------------------------------------------------------

def test_black_screen_full_deterministic_walk(black_screen):
    """The classic black-screen flow yields a deterministic ordered decision.

    At NO point does the router consult an LLM or any
    ``candidate_l1_action`` vocabulary: the same inputs always produce the
    same decision.
    """
    # 1. First contact -> entry step.
    s0 = initial_state()
    d0 = route(s0, in_scope(), black_screen)
    assert d0.action == RouteAction.RECOMMEND_STEP
    assert d0.next_step == "CHECK_POWER"

    # 2. Power check failed -> next step via the declarative edge.
    s1 = dataclasses.replace(s0, last_recommended_step="CHECK_POWER")
    s1 = update_procedure_state(
        s1,
        in_scope(
            actions=(ReportedAction("CHECK_POWER", StepOutcome.FAILED),),
            problem_status=ProblemStatus.ONGOING,
        ),
        black_screen,
    )
    d1 = route(s1, in_scope(), black_screen)
    assert d1.action == RouteAction.RECOMMEND_STEP
    assert d1.next_step == "CHECK_CABLE"
    assert d1.transition_type == "attempted_failed"

    # 3. Cable check failed -> declarative ESCALATE terminator.
    s2 = update_procedure_state(
        dataclasses.replace(s1, last_recommended_step="CHECK_CABLE"),
        in_scope(
            actions=(ReportedAction("CHECK_CABLE", StepOutcome.FAILED),),
            problem_status=ProblemStatus.ONGOING,
        ),
        black_screen,
    )
    d2 = route(s2, in_scope(), black_screen)
    assert d2.action == RouteAction.ESCALATE
    assert d2.escalate_reason == EscalateReason.PROCEDURE_TERMINAL

    # Determinism: re-running each decision is identical.
    assert route(s0, in_scope(), black_screen) == d0
    assert route(s1, in_scope(), black_screen) == d1
    assert route(s2, in_scope(), black_screen) == d2


def test_black_screen_human_request_short_circuits(black_screen):
    s0 = initial_state()
    d = route(s0, in_scope(human_requested=True), black_screen)
    assert d.action == RouteAction.ESCALATE
    assert d.escalate_reason == EscalateReason.USER_REQUESTED_HUMAN
