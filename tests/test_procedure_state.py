"""Procedure State reducer tests (pure: no DB, no network, no LLM).

Covers the approved Slice 2 unit-test matrix. Reuses the synthetic
TEST_PROCEDURE built in-memory (mirrors tests/fixtures/procedures/
test_procedure.yaml from Slice 1).
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


FIXTURE = (
    Path(__file__).parent / "fixtures" / "procedures" / "test_procedure.yaml"
)


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

@pytest.fixture
def procedure() -> Procedure:
    return load_procedure(FIXTURE)


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


# ----------------------------------------------------------------
# 1-2. Purity / totality
# ----------------------------------------------------------------

def test_deterministic_and_pure(procedure):
    s0 = initial_state()
    o = obs(actions=(ReportedAction("STEP_A", StepOutcome.FAILED),),
            problem_status=ProblemStatus.ONGOING)
    a = update_procedure_state(s0, o, procedure)
    b = update_procedure_state(s0, o, procedure)
    assert a == b
    # previous state was not mutated
    assert s0.step_attempts == ()


def test_total_on_none_procedure():
    s0 = initial_state()
    o = obs(actions=(ReportedAction("STEP_A", StepOutcome.FAILED),),
            problem_status=ProblemStatus.ONGOING)
    # Must not crash and must not invent a step id without a procedure.
    result = update_procedure_state(s0, o, None)
    assert result.step_attempts == ()
    assert result.pending_clarification is True


# ----------------------------------------------------------------
# 3-6. Reported actions
# ----------------------------------------------------------------

def test_single_mapped_action(procedure):
    result = update_procedure_state(
        initial_state(),
        obs(actions=(ReportedAction("STEP_A", StepOutcome.FAILED),)),
        procedure,
    )
    assert len(result.step_attempts) == 1
    attempt = result.step_attempts[0]
    assert attempt.step_id == "STEP_A"
    assert attempt.outcome == StepOutcome.FAILED
    assert attempt.sequence == 0


def test_multiple_actions_one_message(procedure):
    result = update_procedure_state(
        initial_state(),
        obs(actions=(
            ReportedAction("STEP_A", StepOutcome.FAILED),
            ReportedAction("STEP_B", StepOutcome.FAILED),
        )),
        procedure,
    )
    assert [a.step_id for a in result.step_attempts] == ["STEP_A", "STEP_B"]
    assert [a.sequence for a in result.step_attempts] == [0, 1]


def test_unmappable_action_sets_clarification(procedure):
    result = update_procedure_state(
        state_with(issue="pump noise", last_recommended_step="STEP_A"),
        obs(actions=(ReportedAction(None, StepOutcome.FAILED),)),
        procedure,
    )
    assert result.step_attempts == ()
    assert result.pending_clarification is True
    assert result.clarification_topic == "STEP_A"


def test_unknown_step_id_sets_clarification_no_attempt(procedure):
    # step_id present but NOT part of the procedure -> defensive handling.
    result = update_procedure_state(
        state_with(issue="pump noise", last_recommended_step="STEP_A"),
        obs(actions=(ReportedAction("TOTALLY_MADE_UP", StepOutcome.FAILED),)),
        procedure,
    )
    assert result.step_attempts == ()          # not appended
    assert result.pending_clarification is True  # not silently ignored
    assert result.clarification_topic == "STEP_A"


def test_unknown_step_id_topic_falls_back_to_issue(procedure):
    result = update_procedure_state(
        state_with(issue="pump noise", last_recommended_step=None),
        obs(actions=(ReportedAction("NOPE", StepOutcome.FAILED),)),
        procedure,
    )
    assert result.step_attempts == ()
    assert result.pending_clarification is True
    assert result.clarification_topic == "pump noise"


# ----------------------------------------------------------------
# 7. "Already did this"
# ----------------------------------------------------------------

def test_already_did_this_is_skipped_not_failed(procedure):
    result = update_procedure_state(
        initial_state(),
        obs(actions=(ReportedAction("STEP_A", StepOutcome.SKIPPED),)),
        procedure,
    )
    assert result.step_attempts[0].outcome == StepOutcome.SKIPPED
    assert "STEP_A" in result.attempted_step_ids()


# ----------------------------------------------------------------
# 8. Resolution
# ----------------------------------------------------------------

def test_problem_resolved_appends_no_attempt_and_keeps_history(procedure):
    prior = state_with(
        attempts=(StepAttempt("STEP_A", StepOutcome.FAILED, 0),),
        problem_status=ProblemStatus.ONGOING,
    )
    result = update_procedure_state(
        prior,
        obs(problem_status=ProblemStatus.RESOLVED),
        procedure,
    )
    assert result.problem_status == ProblemStatus.RESOLVED
    assert result.step_attempts == prior.step_attempts  # history preserved
    assert result.pending_clarification is False


# ----------------------------------------------------------------
# 9-11. Bare "didn't help" -> context resolution
# ----------------------------------------------------------------

def test_bare_no_help_with_outstanding_recommendation(procedure):
    prior = state_with(
        issue="pump noise",
        last_recommended_step="STEP_A",
        problem_status=ProblemStatus.ONGOING,
    )
    # No mappable actions, only an ongoing problem_status conveyed.
    result = update_procedure_state(
        prior,
        obs(problem_status=ProblemStatus.ONGOING),
        procedure,
    )
    assert len(result.step_attempts) == 1
    assert result.step_attempts[0].step_id == "STEP_A"
    assert result.step_attempts[0].outcome == StepOutcome.FAILED


def test_bare_no_help_without_recommendation_sets_clarification(procedure):
    prior = state_with(issue="pump noise", problem_status=ProblemStatus.ONGOING)
    result = update_procedure_state(
        prior,
        obs(problem_status=ProblemStatus.ONGOING),
        procedure,
    )
    assert result.step_attempts == ()
    assert result.pending_clarification is True
    assert result.clarification_topic == "pump noise"


def test_bare_no_help_recommendation_already_recorded_no_duplicate(procedure):
    prior = state_with(
        issue="pump noise",
        last_recommended_step="STEP_A",
        attempts=(StepAttempt("STEP_A", StepOutcome.FAILED, 0),),
        problem_status=ProblemStatus.ONGOING,
    )
    result = update_procedure_state(
        prior,
        obs(problem_status=ProblemStatus.ONGOING),
        procedure,
    )
    # No second attempt for STEP_A; flagged for clarification instead.
    assert [a.step_id for a in result.step_attempts] == ["STEP_A"]
    assert result.pending_clarification is True


# ----------------------------------------------------------------
# 12-13. Scope / new issue are non-destructive
# ----------------------------------------------------------------

def test_out_of_scope_does_not_touch_attempts(procedure):
    prior = state_with(
        issue="pump noise",
        attempts=(StepAttempt("STEP_A", StepOutcome.FAILED, 0),),
        problem_status=ProblemStatus.ONGOING,
    )
    result = update_procedure_state(
        prior,
        obs(scope=ObservationScope.OUT_OF_SCOPE),
        procedure,
    )
    assert result.step_attempts == prior.step_attempts
    assert result.procedure_id == prior.procedure_id
    assert result.last_recommended_step == prior.last_recommended_step


def test_new_issue_is_informational_not_destructive(procedure):
    prior = state_with(
        issue="pump noise",
        procedure_id="TEST_PROCEDURE",
        procedure_version=1,
        last_recommended_step="STEP_A",
        attempts=(StepAttempt("STEP_A", StepOutcome.FAILED, 0),),
    )
    result = update_procedure_state(
        prior,
        obs(new_issue="screen flicker"),
        procedure,
    )
    # issue updated, but nothing destroyed / switched.
    assert result.issue == "screen flicker"
    assert result.procedure_id == "TEST_PROCEDURE"
    assert result.procedure_version == 1
    assert result.last_recommended_step == "STEP_A"
    assert result.step_attempts == prior.step_attempts


# ----------------------------------------------------------------
# 14-15. Reducer must not read transitions / move a cursor
# ----------------------------------------------------------------

def test_reducer_never_reads_procedure_transitions(monkeypatch, procedure):
    """Guard: no code path may touch a step's ``transitions``."""
    original_steps = procedure.steps

    class _NoTransitionsStep:
        def __init__(self, real):
            self.id = real.id
            self.action = real.action
            self.evidence_query = real.evidence_query

        @property
        def transitions(self):  # pragma: no cover - should never run
            raise AssertionError(
                "Reducer read ProcedureStep.transitions -- forbidden."
            )

    class _NoTransitionsProcedure:
        def __init__(self, real):
            self.procedure_id = real.procedure_id
            self.version = real.version
            self.issue = real.issue
            self.entry_step = real.entry_step
            self.terminators = real.terminators
            self.steps = {
                sid: _NoTransitionsStep(step)
                for sid, step in original_steps.items()
            }

        def step_ids(self):
            return frozenset(self.steps)

    guarded = _NoTransitionsProcedure(procedure)

    # Exercise several paths that could plausibly read transitions.
    for o in (
        obs(actions=(ReportedAction("STEP_A", StepOutcome.FAILED),)),
        obs(problem_status=ProblemStatus.ONGOING),
        obs(problem_status=ProblemStatus.RESOLVED),
        obs(actions=(ReportedAction("NOPE", StepOutcome.FAILED),)),
        obs(new_issue="other"),
    ):
        update_procedure_state(initial_state(), o, guarded)


def test_state_has_no_cursor_field(procedure):
    field_names = {f.name for f in dataclasses.fields(ProcedureState)}
    assert "current_step" not in field_names
    assert "procedure_exhausted" not in field_names


def test_router_owned_fields_carried_through(procedure):
    prior = state_with(
        procedure_id="TEST_PROCEDURE",
        procedure_version=1,
        last_recommended_step="STEP_B",
    )
    result = update_procedure_state(
        prior,
        obs(actions=(ReportedAction("STEP_A", StepOutcome.FAILED),)),
        procedure,
    )
    assert result.procedure_id == "TEST_PROCEDURE"
    assert result.procedure_version == 1
    assert result.last_recommended_step == "STEP_B"


# ----------------------------------------------------------------
# 16-20. Invariants
# ----------------------------------------------------------------

def test_append_only(procedure):
    prior = state_with(attempts=(StepAttempt("STEP_A", StepOutcome.FAILED, 0),))
    result = update_procedure_state(
        prior,
        obs(actions=(ReportedAction("STEP_B", StepOutcome.FAILED),)),
        procedure,
    )
    assert result.step_attempts[:1] == prior.step_attempts
    assert len(result.step_attempts) == 2


def test_attempt_domain_within_procedure(procedure):
    result = update_procedure_state(
        initial_state(),
        obs(actions=(
            ReportedAction("STEP_A", StepOutcome.FAILED),
            ReportedAction("STEP_B", StepOutcome.FAILED),
        )),
        procedure,
    )
    for attempt in result.step_attempts:
        assert attempt.step_id in procedure.step_ids()


def test_sequence_contiguous(procedure):
    result = update_procedure_state(
        initial_state(),
        obs(actions=(
            ReportedAction("STEP_A", StepOutcome.FAILED),
            ReportedAction("STEP_B", StepOutcome.FAILED),
        )),
        procedure,
    )
    assert [a.sequence for a in result.step_attempts] == [0, 1]


def test_derived_membership_matches_attempts(procedure):
    result = update_procedure_state(
        initial_state(),
        obs(actions=(
            ReportedAction("STEP_A", StepOutcome.FAILED),
            ReportedAction("STEP_B", StepOutcome.SKIPPED),
        )),
        procedure,
    )
    assert result.attempted_step_ids() == {
        a.step_id for a in result.step_attempts
    }


def test_clarification_topic_implies_pending(procedure):
    result = update_procedure_state(
        state_with(issue="x", last_recommended_step="STEP_A"),
        obs(actions=(ReportedAction("NOPE", StepOutcome.FAILED),)),
        procedure,
    )
    assert result.clarification_topic is not None
    assert result.pending_clarification is True


# ----------------------------------------------------------------
# 21-22. De-dup is local only; no replay/idempotency claim
# ----------------------------------------------------------------

def test_local_dedup_within_single_observation(procedure):
    result = update_procedure_state(
        initial_state(),
        obs(actions=(
            ReportedAction("STEP_A", StepOutcome.FAILED),
            ReportedAction("STEP_A", StepOutcome.FAILED),
        )),
        procedure,
    )
    assert [a.step_id for a in result.step_attempts] == ["STEP_A"]


def test_no_domain_replay_protection(procedure):
    """A distinct observation with repeated content DOES append again.

    Demonstrates there is no (false) domain-level idempotency: transport
    idempotency lives in support.idempotency, not here.
    """
    o = obs(actions=(ReportedAction("STEP_A", StepOutcome.FAILED),))
    s1 = update_procedure_state(initial_state(), o, procedure)
    s2 = update_procedure_state(s1, o, procedure)
    assert [a.step_id for a in s2.step_attempts] == ["STEP_A", "STEP_A"]


def test_issue_refined_from_new_issue_only(procedure):
    result = update_procedure_state(
        initial_state(),
        obs(new_issue="  bearing wear  "),
        procedure,
    )
    assert result.issue == "bearing wear"
