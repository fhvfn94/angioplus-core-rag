"""Contract tests for the Semantic Observation value objects (pure).

These validate the *shape* of the observation contract only: enum
membership, tuple-ness, and basic field typing. They do NOT exercise the
reducer (that is tests/test_procedure_state.py).
"""

from __future__ import annotations

import dataclasses

import pytest

from support.observation import (
    ObservationScope,
    ProblemStatus,
    ReportedAction,
    SemanticObservation,
    StepOutcome,
)


# ----------------------------------------------------------------
# Enum membership
# ----------------------------------------------------------------

def test_scope_members():
    assert set(ObservationScope) == {
        ObservationScope.IN_SCOPE,
        ObservationScope.OUT_OF_SCOPE,
        ObservationScope.UNCLEAR,
    }
    assert ObservationScope.IN_SCOPE.value == "in_scope"
    assert ObservationScope.OUT_OF_SCOPE.value == "out_of_scope"
    assert ObservationScope.UNCLEAR.value == "unclear"


def test_problem_status_members():
    assert set(ProblemStatus) == {
        ProblemStatus.UNKNOWN,
        ProblemStatus.ONGOING,
        ProblemStatus.RESOLVED,
    }
    assert ProblemStatus.UNKNOWN.value == "unknown"
    assert ProblemStatus.ONGOING.value == "ongoing"
    assert ProblemStatus.RESOLVED.value == "resolved"


def test_step_outcome_members():
    assert set(StepOutcome) == {
        StepOutcome.FAILED,
        StepOutcome.SUCCEEDED,
        StepOutcome.SKIPPED,
        StepOutcome.DONE_UNKNOWN,
    }


# ----------------------------------------------------------------
# Value-object shape
# ----------------------------------------------------------------

def test_reported_action_fields():
    assert {f.name for f in dataclasses.fields(ReportedAction)} == {
        "step_id",
        "outcome",
    }


def test_semantic_observation_fields():
    assert {f.name for f in dataclasses.fields(SemanticObservation)} == {
        "scope",
        "reported_actions",
        "problem_status",
        "human_requested",
        "clarification_facts",
        "new_issue",
    }


def test_observation_is_frozen():
    obs = SemanticObservation(
        scope=ObservationScope.IN_SCOPE,
        reported_actions=(),
        problem_status=None,
        human_requested=False,
        clarification_facts=(),
        new_issue=None,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        obs.scope = ObservationScope.UNCLEAR  # type: ignore[misc]


def test_reported_actions_accepts_unmappable_and_mappable():
    obs = SemanticObservation(
        scope=ObservationScope.IN_SCOPE,
        reported_actions=(
            ReportedAction(step_id=None, outcome=StepOutcome.FAILED),
            ReportedAction(step_id="STEP_A", outcome=None),
        ),
        problem_status=ProblemStatus.ONGOING,
        human_requested=False,
        clarification_facts=("model=X",),
        new_issue=None,
    )
    assert isinstance(obs.reported_actions, tuple)
    assert obs.reported_actions[0].step_id is None
    assert obs.reported_actions[1].step_id == "STEP_A"


def test_no_routing_fields_present():
    """The contract must not leak routing concepts."""
    names = {f.name for f in dataclasses.fields(SemanticObservation)}
    for forbidden in (
        "candidate_l1_action",
        "candidate_action_already_attempted",
        "documentation_requires_support",
        "action",
        "decision",
        "needs_clarification",
    ):
        assert forbidden not in names
