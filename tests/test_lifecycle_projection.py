# -*- coding: utf-8 -*-
"""Slice HL-1 -- offline proofs for the pure lifecycle projection.

PURE: NO network, NO credentials, NO database, NO event loop, NO SupportCore.
The projection is exercised with typed doubles only and asserted to be a
total, deterministic, side-effect-free relabeling of an ALREADY-MADE decision
(never a second Router).
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from support.clarification import clarify
from support.domain import (
    ConversationMode,
    HandoffStatus,
    SupportEvent,
    SupportStatus,
)
from support.knowledge_qa import KnowledgeQAStatus, KnowledgeQAResult
from support.lifecycle_projection import (
    CaseAction,
    LifecycleProjection,
    project_turn_outcome,
)
from support.procedure_state import ProcedureState
from support.router import (
    ClarifyReason,
    EscalateReason,
    RouteAction,
    RoutingDecision,
)
from support.support_core import (
    CLARIFICATION,
    ESCALATE,
    KNOWLEDGE_QA,
    NO_PROCEDURE,
    OUT_OF_SCOPE,
    PROCEDURE_NOT_FOUND,
    TURN_EXECUTED,
    SupportTurnOutcome,
)
from support.turn_executor import PersistedTurnResult


# ============================================================
# CANONICAL STATES
# ============================================================

def _state(mode, handoff=None):
    return (mode, SupportStatus.ACTIVE, handoff)


AI_ACTIVE = _state(ConversationMode.AI)
AI_WAITING_USER = _state(ConversationMode.AI, HandoffStatus.WAITING_USER)
HUMAN_WAITING_OPERATOR = _state(
    ConversationMode.HUMAN, HandoffStatus.WAITING_OPERATOR
)
HUMAN_WAITING_RESOLUTION = _state(
    ConversationMode.HUMAN, HandoffStatus.WAITING_RESOLUTION_CONFIRMATION
)

AI_STATES = [AI_ACTIVE, AI_WAITING_USER]
HUMAN_STATES = [HUMAN_WAITING_OPERATOR, HUMAN_WAITING_RESOLUTION]
ALL_STATES = AI_STATES + HUMAN_STATES


# ============================================================
# TYPED OUTCOME DOUBLES
# ============================================================

def _turn_executed(
    action: RouteAction,
    *,
    escalate_reason: EscalateReason | None = None,
    clarify_reason: ClarifyReason | None = None,
) -> SupportTurnOutcome:
    decision = RoutingDecision(
        action=action,
        escalate_reason=escalate_reason,
        clarify_reason=clarify_reason,
    )
    persisted = PersistedTurnResult(
        conversation_id="c1",
        turn=SimpleNamespace(decision=decision, assistant_text="step"),
        state=ProcedureState(),
        revision=1,
    )
    return SupportTurnOutcome(
        status=TURN_EXECUTED, conversation_id="c1", persisted=persisted
    )


def _global_escalate() -> SupportTurnOutcome:
    return SupportTurnOutcome(status=ESCALATE, conversation_id="c1", text="handoff")


def _out_of_scope() -> SupportTurnOutcome:
    return SupportTurnOutcome(
        status=OUT_OF_SCOPE, conversation_id="c1", text="out of scope"
    )


def _knowledge() -> SupportTurnOutcome:
    return SupportTurnOutcome(
        status=KNOWLEDGE_QA,
        conversation_id="c1",
        knowledge=KnowledgeQAResult(
            status=KnowledgeQAStatus.ANSWERED, text="answer"
        ),
    )


def _clarification() -> SupportTurnOutcome:
    return SupportTurnOutcome(
        status=CLARIFICATION, conversation_id="c1", clarification=clarify()
    )


def _no_procedure() -> SupportTurnOutcome:
    return SupportTurnOutcome(status=NO_PROCEDURE, conversation_id="c1")


def _procedure_not_found() -> SupportTurnOutcome:
    return SupportTurnOutcome(status=PROCEDURE_NOT_FOUND, conversation_id="c1")


# ============================================================
# 1. GLOBAL ESCALATE
# ============================================================

def test_global_escalate_from_ai_active_creates_new():
    p = project_turn_outcome(_global_escalate(), AI_ACTIVE)

    assert p.outcome_status == ESCALATE
    assert p.decision == "ESCALATE"
    assert p.reason == "USER_REQUESTED_HUMAN"
    assert p.fsm_event is SupportEvent.ESCALATE_NEW
    assert p.case_action is CaseAction.CREATE_NEW


def test_global_escalate_from_waiting_user_resumes():
    p = project_turn_outcome(_global_escalate(), AI_WAITING_USER)

    assert p.fsm_event is SupportEvent.ESCALATE_RESUME
    assert p.case_action is CaseAction.RESUME


@pytest.mark.parametrize("state", HUMAN_STATES)
def test_global_escalate_while_human_owned_is_defensive(state):
    p = project_turn_outcome(_global_escalate(), state)

    assert p.fsm_event is SupportEvent.USER_MESSAGE_HUMAN
    assert p.case_action is CaseAction.NONE


# ============================================================
# 2. TURN_EXECUTED -- exhaustive RouteAction mapping
# ============================================================

def _action_kwargs(action: RouteAction) -> dict:
    if action is RouteAction.ESCALATE:
        return {"escalate_reason": EscalateReason.PROCEDURE_TERMINAL}
    if action is RouteAction.CLARIFY:
        return {"clarify_reason": ClarifyReason.PENDING_CLARIFICATION}
    return {}


@pytest.mark.parametrize("action", list(RouteAction))
def test_turn_executed_covers_every_route_action(action):
    p = project_turn_outcome(
        _turn_executed(action, **_action_kwargs(action)), AI_ACTIVE
    )

    assert p.outcome_status == TURN_EXECUTED
    assert p.procedure_action is action
    if action is RouteAction.ESCALATE:
        assert p.case_action in (CaseAction.CREATE_NEW, CaseAction.RESUME)
    else:
        assert p.case_action is CaseAction.NONE


def test_turn_executed_recommend_step_is_an_ai_turn():
    p = project_turn_outcome(
        _turn_executed(RouteAction.RECOMMEND_STEP), AI_ACTIVE
    )
    assert p.decision == "ANSWER"
    assert p.reason == "PROCEDURE_STEP"
    assert p.fsm_event is SupportEvent.USER_MESSAGE_AI
    assert p.case_action is CaseAction.NONE


def test_turn_executed_clarify_preserves_router_reason():
    p = project_turn_outcome(
        _turn_executed(
            RouteAction.CLARIFY,
            clarify_reason=ClarifyReason.NEW_ISSUE_RESEELECT,
        ),
        AI_ACTIVE,
    )
    assert p.decision == "CLARIFY"
    assert p.reason == ClarifyReason.NEW_ISSUE_RESEELECT.value
    assert p.clarify_reason is ClarifyReason.NEW_ISSUE_RESEELECT
    assert p.fsm_event is SupportEvent.USER_MESSAGE_AI
    assert p.case_action is CaseAction.NONE


@pytest.mark.parametrize("reason", list(EscalateReason))
def test_turn_executed_escalate_preserves_router_reason(reason):
    p = project_turn_outcome(
        _turn_executed(RouteAction.ESCALATE, escalate_reason=reason), AI_ACTIVE
    )
    assert p.decision == "ESCALATE"
    assert p.reason == reason.value
    assert p.escalation_reason is reason
    assert p.fsm_event is SupportEvent.ESCALATE_NEW
    assert p.case_action is CaseAction.CREATE_NEW


def test_turn_executed_escalate_resumes_from_waiting_user():
    p = project_turn_outcome(
        _turn_executed(
            RouteAction.ESCALATE,
            escalate_reason=EscalateReason.PROCEDURE_TERMINAL,
        ),
        AI_WAITING_USER,
    )
    assert p.fsm_event is SupportEvent.ESCALATE_RESUME
    assert p.case_action is CaseAction.RESUME


def test_turn_executed_out_of_scope_is_an_ai_turn():
    p = project_turn_outcome(
        _turn_executed(RouteAction.OUT_OF_SCOPE), AI_ACTIVE
    )
    assert p.decision == "OUT_OF_SCOPE"
    assert p.fsm_event is SupportEvent.USER_MESSAGE_AI
    assert p.case_action is CaseAction.NONE


# ============================================================
# 3. AI-OWNED FAMILIES: KNOWLEDGE_QA / CLARIFICATION / OUT_OF_SCOPE
# ============================================================

@pytest.mark.parametrize(
    "factory,decision,reason",
    [
        (_knowledge, "ANSWER", "KNOWLEDGE_QA"),
        (_clarification, "CLARIFY", "CLARIFICATION"),
        (_out_of_scope, "OUT_OF_SCOPE", "OUT_OF_SCOPE"),
    ],
)
@pytest.mark.parametrize("state", AI_STATES)
def test_ai_owned_families_apply_user_message_ai(factory, decision, reason, state):
    p = project_turn_outcome(factory(), state)

    assert p.decision == decision
    assert p.reason == reason
    assert p.fsm_event is SupportEvent.USER_MESSAGE_AI
    assert p.case_action is CaseAction.NONE


# ============================================================
# 4. NO_PROCEDURE -- no lifecycle side effect
# ============================================================

@pytest.mark.parametrize("state", ALL_STATES)
def test_no_procedure_creates_no_lifecycle_side_effect(state):
    p = project_turn_outcome(_no_procedure(), state)

    assert p.outcome_status == NO_PROCEDURE
    assert p.reason == "NO_PROCEDURE"
    assert p.fsm_event is None
    assert p.case_action is CaseAction.NONE
    assert p.decision != "ESCALATE"  # no fabricated escalation


# ============================================================
# 5. PROCEDURE_NOT_FOUND -- internal consistency failure
# ============================================================

@pytest.mark.parametrize("state", ALL_STATES)
def test_procedure_not_found_creates_no_lifecycle_side_effect(state):
    p = project_turn_outcome(_procedure_not_found(), state)

    assert p.outcome_status == PROCEDURE_NOT_FOUND
    assert p.decision is None  # NOT escalated, NOT clarified
    assert p.reason == "PROCEDURE_NOT_FOUND"
    assert p.fsm_event is None
    assert p.case_action is CaseAction.NONE


# ============================================================
# 6. RESOLVE (Correction 1)
# ============================================================

def test_resolve_from_ai_active_emits_ai_resolved():
    p = project_turn_outcome(_turn_executed(RouteAction.RESOLVE), AI_ACTIVE)

    assert p.decision == "ANSWER"
    assert p.reason == "RESOLVED"
    assert p.fsm_event is SupportEvent.AI_RESOLVED
    assert p.case_action is CaseAction.NONE


def test_resolve_from_waiting_user_emits_no_invalid_fsm_event():
    # The FSM has NO legal AI-resolution-confirmation transition from
    # waiting_user: the projection must emit NO event, never an invalid one.
    p = project_turn_outcome(_turn_executed(RouteAction.RESOLVE), AI_WAITING_USER)

    assert p.decision == "ANSWER"
    assert p.reason == "RESOLVED"
    assert p.fsm_event is None
    assert p.case_action is CaseAction.NONE


@pytest.mark.parametrize("state", HUMAN_STATES)
def test_resolve_while_human_owned_is_defensive(state):
    p = project_turn_outcome(_turn_executed(RouteAction.RESOLVE), state)

    assert p.fsm_event is SupportEvent.USER_MESSAGE_HUMAN
    assert p.case_action is CaseAction.NONE


# ============================================================
# 7. CLOSED ENUM + TOTAL COVERAGE
# ============================================================

def test_case_action_closed_enum():
    assert {member.name for member in CaseAction} == {
        "NONE",
        "HUMAN_FORWARD",
        "CREATE_NEW",
        "RESUME",
    }
    assert {member.value for member in CaseAction} == {
        "NONE",
        "HUMAN_FORWARD",
        "CREATE_NEW",
        "RESUME",
    }


def test_every_support_turn_outcome_status_is_covered():
    outcomes = [
        _turn_executed(RouteAction.RECOMMEND_STEP),
        _knowledge(),
        _clarification(),
        _global_escalate(),
        _out_of_scope(),
        _no_procedure(),
        _procedure_not_found(),
    ]
    statuses = {o.status for o in outcomes}
    assert statuses == {
        TURN_EXECUTED,
        KNOWLEDGE_QA,
        CLARIFICATION,
        ESCALATE,
        OUT_OF_SCOPE,
        NO_PROCEDURE,
        PROCEDURE_NOT_FOUND,
    }

    for outcome in outcomes:
        projection = project_turn_outcome(outcome, AI_ACTIVE)
        assert isinstance(projection, LifecycleProjection)
        assert projection.outcome_status == outcome.status
        # The projection never invents the pre-engine human-forward action.
        assert projection.case_action is not CaseAction.HUMAN_FORWARD


# ============================================================
# 8. PURE / DETERMINISTIC / SIDE-EFFECT FREE
# ============================================================

def test_projection_is_deterministic_and_side_effect_free():
    outcome = _turn_executed(RouteAction.RECOMMEND_STEP)

    first = project_turn_outcome(outcome, AI_ACTIVE)
    second = project_turn_outcome(outcome, AI_ACTIVE)

    assert first == second
    assert outcome.status == TURN_EXECUTED  # input untouched

    with pytest.raises(dataclasses.FrozenInstanceError):
        first.reason = "mutated"  # type: ignore[misc]


def test_malformed_routing_decision_fails_closed():
    # A RoutingDecision invariant violation must not be silently projected.
    with pytest.raises(ValueError):
        project_turn_outcome(
            _turn_executed(RouteAction.ESCALATE), AI_ACTIVE
        )
