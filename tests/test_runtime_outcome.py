"""
Focused, offline tests for the Slice HL-2a runtime outcome contract.

Pure and isolated: no DB, no network, no Yandex, no HTTP, no Tracker. The
tests instantiate REAL ``SupportTurnOutcome`` objects (and the REAL
``LifecycleProjection`` they delegate to) so any drift in the composition
boundary fails loudly.

The whole point of HL-2a is that it DECIDES NOTHING: it must preserve the
projection owner's ``decision`` / ``reason`` / ``fsm_event`` / ``case_action``
VERBATIM and add ONLY the user-facing ``text`` and the typed
``USER_RESPONSE`` / ``INTERNAL_ERROR`` distinction.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from support import runtime_outcome
from support import support_core as sc
from support.clarification import clarify
from support.domain import (
    AI_ACTIVE,
    AI_ACTIVE_WAITING_USER,
    HUMAN_ACTIVE_WAITING_OPERATOR,
    SupportEvent,
)
from support.global_policy import GlobalAction
from support.knowledge_qa import KnowledgeQAResult, KnowledgeQAStatus
from support.lifecycle_projection import CaseAction
from support.procedure_state import ProcedureState
from support.response_generator import (
    _INTERNAL_ERROR_TEXT as GENERATOR_INTERNAL_ERROR_TEXT,
)
from support.router import (
    ClarifyReason,
    EscalateReason,
    RouteAction,
    RoutingDecision,
)
from support.runtime_outcome import (
    RuntimeReplay,
    RuntimeResult,
    RuntimeTurn,
    RuntimeTurnKind,
    adapt_turn_outcome,
)
from support.turn_executor import PersistedTurnResult
from support.turn_projection import project_global_terminal


# ============================================================
# BUILDERS
# ============================================================

def _routing(action, *, escalate_reason=None, clarify_reason=None):
    return RoutingDecision(
        action=action,
        escalate_reason=escalate_reason,
        clarify_reason=clarify_reason,
    )


def _turn_executed(
    action,
    *,
    text="Шаг процедуры.",
    escalate_reason=None,
    clarify_reason=None,
):
    decision = _routing(
        action, escalate_reason=escalate_reason, clarify_reason=clarify_reason
    )
    turn = SimpleNamespace(assistant_text=text, decision=decision)
    return sc.SupportTurnOutcome(
        status=sc.TURN_EXECUTED,
        conversation_id="chat-1",
        persisted=PersistedTurnResult(
            conversation_id="chat-1",
            turn=turn,
            state=ProcedureState(),
            revision=1,
        ),
    )


# ============================================================
# 1. TURN_EXECUTED (procedure family)
# ============================================================

@pytest.mark.parametrize(
    "action,escalate_reason,clarify_reason,state,exp_decision,exp_reason,exp_event,exp_case",
    [
        (
            RouteAction.RECOMMEND_STEP, None, None, AI_ACTIVE,
            "ANSWER", "PROCEDURE_STEP", SupportEvent.USER_MESSAGE_AI, CaseAction.NONE,
        ),
        (
            RouteAction.CLARIFY, None, ClarifyReason.NEW_ISSUE_RESEELECT, AI_ACTIVE,
            "CLARIFY", "new_issue_reselect", SupportEvent.USER_MESSAGE_AI, CaseAction.NONE,
        ),
        (
            RouteAction.CLARIFY, None, ClarifyReason.PENDING_CLARIFICATION, AI_ACTIVE,
            "CLARIFY", "pending_clarification", SupportEvent.USER_MESSAGE_AI, CaseAction.NONE,
        ),
        (
            RouteAction.ESCALATE, EscalateReason.PROCEDURE_TERMINAL, None, AI_ACTIVE,
            "ESCALATE", "procedure_terminal", SupportEvent.ESCALATE_NEW, CaseAction.CREATE_NEW,
        ),
        (
            RouteAction.ESCALATE, EscalateReason.STEPS_EXHAUSTED, None, AI_ACTIVE,
            "ESCALATE", "steps_exhausted", SupportEvent.ESCALATE_NEW, CaseAction.CREATE_NEW,
        ),
        (
            RouteAction.RESOLVE, None, None, AI_ACTIVE,
            "ANSWER", "RESOLVED", SupportEvent.AI_RESOLVED, CaseAction.NONE,
        ),
        (
            RouteAction.RESOLVE, None, None, AI_ACTIVE_WAITING_USER,
            "ANSWER", "RESOLVED", None, CaseAction.NONE,
        ),
        (
            RouteAction.OUT_OF_SCOPE, None, None, AI_ACTIVE,
            "OUT_OF_SCOPE", "OUT_OF_SCOPE", SupportEvent.USER_MESSAGE_AI, CaseAction.NONE,
        ),
    ],
)
def test_turn_executed_projects_the_decision_verbatim(
    action, escalate_reason, clarify_reason, state,
    exp_decision, exp_reason, exp_event, exp_case,
):
    outcome = _turn_executed(
        action, escalate_reason=escalate_reason, clarify_reason=clarify_reason
    )
    turn = adapt_turn_outcome(outcome, state)

    assert turn.kind is RuntimeTurnKind.USER_RESPONSE
    assert turn.outcome_status == "TURN_EXECUTED"
    assert turn.conversation_id == "chat-1"
    assert turn.decision == exp_decision
    assert turn.reason == exp_reason
    assert turn.fsm_event is exp_event
    assert turn.case_action is exp_case
    assert turn.text == "Шаг процедуры."


# ============================================================
# 2. KNOWLEDGE_QA
# ============================================================

def test_knowledge_qa_uses_the_grounded_answer_text():
    outcome = sc.SupportTurnOutcome(
        status=sc.KNOWLEDGE_QA,
        conversation_id="chat-1",
        knowledge=KnowledgeQAResult(
            status=KnowledgeQAStatus.ANSWERED,
            text="Ответ из документации.",
        ),
    )
    turn = adapt_turn_outcome(outcome, AI_ACTIVE)

    assert turn.kind is RuntimeTurnKind.USER_RESPONSE
    assert turn.decision == "ANSWER"
    assert turn.reason == "KNOWLEDGE_QA"
    assert turn.fsm_event is SupportEvent.USER_MESSAGE_AI
    assert turn.case_action is CaseAction.NONE
    assert turn.text == "Ответ из документации."


# ============================================================
# 3. CLARIFICATION
# ============================================================

def test_clarification_uses_the_outcome_native_text():
    clarification = clarify()
    outcome = sc.SupportTurnOutcome(
        status=sc.CLARIFICATION,
        conversation_id="chat-1",
        clarification=clarification,
    )
    turn = adapt_turn_outcome(outcome, AI_ACTIVE)

    assert turn.decision == "CLARIFY"
    assert turn.reason == "CLARIFICATION"
    assert turn.fsm_event is SupportEvent.USER_MESSAGE_AI
    assert turn.case_action is CaseAction.NONE
    assert turn.text == clarification.text


# ============================================================
# 4. GLOBAL ESCALATE (+ 8. waiting_user resume)
# ============================================================

@pytest.mark.parametrize(
    "state,exp_event,exp_case",
    [
        (AI_ACTIVE, SupportEvent.ESCALATE_NEW, CaseAction.CREATE_NEW),
        (AI_ACTIVE_WAITING_USER, SupportEvent.ESCALATE_RESUME, CaseAction.RESUME),
    ],
)
def test_global_escalate(state, exp_event, exp_case):
    text = project_global_terminal(GlobalAction.ESCALATE)
    outcome = sc.SupportTurnOutcome(
        status=sc.ESCALATE, conversation_id="chat-1", text=text
    )
    turn = adapt_turn_outcome(outcome, state)

    assert turn.kind is RuntimeTurnKind.USER_RESPONSE
    assert turn.decision == "ESCALATE"
    assert turn.reason == "USER_REQUESTED_HUMAN"
    assert turn.fsm_event is exp_event
    assert turn.case_action is exp_case
    assert turn.text == text


def test_turn_executed_escalate_resumes_from_waiting_user():
    outcome = _turn_executed(
        RouteAction.ESCALATE,
        escalate_reason=EscalateReason.PROCEDURE_TERMINAL,
    )
    turn = adapt_turn_outcome(outcome, AI_ACTIVE_WAITING_USER)

    assert turn.decision == "ESCALATE"
    assert turn.reason == "procedure_terminal"
    assert turn.fsm_event is SupportEvent.ESCALATE_RESUME
    assert turn.case_action is CaseAction.RESUME


# ============================================================
# 5. OUT_OF_SCOPE (global)
# ============================================================

def test_global_out_of_scope():
    text = project_global_terminal(GlobalAction.OUT_OF_SCOPE)
    outcome = sc.SupportTurnOutcome(
        status=sc.OUT_OF_SCOPE, conversation_id="chat-1", text=text
    )
    turn = adapt_turn_outcome(outcome, AI_ACTIVE)

    assert turn.kind is RuntimeTurnKind.USER_RESPONSE
    assert turn.decision == "OUT_OF_SCOPE"
    assert turn.reason == "OUT_OF_SCOPE"
    assert turn.fsm_event is SupportEvent.USER_MESSAGE_AI
    assert turn.case_action is CaseAction.NONE
    assert turn.text == text


# ============================================================
# 6. NO_PROCEDURE
# ============================================================

@pytest.mark.parametrize(
    "state", [AI_ACTIVE, AI_ACTIVE_WAITING_USER, HUMAN_ACTIVE_WAITING_OPERATOR]
)
def test_no_procedure_is_a_clarify_without_fsm_move(state):
    outcome = sc.SupportTurnOutcome(
        status=sc.NO_PROCEDURE, conversation_id="chat-1"
    )
    turn = adapt_turn_outcome(outcome, state)

    assert turn.kind is RuntimeTurnKind.USER_RESPONSE
    assert turn.decision == "CLARIFY"
    assert turn.reason == "NO_PROCEDURE"
    assert turn.fsm_event is None
    assert turn.case_action is CaseAction.NONE
    # Reuses the EXISTING deterministic clarification capability.
    assert turn.text == clarify().text
    # No fabricated escalation.
    assert turn.decision != "ESCALATE"


# ============================================================
# 7. PROCEDURE_NOT_FOUND -> INTERNAL_ERROR
# ============================================================

@pytest.mark.parametrize(
    "state", [AI_ACTIVE, AI_ACTIVE_WAITING_USER, HUMAN_ACTIVE_WAITING_OPERATOR]
)
def test_procedure_not_found_is_a_typed_internal_error(state):
    outcome = sc.SupportTurnOutcome(
        status=sc.PROCEDURE_NOT_FOUND, conversation_id="chat-1"
    )
    turn = adapt_turn_outcome(outcome, state)

    assert turn.kind is RuntimeTurnKind.INTERNAL_ERROR
    assert turn.outcome_status == "PROCEDURE_NOT_FOUND"
    assert turn.decision is None
    assert turn.reason == "PROCEDURE_NOT_FOUND"
    assert turn.fsm_event is None
    assert turn.case_action is CaseAction.NONE
    # Deterministic, safe and non-blank, and it reuses the neutral wording.
    assert turn.text.strip() == GENERATOR_INTERNAL_ERROR_TEXT.strip()
    # No success / handoff / Tracker claim.
    lowered = turn.text.lower()
    for forbidden in ("оператор", "специалист", "заявк", "tracker", "передаю"):
        assert forbidden not in lowered


# ============================================================
# 9. HUMAN-OWNED STATE IS DEFENSIVE
# ============================================================

def test_human_owned_turn_mirrors_the_human_event_and_creates_no_case():
    outcome = _turn_executed(RouteAction.RECOMMEND_STEP)
    turn = adapt_turn_outcome(outcome, HUMAN_ACTIVE_WAITING_OPERATOR)

    assert turn.kind is RuntimeTurnKind.USER_RESPONSE
    assert turn.decision == "ANSWER"
    assert turn.fsm_event is SupportEvent.USER_MESSAGE_HUMAN
    assert turn.case_action is CaseAction.NONE


def test_human_owned_internal_error_still_moves_nothing():
    outcome = sc.SupportTurnOutcome(
        status=sc.PROCEDURE_NOT_FOUND, conversation_id="chat-1"
    )
    turn = adapt_turn_outcome(outcome, HUMAN_ACTIVE_WAITING_OPERATOR)

    assert turn.kind is RuntimeTurnKind.INTERNAL_ERROR
    assert turn.fsm_event is None
    assert turn.case_action is CaseAction.NONE


# ============================================================
# 10. DATACLASS INVARIANTS
# ============================================================

def _user_response(**overrides):
    base = dict(
        kind=RuntimeTurnKind.USER_RESPONSE,
        outcome_status="KNOWLEDGE_QA",
        conversation_id="chat-1",
        text="Ответ.",
        decision="ANSWER",
        reason="KNOWLEDGE_QA",
        case_action=CaseAction.NONE,
        fsm_event=SupportEvent.USER_MESSAGE_AI,
    )
    base.update(overrides)
    return RuntimeTurn(**base)


def _internal_error(**overrides):
    base = dict(
        kind=RuntimeTurnKind.INTERNAL_ERROR,
        outcome_status="PROCEDURE_NOT_FOUND",
        conversation_id="chat-1",
        text="Внутренняя ошибка.",
        decision=None,
        reason="PROCEDURE_NOT_FOUND",
        case_action=CaseAction.NONE,
        fsm_event=None,
    )
    base.update(overrides)
    return RuntimeTurn(**base)


def test_user_response_invariants():
    assert _user_response().decision == "ANSWER"

    for bad in (
        dict(text="   "),
        dict(text=""),
        dict(decision=None),
        dict(decision="   "),
        dict(reason=""),
        dict(outcome_status="  "),
        dict(conversation_id=""),
        dict(kind="user_response"),
        dict(case_action="NONE"),
        dict(fsm_event="USER_MESSAGE_AI"),
    ):
        with pytest.raises(ValueError):
            _user_response(**bad)


def test_internal_error_invariants():
    assert _internal_error().decision is None

    for bad in (
        dict(decision="ANSWER"),                      # no normal decision
        dict(reason="ANSWER"),                        # pinned reason only
        dict(outcome_status="KNOWLEDGE_QA"),          # only PROCEDURE_NOT_FOUND
        dict(fsm_event=SupportEvent.ESCALATE_NEW),    # cannot move the FSM
        dict(case_action=CaseAction.CREATE_NEW),      # cannot create a case
        dict(case_action=CaseAction.RESUME),          # cannot resume a case
        dict(text="   "),                             # text stays non-blank
    ):
        with pytest.raises(ValueError):
            _internal_error(**bad)


# ============================================================
# 11. HL-2c.2 ENGINE REPLAY (typed runtime replay boundary)
# ============================================================

def _replayed_turn_executed(*, conversation_id="chat-1", revision=4):
    """The ONE legitimate HL-2b / HL-2c.1 replay shape: ``TURN_EXECUTED``
    carrying a ``PersistedTurnResult`` with ``replayed`` ``True`` and
    ``turn`` ``None``."""
    return sc.SupportTurnOutcome(
        status=sc.TURN_EXECUTED,
        conversation_id=conversation_id,
        persisted=PersistedTurnResult(
            conversation_id=conversation_id,
            turn=None,
            state=ProcedureState(),
            revision=revision,
            replayed=True,
        ),
    )


def test_normal_turn_executed_still_returns_runtime_turn():
    outcome = _turn_executed(RouteAction.RECOMMEND_STEP)
    result = adapt_turn_outcome(outcome, AI_ACTIVE)

    assert isinstance(result, RuntimeTurn)
    assert not isinstance(result, RuntimeReplay)
    assert isinstance(result, RuntimeResult)
    # Normal behaviour UNCHANGED: text / decision / reason / case / fsm.
    assert result.text == "Шаг процедуры."
    assert result.decision == "ANSWER"
    assert result.reason == "PROCEDURE_STEP"
    assert result.case_action is CaseAction.NONE
    assert result.fsm_event is SupportEvent.USER_MESSAGE_AI


def test_replayed_turn_executed_returns_runtime_replay():
    result = adapt_turn_outcome(_replayed_turn_executed(), AI_ACTIVE)

    assert isinstance(result, RuntimeReplay)
    assert isinstance(result, RuntimeResult)
    assert not isinstance(result, RuntimeTurn)
    assert result.outcome_status == "TURN_EXECUTED"
    assert result.conversation_id == "chat-1"


def test_replay_no_longer_raises_attribute_error():
    # Regression for the pre-HL-2c.2 crash:
    # AttributeError: 'NoneType' object has no attribute 'decision'.
    try:
        result = adapt_turn_outcome(_replayed_turn_executed(), AI_ACTIVE)
    except AttributeError as exc:  # pragma: no cover
        pytest.fail(f"engine replay must not raise AttributeError: {exc}")
    assert isinstance(result, RuntimeReplay)


def test_runtime_replay_has_no_effect_fields():
    replay = adapt_turn_outcome(_replayed_turn_executed(), AI_ACTIVE)

    for forbidden in (
        "text", "decision", "reason", "case_action", "fsm_event", "kind",
    ):
        assert not hasattr(replay, forbidden), forbidden


def test_runtime_replay_does_not_expose_persisted_or_turn():
    replay = adapt_turn_outcome(_replayed_turn_executed(), AI_ACTIVE)

    assert not hasattr(replay, "persisted")
    assert not hasattr(replay, "turn")
    assert not hasattr(replay, "revision")
    assert not hasattr(replay, "execution_id")


def test_replay_does_not_call_lifecycle_projection(monkeypatch):
    # A replay must BYPASS LifecycleProjection entirely -- proven with a
    # sentinel, not merely by the absence of an AttributeError.
    calls = []

    def _boom(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError(
            "project_turn_outcome must NOT run for an engine replay"
        )

    monkeypatch.setattr(runtime_outcome, "project_turn_outcome", _boom)

    result = adapt_turn_outcome(_replayed_turn_executed(), AI_ACTIVE)

    assert isinstance(result, RuntimeReplay)
    assert calls == []  # projection was never consulted


def test_replay_is_independent_of_state():
    outcome = _replayed_turn_executed()
    a = adapt_turn_outcome(outcome, AI_ACTIVE)
    b = adapt_turn_outcome(outcome, AI_ACTIVE_WAITING_USER)
    c = adapt_turn_outcome(outcome, HUMAN_ACTIVE_WAITING_OPERATOR)

    # A replay consults no FSM state and moves nothing.
    assert a == b == c
    assert isinstance(a, RuntimeReplay)


def test_malformed_executed_payload_is_not_silently_a_replay():
    # ``turn is None`` WITHOUT the replay flag is NOT a legitimate replay
    # shape: it falls through to the existing projection path and fails
    # closed (AttributeError) -- it must NOT become a RuntimeReplay.
    malformed = SimpleNamespace(
        status="TURN_EXECUTED",
        conversation_id="chat-1",
        persisted=SimpleNamespace(replayed=False, turn=None),
    )
    result = None
    with pytest.raises(AttributeError):
        result = adapt_turn_outcome(malformed, AI_ACTIVE)
    assert not isinstance(result, RuntimeReplay)


def test_contradictory_replay_flag_with_a_turn_is_not_a_replay():
    # ``replayed=True`` WITH a present turn is contradictory/malformed and
    # must NOT be classified as a replay; it follows the existing turn path.
    turn = SimpleNamespace(
        assistant_text="Шаг процедуры.",
        decision=_routing(RouteAction.RECOMMEND_STEP),
    )
    malformed = SimpleNamespace(
        status="TURN_EXECUTED",
        conversation_id="chat-1",
        persisted=SimpleNamespace(replayed=True, turn=turn),
    )
    result = adapt_turn_outcome(malformed, AI_ACTIVE)

    assert isinstance(result, RuntimeTurn)
    assert not isinstance(result, RuntimeReplay)
    assert result.text == "Шаг процедуры."


def test_runtime_replay_invariants():
    assert RuntimeReplay(
        outcome_status="TURN_EXECUTED", conversation_id="chat-1"
    ).conversation_id == "chat-1"

    for bad in (
        dict(outcome_status="", conversation_id="chat-1"),
        dict(outcome_status="   ", conversation_id="chat-1"),
        dict(outcome_status="TURN_EXECUTED", conversation_id=""),
        dict(outcome_status="TURN_EXECUTED", conversation_id="  "),
    ):
        with pytest.raises(ValueError):
            RuntimeReplay(**bad)


def test_runtime_result_alias_covers_both_branches():
    results = (
        adapt_turn_outcome(_turn_executed(RouteAction.RECOMMEND_STEP), AI_ACTIVE),
        adapt_turn_outcome(_replayed_turn_executed(), AI_ACTIVE),
    )
    assert all(isinstance(r, RuntimeResult) for r in results)
    assert {type(r) for r in results} == {RuntimeTurn, RuntimeReplay}
