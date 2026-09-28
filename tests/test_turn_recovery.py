# -*- coding: utf-8 -*-
"""
HL-2c.3a -- the durable turn recovery record (OFFLINE proofs).

PURE: no network, no credentials, no database, no event loop, no LLM.

These tests pin the CONTRACT of :mod:`support.turn_recovery` -- the payload
that is stored atomically with the CAS and the PURE reproducer of the original
runtime turn:

  * the stored field set is CLOSED and fully SCALAR (no ``next_step`` /
    ``transition_type``, which the runtime projection never reads);
  * the reproduction of a stored payload is EXACTLY the projection of the
    ORIGINAL turn for EVERY state-dependent case (escalate / clarify /
    recommend / resolve / out-of-scope, AI-owned and human-owned) -- proving
    the payload is SUFFICIENT without re-running anything;
  * the reproducer consults the PAYLOAD ALONE (never the current conversation
    state), which is why the projection snapshot is stored;
  * ABSENT (``None``) and CORRUPT (a typed, LOUD failure) are different;
  * opaque ids are carried verbatim and are never parsed;
  * the module cannot recompute a turn (no orchestrator / analyzer /
    knowledge / generator / executor / session dependency);
  * the content-free replay vocabulary (``RuntimeReplay``) is NOT widened.
"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from types import SimpleNamespace

import pytest

from support.domain import (
    ConversationMode,
    HandoffStatus,
    SupportStatus,
)
from support.procedure_state import ProcedureState
from support.router import (
    ClarifyReason,
    EscalateReason,
    RouteAction,
    RoutingDecision,
)
from support.runtime_outcome import (
    RuntimeReplay,
    RuntimeTurn,
    adapt_turn_outcome,
)
from support.turn_executor import PersistedTurnResult
from support.turn_recovery import (
    TurnRecoveryCorruption,
    TurnRecoveryPayload,
    reproduce_runtime_turn,
)

# The canonical, whitelisted FSM triples used below.
AI_ACTIVE = (ConversationMode.AI, SupportStatus.ACTIVE, None)
AI_WAITING_USER = (
    ConversationMode.AI,
    SupportStatus.ACTIVE,
    HandoffStatus.WAITING_USER,
)
HUMAN_WAITING_OPERATOR = (
    ConversationMode.HUMAN,
    SupportStatus.ACTIVE,
    HandoffStatus.WAITING_OPERATOR,
)

_ALL_STATES = [AI_ACTIVE, AI_WAITING_USER, HUMAN_WAITING_OPERATOR]

_EXECUTION_ID = "tg:4711"
_CONVERSATION_ID = "chat-1"
_REVISION = 7


class _Turn:
    """Structural ``TurnResult`` double (only what recovery/§projection read).

    The engine's ``TurnResult`` is not persistable and carries
    ``observation`` / ``response`` / ``knowledge`` -- none of which the
    runtime projection reads from a ``TURN_EXECUTED`` turn.
    """

    def __init__(self, text="Reseat the display cable.", decision=None):
        self.assistant_text = text
        self.decision = decision
        self.state = ProcedureState()


def _capture(turn, *, state=AI_ACTIVE, execution_id=_EXECUTION_ID,
             conversation_id=_CONVERSATION_ID) -> TurnRecoveryPayload:
    return TurnRecoveryPayload.capture(
        execution_id=execution_id,
        conversation_id=conversation_id,
        turn=turn,
        projection_state=state,
    )


def _original(turn, state) -> RuntimeTurn:
    """The projection of the ORIGINAL, non-replay turn (the reference)."""
    outcome = SimpleNamespace(
        status="TURN_EXECUTED",
        conversation_id=_CONVERSATION_ID,
        persisted=PersistedTurnResult(
            conversation_id=_CONVERSATION_ID,
            turn=turn,
            state=ProcedureState(),
            revision=_REVISION,
        ),
    )
    result = adapt_turn_outcome(outcome, state)
    assert isinstance(result, RuntimeTurn)
    return result


def _replayed(payload) -> PersistedTurnResult:
    return PersistedTurnResult(
        conversation_id=_CONVERSATION_ID,
        turn=None,
        state=ProcedureState(),
        revision=_REVISION,
        replayed=True,
        recovery=payload,
    )


# ============================================================
# 1. THE STORED FIELD SET IS CLOSED, SCALAR AND MINIMAL
# ============================================================

def test_the_stored_field_set_is_closed_and_fully_scalar():
    names = {field.name for field in fields(TurnRecoveryPayload)}
    assert names == {
        "execution_id",
        "conversation_id",
        "text",
        "action",
        "projection_mode",
        "projection_support_status",
        "escalate_reason",
        "clarify_reason",
        "projection_handoff_status",
    }
    # The projection NEVER reads these, so storing them would be redundant
    # (and would risk freezing fields no recovery actually needs).
    assert "next_step" not in names
    assert "transition_type" not in names

    payload = _capture(
        _Turn(
            text="Step one.",
            decision=RoutingDecision(
                action=RouteAction.RECOMMEND_STEP,
                next_step="STEP_B",
                transition_type="attempted_failed",
            ),
        )
    )
    assert payload.action == RouteAction.RECOMMEND_STEP.value
    assert payload.text == "Step one."
    # Every value is a plain scalar (str / None), never a domain object.
    for value in (
        payload.execution_id,
        payload.conversation_id,
        payload.text,
        payload.action,
        payload.projection_mode,
        payload.projection_support_status,
        payload.escalate_reason,
        payload.clarify_reason,
        payload.projection_handoff_status,
    ):
        assert value is None or isinstance(value, str)


def test_the_payload_is_frozen_and_the_ids_are_carried_verbatim():
    opaque = "tg: 0042/\u00e9v\u00e8nement#1 (weird, unparsed)"
    payload = _capture(
        _Turn(decision=RoutingDecision(action=RouteAction.RESOLVE)),
        execution_id=opaque,
    )
    # NEVER parsed, normalised, hashed or rewritten.
    assert payload.execution_id == opaque
    assert payload.conversation_id == _CONVERSATION_ID

    with pytest.raises(FrozenInstanceError):
        payload.__setattr__("text", "tampered")


@pytest.mark.parametrize("field", ["execution_id", "conversation_id", "text"])
def test_blank_scalars_are_rejected_loudly(field):
    payload = _capture(
        _Turn(decision=RoutingDecision(action=RouteAction.RESOLVE))
    )
    with pytest.raises(TurnRecoveryCorruption):
        TurnRecoveryPayload(
            **{
                **{
                    name: getattr(payload, name)
                    for name in (
                        "execution_id",
                        "conversation_id",
                        "text",
                        "action",
                        "projection_mode",
                        "projection_support_status",
                        "escalate_reason",
                        "clarify_reason",
                        "projection_handoff_status",
                    )
                },
                field: "   ",
            }
        )


# ============================================================
# 2. REPRODUCTION IS EXACTLY THE ORIGINAL PROJECTION
# ============================================================

@pytest.mark.parametrize("state", _ALL_STATES)
@pytest.mark.parametrize(
    "decision",
    [
        RoutingDecision(
            action=RouteAction.RECOMMEND_STEP,
            next_step="STEP_B",
            transition_type="attempted_failed",
        ),
        RoutingDecision(action=RouteAction.RESOLVE),
        RoutingDecision(action=RouteAction.OUT_OF_SCOPE),
        RoutingDecision(
            action=RouteAction.ESCALATE,
            escalate_reason=EscalateReason.PROCEDURE_TERMINAL,
        ),
        RoutingDecision(
            action=RouteAction.CLARIFY,
            clarify_reason=ClarifyReason.PENDING_CLARIFICATION,
        ),
    ],
    ids=lambda d: d.action.value,
)
def test_reproduction_equals_the_original_projection(decision, state):
    """For EVERY decision and EVERY state the stored payload is SUFFICIENT.

    The reference is the SAME single projection owner a NEW execution uses
    (``adapt_turn_outcome``), so equality proves recovery reproduces the
    original turn EXACTLY -- including the STATE-DEPENDENT
    ``fsm_event`` / ``case_action`` -- from the stored payload alone.
    """
    turn = _Turn(text="Reseat the display cable.", decision=decision)

    reproduced = reproduce_runtime_turn(_capture(turn, state=state))

    assert isinstance(reproduced, RuntimeTurn)
    assert reproduced == _original(turn, state)
    assert reproduced.text == turn.assistant_text


def test_the_reproducer_never_consults_the_current_conversation_state():
    """The stored SNAPSHOT decides, even if the live state has moved on.

    An ESCALATE projects ``ESCALATE_NEW``/``CREATE_NEW`` from ``AI_ACTIVE``
    but ``USER_MESSAGE_HUMAN``/``NONE`` from a human-owned conversation. The
    payload captured under ``AI_ACTIVE`` must keep projecting the formER.
    """
    turn = _Turn(
        text="I am escalating.",
        decision=RoutingDecision(
            action=RouteAction.ESCALATE,
            escalate_reason=EscalateReason.STEPS_EXHAUSTED,
        ),
    )
    payload = _capture(turn, state=AI_ACTIVE)

    # The CURRENT (live) state is now human-owned -- it must NOT matter.
    assert payload.conversation_state() == AI_ACTIVE
    reproduced = reproduce_runtime_turn(payload)

    assert reproduced == _original(turn, AI_ACTIVE)
    assert reproduced.fsm_event is not None
    assert reproduced.fsm_event.value == "ESCALATE_NEW"
    assert reproduced.case_action.value == "CREATE_NEW"
    assert reproduced.reason == EscalateReason.STEPS_EXHAUSTED.value


def test_escalation_resume_is_reproduced_from_a_waiting_user_snapshot():
    turn = _Turn(
        text="Escalating again.",
        decision=RoutingDecision(
            action=RouteAction.ESCALATE,
            escalate_reason=EscalateReason.PROCEDURE_TERMINAL,
        ),
    )
    reproduced = reproduce_runtime_turn(_capture(turn, state=AI_WAITING_USER))

    assert reproduced.fsm_event.value == "ESCALATE_RESUME"
    assert reproduced.case_action.value == "RESUME"
    assert reproduced == _original(turn, AI_WAITING_USER)


def test_a_human_owned_snapshot_reproduces_the_human_event():
    turn = _Turn(
        text="Handled by a human.",
        decision=RoutingDecision(action=RouteAction.RECOMMEND_STEP),
    )
    reproduced = reproduce_runtime_turn(
        _capture(turn, state=HUMAN_WAITING_OPERATOR)
    )

    assert reproduced.fsm_event.value == "USER_MESSAGE_HUMAN"
    assert reproduced.case_action.value == "NONE"
    assert reproduced == _original(turn, HUMAN_WAITING_OPERATOR)


def test_clarify_and_escalate_reasons_survive_verbatim():
    escalate = _capture(
        _Turn(
            decision=RoutingDecision(
                action=RouteAction.ESCALATE,
                escalate_reason=EscalateReason.INTERNAL_INCONSISTENCY,
            )
        )
    )
    clarify = _capture(
        _Turn(
            decision=RoutingDecision(
                action=RouteAction.CLARIFY,
                clarify_reason=ClarifyReason.STEP_OUTCOME_UNKNOWN,
            )
        )
    )

    assert escalate.escalate_reason == "internal_inconsistency"
    assert escalate.clarify_reason is None
    assert clarify.clarify_reason == "step_outcome_unknown"
    assert clarify.escalate_reason is None

    assert reproduce_runtime_turn(escalate).reason == "internal_inconsistency"
    assert reproduce_runtime_turn(clarify).reason == "step_outcome_unknown"
    assert reproduce_runtime_turn(escalate).decision == "ESCALATE"
    assert reproduce_runtime_turn(clarify).decision == "CLARIFY"


# ============================================================
# 3. ABSENT != CORRUPT (both fail closed, differently)
# ============================================================

def test_a_normal_turn_cannot_carry_recovery():
    payload = _capture(
        _Turn(decision=RoutingDecision(action=RouteAction.RESOLVE))
    )
    with pytest.raises(ValueError):
        PersistedTurnResult(
            conversation_id=_CONVERSATION_ID,
            turn=_Turn(decision=RoutingDecision(action=RouteAction.RESOLVE)),
            state=ProcedureState(),
            revision=1,
            recovery=payload,
        )


def test_a_replay_may_carry_recovery_or_none():
    payload = _capture(
        _Turn(decision=RoutingDecision(action=RouteAction.RESOLVE))
    )
    assert _replayed(payload).recovery is payload
    # ABSENT is legitimate: the winner simply stored no record.
    assert _replayed(None).recovery is None


def test_corruption_is_a_loud_typed_failure_not_a_silent_none():
    assert issubclass(TurnRecoveryCorruption, Exception)
    assert not issubclass(TurnRecoveryCorruption, (ValueError, AttributeError))

    payload = _capture(
        _Turn(decision=RoutingDecision(action=RouteAction.RESOLVE))
    )
    # A record that EXISTS but cannot be interpreted must fail LOUDLY: this is
    # exactly the shape a damaged / partially-written row yields.
    corrupt = TurnRecoveryPayload(
        execution_id=payload.execution_id,
        conversation_id=payload.conversation_id,
        text=payload.text,
        action="NOT_A_ROUTE_ACTION",
        projection_mode=payload.projection_mode,
        projection_support_status=payload.projection_support_status,
    )
    with pytest.raises(TurnRecoveryCorruption):
        reproduce_runtime_turn(corrupt)


def test_a_malformed_projection_snapshot_is_rejected():
    payload = _capture(
        _Turn(decision=RoutingDecision(action=RouteAction.RESOLVE))
    )
    with pytest.raises(TurnRecoveryCorruption):
        TurnRecoveryPayload(
            execution_id=payload.execution_id,
            conversation_id=payload.conversation_id,
            text=payload.text,
            action=payload.action,
            projection_mode="not-a-mode",
            projection_support_status=payload.projection_support_status,
        ).conversation_state()


def test_capture_rejects_a_turn_it_cannot_recover():
    with pytest.raises(TurnRecoveryCorruption):
        _capture(_Turn(text="   ", decision=RoutingDecision(
            action=RouteAction.RESOLVE
        )))
    with pytest.raises((TurnRecoveryCorruption, AttributeError)):
        _capture(_Turn(text="x", decision=None))


# ============================================================
# 4. IT CANNOT RECOMPUTE A TURN
# ============================================================

def test_the_recovery_module_cannot_recompute_a_turn():
    """Recovery is a PURE function of the stored payload.

    Structurally pinned: the module may not depend on the analyzer, the
    knowledge / retrieval stack, the generator, the orchestrator, the
    executor / dispatcher or any SQL session -- so a duplicate delivery can
    never silently re-run the LLM, retrieval or the procedure.
    """
    source = (
        Path(__file__).resolve().parents[1] / "support/turn_recovery.py"
    ).read_text(encoding="utf-8")

    # The EXACT import surface: stdlib plumbing plus three PURE domain
    # modules. Nothing that could call a provider, a session or the engine.
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")

    assert imported == {
        "__future__",
        "dataclasses",
        "typing",
        "support.domain",
        "support.router",
        "support.runtime_outcome",
    }

    # ...and NO import may reach the LLM / retrieval / persistence / turn
    # infrastructure, however it is spelled.
    for module in imported:
        for token in (
            "orchestrator",
            "alice_analyzer",
            "knowledge",
            "response_generator",
            "yandex",
            "sqlalchemy",
            "turn_executor",
            "turn_dispatch",
            "support_core",
        ):
            assert token not in module, module


def test_the_content_free_replay_vocabulary_is_not_widened():
    """The recovered answer text lives in the DURABLE record, not the marker.

    ``RuntimeReplay`` must stay structurally effect-free: recovery adds NO
    field to it (an engine replay still cannot express wording / decision /
    FSM effect).
    """
    assert {field.name for field in fields(RuntimeReplay)} == {
        "outcome_status",
        "conversation_id",
    }
