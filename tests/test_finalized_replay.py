# -*- coding: utf-8 -*-
"""
HL-2c.3b.1 -- the LEVEL A finalized replay contract (OFFLINE proofs).

PURE: no network, no credentials, no database, no event loop, no LLM.

These tests pin the CONTRACT of :mod:`support.finalized_replay`:

  * the durable snapshot field set is CLOSED and fully SCALAR;
  * reproduction is a PURE, EXACT function of the snapshot -- the returned
    body is the ORIGINAL finalized body, with no re-derivation from the
    mutable CURRENT conversation state;
  * ABSENT and CORRUPT are DIFFERENT typed failures, both fail closed;
  * a legacy finalized record (``retrieval_used`` never persisted) is CORRUPT
    and raises LOUDLY rather than guessing a body;
  * the module cannot re-run the LLM / retrieval / procedure / FSM.
"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from support.finalized_replay import (
    FINALIZED_OUTCOMES,
    KNOWN_OUTCOMES,
    FinalizedReplayCorruption,
    FinalizedReplayUnavailable,
    FinalizedTurnSnapshot,
    is_finalized_outcome,
    reproduce_finalized_response,
)

# ============================================================
# CANONICAL SNAPSHOTS
# ============================================================

_AI_ANSWERED = {
    "inbound_event_id": "tg:4711",
    "conversation_id": "chat-1",
    "outcome": "answered",
    "assistant_message_id": 42,
    "answer": "Reseat the display cable.",
    "decision": "ANSWER",
    "reason": "RESOLVED_BY_L1",
    "case_action": "NONE",
    "retrieval_used": False,
    "conversation_mode": "ai",
    "support_status": "active",
    "handoff_status": None,
    "external_case_id": None,
}

_FORWARDED = {
    "inbound_event_id": "tg:4712",
    "conversation_id": "chat-1",
    "outcome": "forwarded",
    "assistant_message_id": None,
    "answer": "",
    "decision": "",
    "reason": "",
    "case_action": "HUMAN_FORWARD",
    "retrieval_used": False,
    "conversation_mode": "human",
    "support_status": "active",
    "handoff_status": "waiting_operator",
    "external_case_id": "ANGIO-9",
}


def _snapshot(**overrides) -> FinalizedTurnSnapshot:
    return FinalizedTurnSnapshot(**{**_AI_ANSWERED, **overrides})


# ============================================================
# 1. THE DURABLE FIELD SET IS CLOSED AND FULLY SCALAR
# ============================================================

def test_the_snapshot_field_set_is_closed_and_fully_scalar():
    names = {field.name for field in fields(FinalizedTurnSnapshot)}
    assert names == {
        "inbound_event_id",
        "conversation_id",
        "outcome",
        "assistant_message_id",
        "answer",
        "decision",
        "reason",
        "case_action",
        "retrieval_used",
        "conversation_mode",
        "support_status",
        "handoff_status",
        "external_case_id",
    }
    # The answer is NOT duplicated as a durable column: it is linked by the
    # assistant message id (the DB reader resolves the text).
    snapshot = _snapshot()
    assert snapshot.assistant_message_id == 42
    assert snapshot.answer == "Reseat the display cable."

    # Every durable value is a plain scalar -- never a domain object.
    for name in names - {"answer"}:
        value = getattr(snapshot, name)
        assert value is None or isinstance(value, (str, bool, int)), name
    assert isinstance(snapshot.answer, str)


def test_the_snapshot_is_frozen():
    snapshot = _snapshot()
    with pytest.raises(FrozenInstanceError):
        snapshot.__setattr__("answer", "tampered")


def test_the_outcome_sets_are_consistent():
    assert FINALIZED_OUTCOMES == {"answered", "escalated", "forwarded"}
    assert FINALIZED_OUTCOMES <= KNOWN_OUTCOMES
    assert "received" in KNOWN_OUTCOMES
    assert is_finalized_outcome("answered")
    assert not is_finalized_outcome("received")
    assert not is_finalized_outcome(None)


# ============================================================
# 2. REPRODUCTION IS THE EXACT ORIGINAL BODY
# ============================================================

def test_reproduction_returns_the_exact_original_body():
    snapshot = _snapshot(retrieval_used=True, external_case_id="ANGIO-1")

    response = reproduce_finalized_response(snapshot)

    assert response.ok is True
    assert response.answer == snapshot.answer
    assert response.retrieval_used is True
    assert response.decision == "ANSWER"
    assert response.reason == "RESOLVED_BY_L1"
    assert response.conversation_id == "chat-1"
    assert response.conversation_mode == "ai"
    assert response.support_status == "active"
    assert response.handoff_status is None
    assert response.case_action == "NONE"
    assert response.external_case_id == "ANGIO-1"


def test_reproduction_marks_the_body_deduplicated():
    # The ONLY caller is a duplicate delivery: the first one finalized.
    assert reproduce_finalized_response(_snapshot()).deduplicated is True


def test_retrieval_used_is_replayed_verbatim_not_hardcoded():
    # THE fidelity gap this slice closes: the old duplicate branch hardcoded
    # ``retrieval_used: False`` regardless of the original turn.
    assert reproduce_finalized_response(
        _snapshot(retrieval_used=False)
    ).retrieval_used is False
    assert reproduce_finalized_response(
        _snapshot(retrieval_used=True)
    ).retrieval_used is True


def test_every_projected_field_is_replayed_verbatim_from_the_snapshot():
    snapshot = _snapshot(
        decision="ESCALATE",
        reason="STEPS_EXHAUSTED",
        case_action="CREATE_NEW",
        conversation_mode="ai",
        support_status="active",
        handoff_status=None,
        external_case_id="ANGIO-77",
    )
    response = reproduce_finalized_response(snapshot)
    assert (
        response.decision,
        response.reason,
        response.case_action,
        response.conversation_mode,
        response.support_status,
        response.handoff_status,
        response.external_case_id,
    ) == (
        "ESCALATE",
        "STEPS_EXHAUSTED",
        "CREATE_NEW",
        "ai",
        "active",
        None,
        "ANGIO-77",
    )


def test_the_reproducer_never_consults_the_current_conversation_state():
    # A snapshot captured under AI/active replays as AI/active even though the
    # live conversation has since become human-owned.
    snapshot = _snapshot(
        conversation_mode="ai",
        support_status="active",
        handoff_status=None,
    )
    response = reproduce_finalized_response(snapshot)
    assert response.conversation_mode == "ai"
    assert response.support_status == "active"
    assert response.handoff_status is None


def test_a_forwarded_turn_replays_empty_answer_and_human_forward():
    response = reproduce_finalized_response(FinalizedTurnSnapshot(**_FORWARDED))
    assert response.answer == ""
    assert response.decision == ""
    assert response.reason == ""
    assert response.case_action == "HUMAN_FORWARD"
    assert response.conversation_mode == "human"
    assert response.handoff_status == "waiting_operator"
    assert response.external_case_id == "ANGIO-9"


# ============================================================
# 3. ABSENT != CORRUPT (both fail closed, differently)
# ============================================================

def test_in_flight_outcome_is_unavailable_not_corruption():
    with pytest.raises(FinalizedReplayUnavailable):
        _snapshot(outcome="received")


def test_a_non_replayable_outcome_is_corruption():
    with pytest.raises(FinalizedReplayCorruption):
        _snapshot(outcome="duplicate")


def test_a_missing_retrieval_used_is_corruption():
    # A legacy finalized row never persisted ``retrieval_used`` (None): an
    # EXACT body is impossible, so we fail closed LOUDLY.
    with pytest.raises(FinalizedReplayCorruption):
        _snapshot(retrieval_used=None)


def test_a_missing_lifecycle_snapshot_is_corruption():
    with pytest.raises(FinalizedReplayCorruption):
        _snapshot(conversation_mode=None)
    with pytest.raises(FinalizedReplayCorruption):
        _snapshot(support_status=None)


def test_an_invalid_lifecycle_triple_is_corruption():
    with pytest.raises(FinalizedReplayCorruption):
        _snapshot(conversation_mode="not-a-mode")
    # A recognized value in a FORBIDDEN combination (whitelist miss).
    with pytest.raises(FinalizedReplayCorruption):
        _snapshot(conversation_mode="ai", handoff_status="waiting_operator")


def test_an_ai_turn_requires_an_assistant_message_and_answer():
    with pytest.raises(FinalizedReplayCorruption):
        _snapshot(assistant_message_id=None)
    with pytest.raises(FinalizedReplayCorruption):
        _snapshot(answer="   ")
    with pytest.raises(FinalizedReplayCorruption):
        _snapshot(decision="")
    with pytest.raises(FinalizedReplayCorruption):
        _snapshot(reason="   ")


def test_a_forwarded_turn_cannot_carry_an_ai_answer_or_message():
    with pytest.raises(FinalizedReplayCorruption):
        FinalizedTurnSnapshot(**{**_FORWARDED, "assistant_message_id": 7})
    with pytest.raises(FinalizedReplayCorruption):
        FinalizedTurnSnapshot(**{**_FORWARDED, "answer": "an AI reply"})


def test_blank_identity_scalars_are_rejected_loudly():
    for field in ("inbound_event_id", "conversation_id", "case_action"):
        with pytest.raises(FinalizedReplayCorruption):
            _snapshot(**{field: "   "})


def test_the_two_failure_types_are_distinct_and_loud():
    assert issubclass(FinalizedReplayUnavailable, RuntimeError)
    assert issubclass(FinalizedReplayCorruption, RuntimeError)
    # A caller can tell "nothing finalized yet" from "damaged record".
    assert not issubclass(FinalizedReplayUnavailable, FinalizedReplayCorruption)
    assert not issubclass(FinalizedReplayCorruption, FinalizedReplayUnavailable)
    # Neither is a silent value error / attribute error.
    assert not issubclass(FinalizedReplayUnavailable, (ValueError, AttributeError))
    assert not issubclass(FinalizedReplayCorruption, (ValueError, AttributeError))


# ============================================================
# 4. IT CANNOT RE-RUN ANYTHING
# ============================================================

def test_the_module_cannot_recompute_a_turn_or_touch_persistence():
    """Replay is a PURE function of the snapshot.

    Structurally pinned: the module may not depend on the analyzer, the
    knowledge / retrieval stack, the generator, the orchestrator, the
    executor / dispatcher, ``support_core`` or any SQL session -- so a
    duplicate delivery can never silently re-run the LLM, retrieval or the
    procedure.
    """
    source = (
        Path(__file__).resolve().parents[1] / "support/finalized_replay.py"
    ).read_text(encoding="utf-8")

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
        "support.schemas",
    }

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
            "models",
            "idempotency",
        ):
            assert token not in module, module
