# -*- coding: utf-8 -*-
"""
HL-2c.3b.2 -- the Stateful Turn Runtime (OFFLINE proofs).

PURE: no network, no database, no LLM. These tests pin the CONTRACT of
:mod:`support.runtime_service` -- the ONE application runtime boundary that
composes the THREE delivery paths of ONE inbound turn:

  * FRESH (:meth:`StatefulTurnRuntime.execute_fresh_turn`) -- invokes
    ``SupportCore.handle_turn`` EXACTLY ONCE, forwards every input VERBATIM and
    adapts the outcome with ``adapt_turn_outcome`` -> ``FreshTurnDelivery``;
  * W3 RECOVERY (:meth:`StatefulTurnRuntime.resolve_duplicate`, LEVEL B) --
    reproduces the ORIGINAL turn from the durable ``turn_recovery`` record with
    NO ``SupportCore`` / provider call -> ``RecoveredTurnDelivery``;
  * FINALIZED DUPLICATE (:meth:`StatefulTurnRuntime.resolve_duplicate`,
    LEVEL A) -- reproduces the EXACT persisted body with NO ``SupportCore``
    call -> ``FinalizedDelivery``.

It also pins the FAILURE contract (``RuntimeRecoveryUnavailable`` for anything
durable-absent -- including an engine replay whose winner stored no record) and
the architecture contract (exactly ONE module may import ``support_core``; no
lifecycle mapping re-derived here; no new persistence).

The seam doubles are structural (they satisfy the SAME
``InboundEventService`` / ``DurableRecoveryStore`` Protocols the production
implementations do) and the outcome values are REAL ``SupportTurnOutcome`` /
``PersistedTurnResult`` objects, so the projection under test is the production
one.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from support.domain import AI_ACTIVE, AI_ACTIVE_WAITING_USER
from support.finalized_replay import (
    FinalizedReplayCorruption,
    FinalizedReplayUnavailable,
    FinalizedTurnSnapshot,
    reproduce_finalized_response,
)
from support.idempotency import IdempotencyService
from support.procedure_state import ProcedureState
from support.router import (
    EscalateReason,
    RouteAction,
    RoutingDecision,
)
from support.runtime_outcome import (
    RuntimeReplay,
    RuntimeTurn,
    RuntimeTurnKind,
    adapt_turn_outcome,
)
from support.runtime_service import (
    DurableRecoveryStore,
    FinalizedDelivery,
    FreshTurnDelivery,
    InboundEventService,
    RecoveredTurnDelivery,
    RuntimeDelivery,
    RuntimeRecoveryUnavailable,
    StatefulTurnRuntime,
)
from support.schemas import ClaimResult
from support.support_core import (
    TURN_EXECUTED,
    SupportCore,
    SupportTurnOutcome,
)
from support.turn_executor import PersistedTurnResult
from support.turn_recovery import TurnRecoveryPayload, reproduce_runtime_turn
from support.turn_recovery_store import TurnRecoveryStore

import support.runtime_service as rs


# ============================================================
# DOUBLES (structural: satisfy the SAME seams as production)
# ============================================================

class _Session:
    def __init__(self, events: list) -> None:
        self._events = events

    async def __aenter__(self):
        self._events.append("open")
        return self

    async def __aexit__(self, *exc_info):
        self._events.append("close")
        return False

    async def commit(self):
        self._events.append("commit")

    async def rollback(self):
        self._events.append("rollback")


class _SessionFactory:
    """Session-factory double: opening it records the transaction boundary."""

    def __init__(self, events: list) -> None:
        self._events = events
        self.opened = 0

    def __call__(self):
        self.opened += 1
        return _Session(self._events)


class _FakeInbound:
    """``InboundEventService`` double: records the ONE replay read."""

    def __init__(self, *, snapshot=None, error=None, claim_result=None):
        self.snapshot = snapshot
        self.error = error
        self.claim_result = claim_result
        self.claim_calls: list = []
        self.replay_reads: list = []

    async def claim(self, inbound_event_id, channel):
        self.claim_calls.append((inbound_event_id, channel))
        return self.claim_result

    async def load_finalized_replay(self, inbound_event_id):
        self.replay_reads.append(inbound_event_id)
        if self.error is not None:
            raise self.error
        return self.snapshot


class _FakeRecovery:
    """``DurableRecoveryStore`` double. ``save`` must never be called here."""

    def __init__(self, payload=None, *, error=None):
        self.payload = payload
        self.error = error
        self.reads: list = []
        self.saves: list = []

    async def read(self, *, execution_id, conversation_id):
        self.reads.append((execution_id, conversation_id))
        if self.error is not None:
            raise self.error
        return self.payload

    async def save(self, payload):
        self.saves.append(payload)


class _FakeCore:
    """``SupportCore`` double: records the ONE delegated ``handle_turn`` call.

    Structural -- the runtime boundary depends on ``SupportCore``'s public
    ``handle_turn`` contract, which this satisfies. ``calls`` is the proof that
    the duplicate paths never touch the engine.
    """

    def __init__(self, outcome=None, *, error=None) -> None:
        self.outcome = outcome
        self.error = error
        self.calls: list = []

    async def handle_turn(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.outcome


class _Turn:
    """Structural engine-turn double (only what projection / recovery read)."""

    def __init__(self, text, decision):
        self.assistant_text = text
        self.decision = decision
        self.state = ProcedureState()


# ============================================================
# BUILDERS
# ============================================================

def _escalate() -> RoutingDecision:
    return RoutingDecision(
        action=RouteAction.ESCALATE,
        escalate_reason=EscalateReason.PROCEDURE_TERMINAL,
    )


def _recommend_step() -> RoutingDecision:
    return RoutingDecision(
        action=RouteAction.RECOMMEND_STEP, next_step="STEP_A"
    )


def _snapshot(**overrides) -> FinalizedTurnSnapshot:
    base = dict(
        inbound_event_id="tg:1",
        conversation_id="chat-1",
        outcome="answered",
        assistant_message_id=7,
        answer="Reseat the cable firmly.",
        decision="ANSWER",
        reason="PROCEDURE_STEP",
        case_action="NONE",
        retrieval_used=True,
        conversation_mode="ai",
        support_status="active",
        handoff_status=None,
        external_case_id="ANGIO-1",
    )
    base.update(overrides)
    return FinalizedTurnSnapshot(**base)


def _payload(*, execution_id="tg:1", conversation_id="chat-1", state=AI_ACTIVE):
    turn = _Turn(
        text="I am escalating this to a human.", decision=_escalate()
    )
    return TurnRecoveryPayload.capture(
        execution_id=execution_id,
        conversation_id=conversation_id,
        turn=turn,
        projection_state=state,
    )


def _fresh_outcome(
    *, conversation_id="chat-1", text="Reseat the cable firmly."
) -> SupportTurnOutcome:
    """A REAL ``SupportTurnOutcome`` for a genuine (non-replay) turn."""
    return SupportTurnOutcome(
        status=TURN_EXECUTED,
        conversation_id=conversation_id,
        persisted=PersistedTurnResult(
            conversation_id=conversation_id,
            turn=_Turn(text, _recommend_step()),
            state=ProcedureState(),
            revision=1,
        ),
    )


def _escalating_outcome(*, conversation_id="chat-1") -> SupportTurnOutcome:
    return SupportTurnOutcome(
        status=TURN_EXECUTED,
        conversation_id=conversation_id,
        persisted=PersistedTurnResult(
            conversation_id=conversation_id,
            turn=_Turn("I am escalating this to a human.", _escalate()),
            state=ProcedureState(),
            revision=1,
        ),
    )


def _replay_outcome(
    *, conversation_id="chat-1", recovery=None, execution_id="tg:1"
) -> SupportTurnOutcome:
    """The ONE legitimate engine-replay shape (``replayed`` + no turn)."""
    return SupportTurnOutcome(
        status=TURN_EXECUTED,
        conversation_id=conversation_id,
        persisted=PersistedTurnResult(
            conversation_id=conversation_id,
            turn=None,
            state=ProcedureState(),
            revision=1,
            replayed=True,
            recovery=recovery,
        ),
    )


def _runtime(inbound=None, recovery=None, core=None, events=None):
    events = events if events is not None else []
    sessions = _SessionFactory(events)
    runtime = StatefulTurnRuntime(
        session_factory=sessions,
        inbound_event_factory=lambda session: (
            inbound if inbound is not None else _FakeInbound()
        ),
        recovery_store_factory=lambda session: (
            recovery if recovery is not None else _FakeRecovery()
        ),
        support_core=core,
    )
    return runtime, sessions, events


# ============================================================
# 1. THE SEAMS DEFAULT TO THE EXISTING IMPLEMENTATIONS
# ============================================================

def test_default_factories_are_the_existing_implementations():
    params = inspect.signature(StatefulTurnRuntime.__init__).parameters
    assert params["inbound_event_factory"].default is IdempotencyService
    assert params["recovery_store_factory"].default is TurnRecoveryStore
    assert params["support_core"].default is None  # lazily composed


def test_existing_production_objects_satisfy_the_runtime_seams():
    # Structural (runtime_checkable) proof: the runtime depends on the
    # EXISTING code, not on a re-implementation.
    assert isinstance(IdempotencyService(object()), InboundEventService)
    assert isinstance(TurnRecoveryStore(object()), DurableRecoveryStore)


def test_the_default_support_core_is_the_real_composition():
    """The ONE sanctioned composition dependency: the REAL ``SupportCore``."""
    events: list = []
    sessions = _SessionFactory(events)
    runtime = StatefulTurnRuntime(session_factory=sessions)

    assert sessions.opened == 0  # composing resolves nothing
    core = runtime.support_core
    assert isinstance(core, SupportCore)
    assert core is runtime.support_core  # built once, then reused
    # It shares THIS boundary's session factory (one configured database),
    # and resolving it still opened no session.
    assert core.session_factory is sessions
    assert sessions.opened == 0

    injected = _FakeCore()
    assert (
        StatefulTurnRuntime(
            session_factory=sessions, support_core=injected
        ).support_core
        is injected
    )


# ============================================================
# 2. CLAIM (TX1) -- forwards verbatim, commits its own transaction
# ============================================================

@pytest.mark.asyncio
async def test_claim_forwards_and_commits_its_own_transaction():
    inbound = _FakeInbound(
        claim_result=ClaimResult(claimed=True, outcome="received")
    )
    runtime, sessions, events = _runtime(inbound=inbound)

    result = await runtime.claim(inbound_event_id="tg:1", channel="telegram")

    assert result == ClaimResult(claimed=True, outcome="received")
    assert inbound.claim_calls == [("tg:1", "telegram")]
    assert sessions.opened == 1
    assert events == ["open", "commit", "close"]


# ============================================================
# 3. PATH 1 -- FRESH DELIVERY (SupportCore exactly once)
# ============================================================

@pytest.mark.asyncio
async def test_fresh_delivery_calls_support_core_exactly_once():
    outcome = _fresh_outcome()
    core = _FakeCore(outcome)
    runtime, sessions, _ = _runtime(core=core)

    delivery = await runtime.execute_fresh_turn(
        conversation_id="chat-1",
        message="black screen on boot",
        projection_state=AI_ACTIVE,
    )

    assert isinstance(delivery, FreshTurnDelivery)
    assert isinstance(delivery, RuntimeDelivery)
    assert len(core.calls) == 1  # EXACTLY once, never twice
    assert sessions.opened == 0  # the boundary opened no session of its own


@pytest.mark.asyncio
async def test_fresh_delivery_returns_the_adapted_runtime_turn():
    outcome = _fresh_outcome(text="Reseat the cable firmly.")
    core = _FakeCore(outcome)
    runtime, _, _ = _runtime(core=core)

    delivery = await runtime.execute_fresh_turn(
        conversation_id="chat-1",
        message="black screen",
        projection_state=AI_ACTIVE,
    )

    assert isinstance(delivery, FreshTurnDelivery)
    turn = delivery.turn
    assert isinstance(turn, RuntimeTurn)
    # EXACTLY the production adaptation of EXACTLY the core's outcome.
    assert turn == adapt_turn_outcome(outcome, AI_ACTIVE)
    assert turn.kind is RuntimeTurnKind.USER_RESPONSE
    assert turn.outcome_status == TURN_EXECUTED
    assert turn.conversation_id == "chat-1"
    assert turn.text == "Reseat the cable firmly."
    assert turn.decision == "ANSWER"
    assert turn.reason == "PROCEDURE_STEP"
    assert turn.fsm_event.value == "USER_MESSAGE_AI"
    assert turn.case_action.value == "NONE"


@pytest.mark.asyncio
async def test_fresh_delivery_forwards_every_input_verbatim():
    core = _FakeCore(_fresh_outcome())
    runtime, _, _ = _runtime(core=core)
    history = [
        {"role": "user", "text": "hi"},
        {"role": "assistant", "text": "how can I help?"},
    ]
    state = AI_ACTIVE_WAITING_USER

    await runtime.execute_fresh_turn(
        conversation_id="chat-1",
        message="black screen on boot",
        projection_state=state,
        history=history,
        execution_id="tg:opaque-123",
    )

    call = core.calls[0]
    assert set(call) == {
        "conversation_id",
        "message",
        "history",
        "execution_id",
        "projection_state",
    }
    assert call["conversation_id"] == "chat-1"
    assert call["message"] == "black screen on boot"  # unmodified
    assert call["history"] is history  # opaque pass-through, by identity
    assert call["execution_id"] == "tg:opaque-123"  # verbatim, unmodified
    assert call["projection_state"] is state


@pytest.mark.asyncio
async def test_fresh_delivery_projects_from_the_forwarded_projection_state():
    """The SAME outcome projects differently per state -> the state is real."""
    core_ai = _FakeCore(_escalating_outcome())
    core_waiting = _FakeCore(_escalating_outcome())

    ai_runtime, _, _ = _runtime(core=core_ai)
    waiting_runtime, _, _ = _runtime(core=core_waiting)

    ai = await ai_runtime.execute_fresh_turn(
        conversation_id="chat-1",
        message="connect me to a human",
        projection_state=AI_ACTIVE,
    )
    waiting = await waiting_runtime.execute_fresh_turn(
        conversation_id="chat-1",
        message="connect me to a human",
        projection_state=AI_ACTIVE_WAITING_USER,
    )

    assert isinstance(ai, FreshTurnDelivery)
    assert isinstance(waiting, FreshTurnDelivery)
    assert ai.turn.fsm_event.value == "ESCALATE_NEW"
    assert ai.turn.case_action.value == "CREATE_NEW"
    assert waiting.turn.fsm_event.value == "ESCALATE_RESUME"
    assert waiting.turn.case_action.value == "RESUME"


@pytest.mark.asyncio
async def test_fresh_delivery_without_an_execution_id_stays_legacy():
    core = _FakeCore(_fresh_outcome())
    runtime, _, _ = _runtime(core=core)

    await runtime.execute_fresh_turn(
        conversation_id="chat-1",
        message="black screen",
        projection_state=AI_ACTIVE,
    )

    assert core.calls[0]["execution_id"] is None


@pytest.mark.asyncio
async def test_fresh_path_never_touches_the_durable_replay_seams():
    inbound = _FakeInbound(snapshot=_snapshot())
    recovery = _FakeRecovery(payload=_payload())
    runtime, _, _ = _runtime(
        inbound=inbound, recovery=recovery, core=_FakeCore(_fresh_outcome())
    )

    delivery = await runtime.execute_fresh_turn(
        conversation_id="chat-1",
        message="black screen",
        projection_state=AI_ACTIVE,
    )

    assert isinstance(delivery, FreshTurnDelivery)
    # A fresh turn reproduces NOTHING: no finalized replay read, no recovery
    # read, no write.
    assert inbound.replay_reads == []
    assert recovery.reads == []
    assert recovery.saves == []


# ============================================================
# 4. PATH 1b -- THE ENGINE REPLAY INSIDE THE FRESH PATH
# ============================================================

@pytest.mark.asyncio
async def test_engine_replay_is_resolved_through_durable_recovery():
    payload = _payload(execution_id="tg:1")
    expected = reproduce_runtime_turn(payload)
    core = _FakeCore(_replay_outcome(recovery=payload))
    runtime, _, _ = _runtime(core=core)

    delivery = await runtime.execute_fresh_turn(
        conversation_id="chat-1",
        message="black screen",
        projection_state=AI_ACTIVE,
        execution_id="tg:1",
    )

    # NOT a fresh turn: the ORIGINAL turn is reproduced from the durable
    # payload the engine loaded.
    assert isinstance(delivery, RecoveredTurnDelivery)
    assert not isinstance(delivery, FreshTurnDelivery)
    assert isinstance(delivery, RuntimeDelivery)
    assert delivery.turn == expected
    assert delivery.turn.text == "I am escalating this to a human."
    assert delivery.turn.decision == "ESCALATE"
    assert len(core.calls) == 1  # no second SupportCore call


@pytest.mark.asyncio
async def test_engine_replay_without_a_durable_record_fails_closed():
    core = _FakeCore(_replay_outcome(recovery=None))
    runtime, _, _ = _runtime(core=core)

    with pytest.raises(RuntimeRecoveryUnavailable) as excinfo:
        await runtime.execute_fresh_turn(
            conversation_id="chat-1",
            message="black screen",
            projection_state=AI_ACTIVE,
            execution_id="tg:1",
        )

    assert isinstance(excinfo.value, RuntimeError)
    assert len(core.calls) == 1  # fail closed, never re-run
    assert "tg:1" in str(excinfo.value)


def test_runtime_replay_stays_content_free():
    replay = adapt_turn_outcome(
        _replay_outcome(recovery=_payload()), AI_ACTIVE
    )

    assert isinstance(replay, RuntimeReplay)
    for attribute in (
        "text",
        "decision",
        "reason",
        "fsm_event",
        "case_action",
        "kind",
    ):
        assert not hasattr(replay, attribute), attribute


# ============================================================
# 5. PATH 3 -- LEVEL A FINALIZED REPLAY (the EXACT original body)
# ============================================================

@pytest.mark.asyncio
async def test_resolve_duplicate_reproduces_the_exact_finalized_body():
    snapshot = _snapshot(retrieval_used=True)
    inbound = _FakeInbound(snapshot=snapshot)
    recovery = _FakeRecovery()  # must never be consulted
    core = _FakeCore(_fresh_outcome())
    runtime, sessions, events = _runtime(
        inbound=inbound, recovery=recovery, core=core
    )

    delivery = await runtime.resolve_duplicate(
        inbound_event_id="tg:1", conversation_id="chat-1"
    )

    assert isinstance(delivery, FinalizedDelivery)
    assert isinstance(delivery, RuntimeDelivery)
    assert delivery.body == reproduce_finalized_response(snapshot)
    # retrieval_used is the PERSISTED value -- never hardcoded False.
    assert delivery.body.retrieval_used is True
    assert delivery.body.external_case_id == "ANGIO-1"
    assert delivery.body.deduplicated is True
    assert delivery.body.conversation_mode == "ai"

    # SupportCore was bypassed ENTIRELY; LEVEL B was never reached.
    assert core.calls == []
    assert recovery.reads == []
    assert recovery.saves == []
    assert events == ["open", "close"]


@pytest.mark.asyncio
async def test_resolve_duplicate_never_consults_level_b_on_a_finalized_replay():
    inbound = _FakeInbound(snapshot=_snapshot())
    recovery = _FakeRecovery(payload=_payload())
    runtime, _, _ = _runtime(inbound=inbound, recovery=recovery)

    await runtime.resolve_duplicate(
        inbound_event_id="tg:1", conversation_id="chat-1"
    )

    assert recovery.reads == []  # LEVEL A wins; recovery is not even read


@pytest.mark.asyncio
async def test_resolve_duplicate_refuses_a_replay_across_conversations():
    inbound = _FakeInbound(snapshot=_snapshot(conversation_id="chat-1"))
    runtime, _, _ = _runtime(inbound=inbound)

    with pytest.raises(FinalizedReplayCorruption):
        await runtime.resolve_duplicate(
            inbound_event_id="tg:1", conversation_id="chat-OTHER"
        )


# ============================================================
# 6. PATH 2 -- LEVEL B / CRASH RECOVERY
# ============================================================

@pytest.mark.asyncio
async def test_resolve_duplicate_recovers_a_crashed_turn_from_the_record():
    payload = _payload()
    expected = reproduce_runtime_turn(payload)
    inbound = _FakeInbound(error=FinalizedReplayUnavailable("in flight"))
    recovery = _FakeRecovery(payload=payload)
    core = _FakeCore(_fresh_outcome())
    runtime, _, events = _runtime(
        inbound=inbound, recovery=recovery, core=core
    )

    delivery = await runtime.resolve_duplicate(
        inbound_event_id="tg:1", conversation_id="chat-1"
    )

    assert isinstance(delivery, RecoveredTurnDelivery)
    assert isinstance(delivery, RuntimeDelivery)
    assert isinstance(delivery.turn, RuntimeTurn)
    assert delivery.turn == expected
    assert delivery.turn.decision == "ESCALATE"

    # NO SupportCore call, NO provider re-run, nothing written.
    assert core.calls == []
    assert recovery.reads == [("tg:1", "chat-1")]
    assert recovery.saves == []
    assert events == ["open", "close"]


@pytest.mark.asyncio
async def test_recovered_turn_uses_the_snapshotted_projection_not_live_state():
    # The record's projection snapshot (AI_ACTIVE -> ESCALATE_NEW) wins, even
    # though the caller could now be in a human-owned state.
    payload = _payload(state=AI_ACTIVE)
    inbound = _FakeInbound(error=FinalizedReplayUnavailable("in flight"))
    runtime, _, _ = _runtime(
        inbound=inbound, recovery=_FakeRecovery(payload=payload)
    )

    delivery = await runtime.resolve_duplicate(
        inbound_event_id="tg:1", conversation_id="chat-1"
    )

    assert isinstance(delivery, RecoveredTurnDelivery)
    assert delivery.turn.case_action.value == "CREATE_NEW"
    assert delivery.turn.fsm_event.value == "ESCALATE_NEW"


# ============================================================
# 7. PATH 3 -- NOTHING TO REPRODUCE (fail closed, never fabricate)
# ============================================================

@pytest.mark.asyncio
async def test_nothing_to_reproduce_is_a_typed_unavailable():
    inbound = _FakeInbound(error=FinalizedReplayUnavailable("in flight"))
    recovery = _FakeRecovery(payload=None)  # legacy caller: no record
    core = _FakeCore(_fresh_outcome())
    runtime, _, events = _runtime(
        inbound=inbound, recovery=recovery, core=core
    )

    with pytest.raises(RuntimeRecoveryUnavailable) as excinfo:
        await runtime.resolve_duplicate(
            inbound_event_id="tg:1", conversation_id="chat-1"
        )

    # It is a RuntimeError, DISTINCT from the two LEVEL A conditions.
    assert isinstance(excinfo.value, RuntimeError)
    assert not isinstance(
        excinfo.value,
        (FinalizedReplayUnavailable, FinalizedReplayCorruption),
    )
    # The legitimate "still in flight" cause is chained, not swallowed.
    assert isinstance(excinfo.value.__cause__, FinalizedReplayUnavailable)
    assert core.calls == []
    assert recovery.saves == []


# ============================================================
# 8. A DAMAGED LEVEL A RECORD IS LOUD (corrupt != absent)
# ============================================================

@pytest.mark.asyncio
async def test_damaged_snapshot_without_a_record_stays_loud():
    inbound = _FakeInbound(error=FinalizedReplayCorruption("legacy row"))
    recovery = _FakeRecovery(payload=None)
    runtime, _, _ = _runtime(inbound=inbound, recovery=recovery)

    with pytest.raises(FinalizedReplayCorruption):
        await runtime.resolve_duplicate(
            inbound_event_id="tg:1", conversation_id="chat-1"
        )


@pytest.mark.asyncio
async def test_a_valid_record_rescues_a_damaged_level_a_snapshot():
    # The "legacy finalized row" case: LEVEL A cannot serve, but the durable
    # recovery record CAN reproduce the ORIGINAL turn -- so the runtime does
    # NOT fail, it recovers.
    payload = _payload()
    inbound = _FakeInbound(error=FinalizedReplayCorruption("legacy row"))
    recovery = _FakeRecovery(payload=payload)
    runtime, _, _ = _runtime(inbound=inbound, recovery=recovery)

    delivery = await runtime.resolve_duplicate(
        inbound_event_id="tg:1", conversation_id="chat-1"
    )

    assert isinstance(delivery, RecoveredTurnDelivery)
    assert delivery.turn == reproduce_runtime_turn(payload)


@pytest.mark.asyncio
async def test_a_damaged_record_is_never_swallowed():
    inbound = _FakeInbound(error=FinalizedReplayUnavailable("in flight"))
    recovery = _FakeRecovery(error=RuntimeError("damaged record"))
    runtime, _, _ = _runtime(inbound=inbound, recovery=recovery)

    with pytest.raises(RuntimeError, match="damaged record"):
        await runtime.resolve_duplicate(
            inbound_event_id="tg:1", conversation_id="chat-1"
        )


# ============================================================
# 9. ONE BOUNDARY -- handle_delivery DISPATCHES ALL THREE PATHS
# ============================================================

@pytest.mark.asyncio
async def test_handle_delivery_dispatches_a_claim_to_the_fresh_path():
    inbound = _FakeInbound(
        claim_result=ClaimResult(claimed=True, outcome="received")
    )
    core = _FakeCore(_fresh_outcome())
    runtime, sessions, _ = _runtime(inbound=inbound, core=core)

    delivery = await runtime.handle_delivery(
        inbound_event_id="tg:1",
        channel="telegram",
        conversation_id="chat-1",
        message="black screen",
        projection_state=AI_ACTIVE,
    )

    assert isinstance(delivery, FreshTurnDelivery)
    assert inbound.claim_calls == [("tg:1", "telegram")]
    # The SAME id is forwarded to the engine (claim id == execution id).
    assert core.calls[0]["execution_id"] == "tg:1"
    assert inbound.replay_reads == []  # the fresh path never replays
    assert sessions.opened == 1  # the claim transaction only


@pytest.mark.asyncio
async def test_handle_delivery_dispatches_a_duplicate_to_durable_state():
    inbound = _FakeInbound(
        claim_result=ClaimResult(claimed=False, outcome="received"),
        snapshot=_snapshot(),
    )
    core = _FakeCore(_fresh_outcome())
    runtime, sessions, _ = _runtime(inbound=inbound, core=core)

    delivery = await runtime.handle_delivery(
        inbound_event_id="tg:1",
        channel="telegram",
        conversation_id="chat-1",
        message="black screen",
        projection_state=AI_ACTIVE,
    )

    assert isinstance(delivery, FinalizedDelivery)
    assert core.calls == []  # the engine is never consulted for a duplicate
    assert inbound.claim_calls == [("tg:1", "telegram")]
    assert sessions.opened == 2  # claim TX + the read-only resolution TX


@pytest.mark.asyncio
async def test_handle_delivery_dispatches_an_in_flight_duplicate_to_w3():
    inbound = _FakeInbound(
        claim_result=ClaimResult(claimed=False, outcome="received"),
        error=FinalizedReplayUnavailable("in flight"),
    )
    core = _FakeCore(_fresh_outcome())
    runtime, _, _ = _runtime(
        inbound=inbound, recovery=_FakeRecovery(payload=_payload()), core=core
    )

    delivery = await runtime.handle_delivery(
        inbound_event_id="tg:1",
        channel="telegram",
        conversation_id="chat-1",
        message="black screen",
        projection_state=AI_ACTIVE,
    )

    assert isinstance(delivery, RecoveredTurnDelivery)
    assert core.calls == []


# ============================================================
# 10. THE DELIVERY VOCABULARY IS CLOSED
# ============================================================

def test_delivery_values_are_frozen_and_single_payload():
    body = reproduce_finalized_response(_snapshot())
    finalized = FinalizedDelivery(body=body)
    assert finalized.body is body
    with pytest.raises(AttributeError):
        finalized.body = body  # type: ignore[misc]

    turn = reproduce_runtime_turn(_payload())
    recovered = RecoveredTurnDelivery(turn=turn)
    assert recovered.turn is turn
    with pytest.raises(AttributeError):
        recovered.turn = turn  # type: ignore[misc]

    fresh = FreshTurnDelivery(turn=turn)
    assert fresh.turn is turn
    with pytest.raises(AttributeError):
        fresh.turn = turn  # type: ignore[misc]


def test_runtime_delivery_alias_covers_all_three_branches():
    turn = reproduce_runtime_turn(_payload())
    deliveries = (
        FreshTurnDelivery(turn=turn),
        RecoveredTurnDelivery(turn=turn),
        FinalizedDelivery(body=reproduce_finalized_response(_snapshot())),
    )
    for delivery in deliveries:
        assert isinstance(delivery, RuntimeDelivery)
    assert {type(d) for d in deliveries} == {
        FreshTurnDelivery,
        RecoveredTurnDelivery,
        FinalizedDelivery,
    }


def test_a_bare_engine_replay_is_not_a_delivery_value():
    """A replay with nothing durable behind it is a FAILURE, not a delivery."""
    assert not hasattr(rs, "ReplayDelivery")
    assert not hasattr(rs, "EngineReplayDelivery")
    assert RuntimeReplay not in {
        FreshTurnDelivery,
        RecoveredTurnDelivery,
        FinalizedDelivery,
    }


# ============================================================
# 11. ARCHITECTURE: ONE SANCTIONED SupportCore OWNER, NO NEW POLICY
# ============================================================

_MODULE_PATH = Path(rs.__file__).resolve()
_REPO_ROOT = _MODULE_PATH.parents[1]

_RUNTIME_MODULES = (
    Path("yandex/chat/server.py"),
    Path("yandex/chat/alice_support_agent.py"),
    Path("yandex/realtime/web/server.py"),
)


def _source() -> str:
    return _MODULE_PATH.read_text(encoding="utf-8")


def _tree() -> ast.Module:
    return ast.parse(_source())


def _module_level_imports(tree: ast.Module) -> set[str]:
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def _code_only_text(tree: ast.Module) -> str:
    """Identifiers / attribute names / keywords / def+class names and string
    literals -- everything EXCEPT comments and docstrings."""
    docstrings: set[int] = set()
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
                docstrings.add(id(body[0].value))

    parts: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            parts.append(node.attr)
        elif isinstance(node, ast.Name):
            parts.append(node.id)
        elif isinstance(node, ast.keyword) and node.arg:
            parts.append(node.arg)
        elif isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            parts.append(node.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                parts.append(node.value)

    return " ".join(parts).lower()


def test_runtime_service_is_the_one_sanctioned_support_core_owner():
    imported = _module_level_imports(_tree())

    # EXPECTED for the full runtime boundary: the composition owner is here.
    assert "support.support_core" in imported

    # ...and ONLY the existing pieces are composed: no engine / routing /
    # projection / LLM re-implementation, no transport.
    for forbidden in (
        "support.orchestrator",
        "support.router",
        "support.lifecycle_projection",
        "support.procedure_catalog",
        "support.response_generator",
        "support.alice_analyzer",
    ):
        assert forbidden not in imported, forbidden
    assert {name.split(".")[0] for name in imported}.isdisjoint({"yandex"})

    assert {
        "support.runtime_outcome",
        "support.idempotency",
        "support.finalized_replay",
        "support.turn_recovery",
        "support.turn_recovery_store",
        "support.schemas",
        "support.domain",
    } <= imported


def test_runtime_service_re_derives_no_lifecycle_mapping():
    """``decision`` / ``reason`` / the FSM event / the case action are the
    projection owner's; they are never computed here."""
    code = _code_only_text(_tree())

    for token in (
        "fsm_event",
        "case_action",
        "supportevent",
        "caseaction",
        "routeaction",
        "escalate_reason",
        "clarify_reason",
        "user_message_ai",
        "user_message_human",
        "escalate_new",
        "escalate_resume",
        "waiting_user",
        "project_turn_outcome",
        "lifecycleprojection",
        "is_transition_allowed",
        "next_state",
    ):
        assert token not in code, token


def test_runtime_service_introduces_no_persistence_or_migration():
    tree = _tree()

    imported = _module_level_imports(tree)
    for forbidden in (
        "support.models",
        "support.db",
        "sqlalchemy.orm",
        "alembic",
    ):
        assert forbidden not in imported, forbidden

    code = _code_only_text(tree)
    for token in (
        "create table",
        "create_all",
        "alembic",
        "migration",
        "column",
        "mapped",
        "metadata",
        "save_cas",
        "add_all",
        "turnrecoveryrow",
    ):
        assert token not in code, token

    # The module defines NO persistence object: exactly the delivery values,
    # the boundary, the typed failure and the two structural seam Protocols.
    assert {node.name for node in tree.body if isinstance(node, ast.ClassDef)} == {
        "InboundEventService",
        "DurableRecoveryStore",
        "RuntimeRecoveryUnavailable",
        "FreshTurnDelivery",
        "RecoveredTurnDelivery",
        "FinalizedDelivery",
        "StatefulTurnRuntime",
    }


def test_runtime_service_public_api_surface():
    assert set(rs.__all__) == {
        "DurableRecoveryStore",
        "FinalizedDelivery",
        "FreshTurnDelivery",
        "InboundEventService",
        "RecoveredTurnDelivery",
        "RuntimeDelivery",
        "RuntimeRecoveryUnavailable",
        "StatefulTurnRuntime",
    }


def _imports_module(path: Path, module: str) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (SyntaxError, ValueError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module or alias.name.startswith(module + "."):
                    return True
        elif isinstance(node, ast.ImportFrom):
            found = node.module or ""
            if found == module or found.startswith(module + "."):
                return True
            if node.level >= 1 and found.split(".")[0] == module.split(".")[-1]:
                return True
    return False


def test_only_the_sanctioned_composition_root_imports_the_runtime_service():
    """Post-cutover: EXACTLY ONE production module wires the runtime.

    ``yandex/chat/server.py`` is the transport composition root (HL-2c.3b.3);
    it is the ONE place that delegates the turn to the runtime boundary. No
    OTHER production module -- and no runtime module -- may import it.
    """
    sanctioned = {(_REPO_ROOT / "yandex/chat/server.py").resolve()}

    offenders: set[Path] = set()
    for pattern in ("support/**/*.py", "yandex/**/*.py", "*.py"):
        for path in _REPO_ROOT.glob(pattern):
            if path.resolve() == _MODULE_PATH or "__pycache__" in path.parts:
                continue
            if "tests" in path.parts:
                continue
            if _imports_module(path, "support.runtime_service"):
                offenders.add(path.resolve())

    assert offenders == sanctioned

    # A runtime module must never import the runtime service (no cycles);
    # the transport composition root is the ONLY sanctioned importer.
    for relative in _RUNTIME_MODULES:
        if relative == Path("yandex/chat/server.py"):
            continue
        path = _REPO_ROOT / relative
        if path.exists():
            assert not _imports_module(path, "support.runtime_service"), relative


def test_server_py_is_the_composition_root_for_the_stateful_cutover():
    server = _REPO_ROOT / "yandex/chat/server.py"
    assert server.exists()
    text = server.read_text(encoding="utf-8")

    # The transport delegates EVERY turn decision to the ONE runtime boundary
    # and never imports the composition boundary (``support_core``) directly.
    assert "_turn_runtime" in text
    assert "StatefulTurnRuntime" in text
    assert "support.runtime_service" in text
    assert "support.support_core" not in text

    # The engine's optimistic-CAS conflict maps to a STABLE conflict signal
    # (409): no retry, no second engine call, no partial write.
    assert "ProcedureStateConflict" in text
    assert "procedure_state_conflict" in text
    assert "status=409" in text


def test_not_exported_from_the_support_package_init():
    init = (_REPO_ROOT / "support/__init__.py").read_text(encoding="utf-8")
    assert "runtime_service" not in init
    assert "StatefulTurnRuntime" not in init


def test_no_other_support_module_imports_support_core():
    """The sanctioned owner is EXACTLY this one module (guard cross-check)."""
    offenders: set[str] = set()
    for path in _REPO_ROOT.glob("support/**/*.py"):
        if path.resolve() == _MODULE_PATH or "__pycache__" in path.parts:
            continue
        if _imports_module(path, "support.support_core"):
            offenders.add(path.name)
    assert offenders == set()
