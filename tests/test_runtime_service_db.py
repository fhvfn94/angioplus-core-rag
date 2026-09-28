# -*- coding: utf-8 -*-
"""
HL-2c.3b.2 -- Stateful Turn Runtime (DB-BACKED proofs).

REAL PostgreSQL (no SQLite, no in-memory double): these tests prove the ONE
runtime boundary resolves EVERY path of an inbound turn from the REAL durable
state, end to end, with the PRODUCTION defaults (the real ``IdempotencyService``
and the real ``TurnRecoveryStore`` -- nothing is faked):

  * FRESH / ENGINE REPLAY -- when the turn engine reports an idempotent REPLAY,
    the boundary reproduces the ORIGINAL turn from the REAL ``turn_recovery``
    record instead of inventing one. The provider-dependent happy path is
    proven OFFLINE (``tests/test_runtime_service.py``); here the engine result
    is produced by the REAL executor (real CAS, real recovery record), so the
    recovery read is exercised against real durable state;

  * PATH 1 -- a finalized turn is replayed EXACTLY from the durable snapshot
    (``retrieval_used`` preserved, not hardcoded), including a ``forwarded``
    turn;
  * PATH 2 -- a turn whose engine part applied but which the runtime never
    finalized is RECOVERED from the ``turn_recovery`` record, reproducing the
    ORIGINAL runtime turn WITHOUT re-running anything;
  * PATH 3 -- an in-flight / legacy-caller event with no durable material fails
    CLOSED with a typed ``RuntimeRecoveryUnavailable``;
  * a DAMAGED LEVEL A row stays LOUD unless a VALID recovery record can serve;
  * a recovery record NEVER crosses conversations;
  * resolving is READ ONLY: it creates no message and moves no FSM.

Every test is marked ``db`` and therefore goes through the fail-closed
``tests/conftest.py`` guard; run with a dedicated ``TEST_DATABASE_URL``:

  postgresql+asyncpg://angioplus_test:<pw>@127.0.0.1:5432/angioplus_test

e.g. (PowerShell)::

    $env:TEST_DATABASE_URL = "<that DSN>"
    python -m pytest tests/test_runtime_service_db.py -q
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, inspect as sa_inspect, select
from sqlalchemy import text as sql_text

from support.domain import AI_ACTIVE, MessageRole
from support.finalized_replay import FinalizedReplayCorruption
from support.idempotency import IdempotencyService
from support.message_service import MessageService
from support.models import Conversation, Message, TurnRecoveryRow
from support.procedure_state import ProcedureState
from support.procedure_state_store import ProcedureStateStore
from support.router import EscalateReason, RouteAction, RoutingDecision
from support.runtime_service import (
    FinalizedDelivery,
    FreshTurnDelivery,
    RecoveredTurnDelivery,
    RuntimeRecoveryUnavailable,
    StatefulTurnRuntime,
)
from support.session_service import SessionService
from support.support_core import TURN_EXECUTED, SupportTurnOutcome
from support.turn_executor import SupportTurnExecutor
from support.turn_recovery import (
    TurnRecoveryCorruption,
    TurnRecoveryPayload,
    reproduce_runtime_turn,
)
from support.turn_recovery_store import TurnRecoveryStore


# ============================================================
# DOUBLES / HELPERS
# ============================================================

class _Turn:
    """Structural ``TurnResult`` double (only what recovery reads)."""

    def __init__(self, text, decision):
        self.assistant_text = text
        self.decision = decision
        self.state = ProcedureState()


def _escalate() -> RoutingDecision:
    return RoutingDecision(
        action=RouteAction.ESCALATE,
        escalate_reason=EscalateReason.PROCEDURE_TERMINAL,
    )


def _turn(text="I am escalating this to a human.") -> _Turn:
    return _Turn(text=text, decision=_escalate())


class _FakeCore:
    """Structural ``SupportCore`` double: a preset outcome + a call count.

    The fresh path needs a turn outcome; the provider-dependent execution is
    proven offline, so here the double supplies an outcome built from a REAL
    ``PersistedTurnResult`` (produced by the real executor against the real
    recovery table) and records how many times the engine was consulted.
    """

    def __init__(self, outcome=None) -> None:
        self.outcome = outcome
        self.calls: list = []

    async def handle_turn(self, **kwargs):
        self.calls.append(kwargs)
        return self.outcome


def _service(session_factory) -> StatefulTurnRuntime:
    # PRODUCTION defaults: the real IdempotencyService + TurnRecoveryStore.
    return StatefulTurnRuntime(session_factory=session_factory)


def _runtime(session_factory, core) -> StatefulTurnRuntime:
    """The boundary over the PRODUCTION seams with a structural core double."""
    return StatefulTurnRuntime(
        session_factory=session_factory, support_core=core
    )


async def _table_names(engine) -> set:
    """The REAL tables of the test database (persistence snapshot)."""
    async with engine.connect() as connection:
        names = await connection.run_sync(
            lambda sync_connection: sa_inspect(
                sync_connection
            ).get_table_names()
        )
    return set(names)


def _executor(session_factory) -> SupportTurnExecutor:
    return SupportTurnExecutor(
        session_factory=session_factory, store_factory=ProcedureStateStore
    )


async def _conversation(session_factory, uid="u1", cid="chat1") -> str:
    async with session_factory() as session:
        conv = await SessionService(session).create("telegram", uid, cid)
        await session.commit()
        return conv.conversation_id


async def _assistant_message(session_factory, conversation_id, text) -> int:
    async with session_factory() as session:
        message_id = await MessageService(session).append(
            conversation_id, MessageRole.ASSISTANT, text
        )
        await session.commit()
        return message_id


async def _finalize(
    session_factory,
    *,
    event_id,
    outcome,
    conversation_id,
    assistant_message_id,
    decision,
    reason,
    case_action,
    retrieval_used,
    conversation_mode,
    support_status,
    handoff_status=None,
    external_case_id=None,
) -> None:
    async with session_factory() as session:
        await IdempotencyService(session).finalize(
            event_id,
            outcome,
            conversation_id,
            assistant_message_id,
            decision=decision,
            reason=reason,
            case_action=case_action,
            retrieval_used=retrieval_used,
            conversation_mode=conversation_mode,
            support_status=support_status,
            handoff_status=handoff_status,
            external_case_id=external_case_id,
        )
        await session.commit()


async def _count_messages(session_factory, conversation_id) -> int:
    async with session_factory() as session:
        return int(
            (
                await session.execute(
                    select(func.count())
                    .select_from(Message)
                    .where(Message.conversation_id == conversation_id)
                )
            ).scalar_one()
        )


async def _conversation_triple(session_factory, conversation_id):
    async with session_factory() as session:
        row = (
            await session.execute(
                select(Conversation).where(
                    Conversation.conversation_id == conversation_id
                )
            )
        ).scalar_one()
        return (
            row.conversation_mode,
            row.support_status,
            row.handoff_status,
            row.external_case_id,
        )


async def _revision(session_factory, conversation_id):
    async with session_factory() as session:
        stored = await ProcedureStateStore(session).read(conversation_id)
        return None if stored is None else stored.revision


async def _recovery_rows(session_factory) -> list[TurnRecoveryRow]:
    async with session_factory() as session:
        return list(
            (await session.execute(select(TurnRecoveryRow))).scalars().all()
        )


async def _load_record(session_factory, *, execution_id, conversation_id):
    async with session_factory() as session:
        return await TurnRecoveryStore(session).read(
            execution_id=execution_id, conversation_id=conversation_id
        )


async def _replayed_result(session_factory, *, cid, execution_id):
    """A REAL ``PersistedTurnResult`` replay for an already-applied id.

    The REAL executor detects the idempotent hit, loads the WINNER's record
    VERBATIM from the ``turn_recovery`` table and returns it; ``process``
    deliberately fails if the engine ever tried to re-run the turn.
    """
    return await _executor(session_factory)._execute_persisted(
        conversation_id=cid,
        process=lambda previous: pytest.fail("must never re-process"),
        execution_id=execution_id,
        projection_state=AI_ACTIVE,
    )


# ============================================================
# 1. CLAIM (the service owns the short TX1)
# ============================================================

@pytest.mark.asyncio
@pytest.mark.db
async def test_db_service_claim_reports_first_then_duplicate(session_factory):
    service = _service(session_factory)

    first = await service.claim(inbound_event_id="tg:1", channel="telegram")
    assert first.claimed is True
    assert first.outcome == "received"

    second = await service.claim(inbound_event_id="tg:1", channel="telegram")
    assert second.claimed is False
    assert second.prior is None  # nothing finalized yet (in flight)
    assert second.outcome == "received"


# ============================================================
# 2. PATH 1 -- LEVEL A FINALIZED REPLAY (exact, read-only)
# ============================================================

@pytest.mark.asyncio
@pytest.mark.db
async def test_db_service_replays_a_finalized_duplicate_exactly(session_factory):
    cid = await _conversation(session_factory)
    mid = await _assistant_message(session_factory, cid, "Reseat the cable.")

    service = _service(session_factory)
    await service.claim(inbound_event_id="tg:1", channel="telegram")
    await _finalize(
        session_factory,
        event_id="tg:1",
        outcome="answered",
        conversation_id=cid,
        assistant_message_id=mid,
        decision="ANSWER",
        reason="RESOLVED_BY_L1",
        case_action="NONE",
        retrieval_used=True,
        conversation_mode="ai",
        support_status="active",
        handoff_status=None,
        external_case_id="ANGIO-1",
    )

    before_messages = await _count_messages(session_factory, cid)
    before_state = await _conversation_triple(session_factory, cid)

    delivery = await service.resolve_duplicate(
        inbound_event_id="tg:1", conversation_id=cid
    )

    assert isinstance(delivery, FinalizedDelivery)
    body = delivery.body
    assert body.answer == "Reseat the cable."
    assert body.retrieval_used is True  # the PERSISTED value
    assert body.decision == "ANSWER"
    assert body.reason == "RESOLVED_BY_L1"
    assert body.case_action == "NONE"
    assert body.conversation_id == cid
    assert body.external_case_id == "ANGIO-1"
    assert body.deduplicated is True

    # READ ONLY: no message, no FSM move.
    assert await _count_messages(session_factory, cid) == before_messages
    assert await _conversation_triple(session_factory, cid) == before_state


@pytest.mark.asyncio
@pytest.mark.db
async def test_db_service_replays_a_forwarded_finalized_duplicate(session_factory):
    cid = await _conversation(session_factory)
    service = _service(session_factory)
    await service.claim(inbound_event_id="tg:fw", channel="telegram")
    await _finalize(
        session_factory,
        event_id="tg:fw",
        outcome="forwarded",
        conversation_id=cid,
        assistant_message_id=None,
        decision="",
        reason="",
        case_action="HUMAN_FORWARD",
        retrieval_used=False,
        conversation_mode="human",
        support_status="active",
        handoff_status="waiting_operator",
        external_case_id="ANGIO-9",
    )

    delivery = await service.resolve_duplicate(
        inbound_event_id="tg:fw", conversation_id=cid
    )

    assert isinstance(delivery, FinalizedDelivery)
    assert delivery.body.answer == ""
    assert delivery.body.case_action == "HUMAN_FORWARD"
    assert delivery.body.conversation_mode == "human"
    assert delivery.body.handoff_status == "waiting_operator"
    assert delivery.body.external_case_id == "ANGIO-9"


@pytest.mark.asyncio
@pytest.mark.db
async def test_db_service_finalized_replay_ignores_live_state_drift(
    session_factory,
):
    cid = await _conversation(session_factory)
    mid = await _assistant_message(session_factory, cid, "Escalating now.")

    service = _service(session_factory)
    await service.claim(inbound_event_id="tg:drift", channel="telegram")
    await _finalize(
        session_factory,
        event_id="tg:drift",
        outcome="escalated",
        conversation_id=cid,
        assistant_message_id=mid,
        decision="ESCALATE",
        reason="STEPS_EXHAUSTED",
        case_action="CREATE_NEW",
        retrieval_used=False,
        conversation_mode="ai",
        support_status="active",
        handoff_status=None,
        external_case_id=None,
    )

    # The live conversation moves on AFTER finalization.
    async with session_factory() as session:
        await session.execute(
            sql_text(
                "UPDATE conversations SET conversation_mode = 'human', "
                "handoff_status = 'waiting_operator' WHERE conversation_id = :cid"
            ),
            {"cid": cid},
        )
        await session.commit()

    delivery = await service.resolve_duplicate(
        inbound_event_id="tg:drift", conversation_id=cid
    )

    assert isinstance(delivery, FinalizedDelivery)
    # The SNAPSHOT wins over the now-human conversation.
    assert delivery.body.conversation_mode == "ai"
    assert delivery.body.handoff_status is None
    assert delivery.body.case_action == "CREATE_NEW"


# ============================================================
# 3. PATH 2 -- LEVEL B / CRASH RECOVERY
# ============================================================

@pytest.mark.asyncio
@pytest.mark.db
async def test_db_service_recovers_a_crashed_engine_turn(session_factory):
    cid = await _conversation(session_factory)
    service = _service(session_factory)

    # The runtime claimed the event, then the engine applied the turn, then
    # the process crashed BEFORE the runtime finalized anything.
    await service.claim(inbound_event_id="tg:crash", channel="telegram")
    winner_turn = _turn(text="The winner's original answer.")
    await _executor(session_factory)._execute_persisted(
        conversation_id=cid,
        process=lambda previous: winner_turn,
        execution_id="tg:crash",
        projection_state=AI_ACTIVE,
    )

    # The inbound row is still ``received``: LEVEL A has nothing.
    duplicate = await service.claim(
        inbound_event_id="tg:crash", channel="telegram"
    )
    assert duplicate.claimed is False
    assert duplicate.prior is None

    delivery = await service.resolve_duplicate(
        inbound_event_id="tg:crash", conversation_id=cid
    )

    assert isinstance(delivery, RecoveredTurnDelivery)
    payload = await _load_record(
        session_factory, execution_id="tg:crash", conversation_id=cid
    )
    assert delivery.turn == reproduce_runtime_turn(payload)
    assert delivery.turn.text == winner_turn.assistant_text
    assert delivery.turn.decision == "ESCALATE"
    assert delivery.turn.fsm_event.value == "ESCALATE_NEW"
    assert delivery.turn.case_action.value == "CREATE_NEW"

    # RECOVERY is READ ONLY: no message was written and the engine revision
    # did not move.
    assert await _count_messages(session_factory, cid) == 0
    assert await _revision(session_factory, cid) == 1
    assert len(await _recovery_rows(session_factory)) == 1


@pytest.mark.asyncio
@pytest.mark.db
async def test_db_service_recovered_turn_uses_the_snapshotted_projection(
    session_factory,
):
    cid = await _conversation(session_factory)
    service = _service(session_factory)
    await service.claim(inbound_event_id="tg:crash2", channel="telegram")
    await _executor(session_factory)._execute_persisted(
        conversation_id=cid,
        process=lambda previous: _turn(),
        execution_id="tg:crash2",
        projection_state=AI_ACTIVE,
    )

    delivery = await service.resolve_duplicate(
        inbound_event_id="tg:crash2", conversation_id=cid
    )

    assert isinstance(delivery, RecoveredTurnDelivery)
    # AI_ACTIVE at CAS time -> ESCALATE_NEW / CREATE_NEW, regardless of any
    # later move of the live conversation.
    assert delivery.turn.fsm_event.value == "ESCALATE_NEW"
    assert delivery.turn.case_action.value == "CREATE_NEW"


# ============================================================
# 4. PATH 3 -- NOTHING TO REPRODUCE (fail closed)
# ============================================================

@pytest.mark.asyncio
@pytest.mark.db
async def test_db_service_in_flight_event_has_nothing_to_reproduce(
    session_factory,
):
    service = _service(session_factory)
    await service.claim(inbound_event_id="tg:in-flight", channel="telegram")

    with pytest.raises(RuntimeRecoveryUnavailable):
        await service.resolve_duplicate(
            inbound_event_id="tg:in-flight", conversation_id="chat-missing"
        )


@pytest.mark.asyncio
@pytest.mark.db
async def test_db_service_legacy_engine_caller_has_nothing_to_reproduce(
    session_factory,
):
    """An engine caller that passed no projection snapshot writes NO record."""
    cid = await _conversation(session_factory)
    service = _service(session_factory)
    await service.claim(inbound_event_id="tg:legacy", channel="telegram")

    # LEGACY call shape: no ``projection_state`` -> no recovery record.
    await _executor(session_factory)._execute_persisted(
        conversation_id=cid,
        process=lambda previous: _turn(),
        execution_id="tg:legacy",
    )
    assert await _recovery_rows(session_factory) == []

    with pytest.raises(RuntimeRecoveryUnavailable):
        await service.resolve_duplicate(
            inbound_event_id="tg:legacy", conversation_id=cid
        )


# ============================================================
# 5. A DAMAGED LEVEL A ROW IS LOUD (corrupt != absent)
# ============================================================

async def _make_legacy_finalized_row(session_factory, *, event_id, cid, mid):
    """Simulate a pre-migration finalized row: outcome + metadata, NULL snapshot."""
    async with session_factory() as session:
        await session.execute(
            sql_text(
                "UPDATE inbound_events SET outcome = 'answered', "
                "conversation_id = :cid, assistant_message_id = :mid, "
                "decision = 'ANSWER', reason = 'RESOLVED_BY_L1', "
                "case_action = 'NONE' "
                "WHERE inbound_event_id = :eid"
            ),
            {"cid": cid, "mid": mid, "eid": event_id},
        )
        await session.commit()


@pytest.mark.asyncio
@pytest.mark.db
async def test_db_service_damaged_level_a_row_stays_loud(session_factory):
    cid = await _conversation(session_factory)
    mid = await _assistant_message(session_factory, cid, "Legacy answer.")
    service = _service(session_factory)
    await service.claim(inbound_event_id="tg:legacy-row", channel="telegram")
    await _make_legacy_finalized_row(
        session_factory, event_id="tg:legacy-row", cid=cid, mid=mid
    )

    with pytest.raises(FinalizedReplayCorruption):
        await service.resolve_duplicate(
            inbound_event_id="tg:legacy-row", conversation_id=cid
        )


@pytest.mark.asyncio
@pytest.mark.db
async def test_db_service_a_valid_record_rescues_a_damaged_row(session_factory):
    """The 'legacy finalized row' with a durable recovery record recovers."""
    cid = await _conversation(session_factory)
    mid = await _assistant_message(session_factory, cid, "Legacy answer.")
    service = _service(session_factory)
    await service.claim(inbound_event_id="tg:legacy-rescued", channel="telegram")
    await _make_legacy_finalized_row(
        session_factory, event_id="tg:legacy-rescued", cid=cid, mid=mid
    )

    # A durable recovery record exists for the SAME execution id.
    payload = TurnRecoveryPayload.capture(
        execution_id="tg:legacy-rescued",
        conversation_id=cid,
        turn=_turn(),
        projection_state=AI_ACTIVE,
    )
    async with session_factory() as session:
        await TurnRecoveryStore(session).save(payload)
        await session.commit()

    delivery = await service.resolve_duplicate(
        inbound_event_id="tg:legacy-rescued", conversation_id=cid
    )

    assert isinstance(delivery, RecoveredTurnDelivery)
    assert delivery.turn == reproduce_runtime_turn(payload)
    assert delivery.turn.decision == "ESCALATE"


# ============================================================
# 6. A RECORD NEVER CROSSES CONVERSATIONS
# ============================================================

@pytest.mark.asyncio
@pytest.mark.db
async def test_db_service_recovery_never_crosses_conversations(session_factory):
    cid_a = await _conversation(session_factory, uid="a", cid="chat-a")
    cid_b = await _conversation(session_factory, uid="b", cid="chat-b")
    service = _service(session_factory)
    await service.claim(inbound_event_id="tg:x", channel="telegram")

    payload = TurnRecoveryPayload.capture(
        execution_id="tg:x",
        conversation_id=cid_a,
        turn=_turn(),
        projection_state=AI_ACTIVE,
    )
    async with session_factory() as session:
        await TurnRecoveryStore(session).save(payload)
        await session.commit()

    # The OWNER conversation is served the record...
    own = await service.resolve_duplicate(
        inbound_event_id="tg:x", conversation_id=cid_a
    )
    assert isinstance(own, RecoveredTurnDelivery)

    # ...but another conversation asking for it must NEVER be served.
    with pytest.raises(TurnRecoveryCorruption):
        await service.resolve_duplicate(
            inbound_event_id="tg:x", conversation_id=cid_b
        )


# ============================================================
# 7. THE WHOLE RESOLUTION IS READ ONLY
# ============================================================

@pytest.mark.asyncio
@pytest.mark.db
async def test_db_service_resolution_writes_nothing(session_factory):
    cid = await _conversation(session_factory)
    service = _service(session_factory)
    await service.claim(inbound_event_id="tg:ro", channel="telegram")
    await _executor(session_factory)._execute_persisted(
        conversation_id=cid,
        process=lambda previous: _turn(),
        execution_id="tg:ro",
        projection_state=AI_ACTIVE,
    )

    before = {
        "messages": await _count_messages(session_factory, cid),
        "triple": await _conversation_triple(session_factory, cid),
        "revision": await _revision(session_factory, cid),
        "rows": len(await _recovery_rows(session_factory)),
    }

    await service.resolve_duplicate(inbound_event_id="tg:ro", conversation_id=cid)

    after = {
        "messages": await _count_messages(session_factory, cid),
        "triple": await _conversation_triple(session_factory, cid),
        "revision": await _revision(session_factory, cid),
        "rows": len(await _recovery_rows(session_factory)),
    }
    assert after == before


# ============================================================
# 8. THE FRESH PATH + THE ENGINE REPLAY (real durable record)
# ============================================================

@pytest.mark.asyncio
@pytest.mark.db
async def test_db_fresh_path_engine_replay_is_served_from_the_real_record(
    session_factory,
):
    cid = await _conversation(session_factory)

    # The engine applied the turn under this execution id, then the process
    # crashed BEFORE the runtime finalized anything.
    await _executor(session_factory)._execute_persisted(
        conversation_id=cid,
        process=lambda previous: _turn(text="The winner's original answer."),
        execution_id="tg:fresh",
        projection_state=AI_ACTIVE,
    )
    assert len(await _recovery_rows(session_factory)) == 1

    # The REAL engine result for the SAME id: an idempotent replay whose
    # recovery payload is read VERBATIM from the real ``turn_recovery`` row.
    replay = await _replayed_result(
        session_factory, cid=cid, execution_id="tg:fresh"
    )
    assert replay.replayed is True
    assert replay.turn is None
    assert replay.recovery is not None

    core = _FakeCore(
        SupportTurnOutcome(
            status=TURN_EXECUTED, conversation_id=cid, persisted=replay
        )
    )
    runtime = _runtime(session_factory, core)

    before_revision = await _revision(session_factory, cid)
    delivery = await runtime.execute_fresh_turn(
        conversation_id=cid,
        message="black screen",
        projection_state=AI_ACTIVE,
        execution_id="tg:fresh",
    )

    # A replay is NOT a fresh turn: the ORIGINAL turn is reproduced.
    assert isinstance(delivery, RecoveredTurnDelivery)
    assert not isinstance(delivery, FreshTurnDelivery)
    assert delivery.turn.text == "The winner's original answer."
    assert delivery.turn.decision == "ESCALATE"
    assert delivery.turn.fsm_event.value == "ESCALATE_NEW"
    assert len(core.calls) == 1  # the engine was consulted exactly once
    assert core.calls[0]["execution_id"] == "tg:fresh"

    # NOTHING was re-run and NOTHING was written.
    assert await _revision(session_factory, cid) == before_revision
    assert await _count_messages(session_factory, cid) == 0
    assert len(await _recovery_rows(session_factory)) == 1


@pytest.mark.asyncio
@pytest.mark.db
async def test_db_runtime_handle_delivery_recovers_with_no_second_engine_call(
    session_factory,
):
    cid = await _conversation(session_factory)
    await _executor(session_factory)._execute_persisted(
        conversation_id=cid,
        process=lambda previous: _turn(text="Recovered answer."),
        execution_id="tg:hd",
        projection_state=AI_ACTIVE,
    )
    replay = await _replayed_result(
        session_factory, cid=cid, execution_id="tg:hd"
    )
    core = _FakeCore(
        SupportTurnOutcome(
            status=TURN_EXECUTED, conversation_id=cid, persisted=replay
        )
    )
    runtime = _runtime(session_factory, core)

    # 1st delivery: the claim is WON -> the fresh path (engine replay).
    first = await runtime.handle_delivery(
        inbound_event_id="tg:hd",
        channel="telegram",
        conversation_id=cid,
        message="black screen",
        projection_state=AI_ACTIVE,
    )
    assert isinstance(first, RecoveredTurnDelivery)
    assert first.turn.text == "Recovered answer."
    assert len(core.calls) == 1

    # 2nd delivery: the claim is LOST -> the durable W3 recovery, with the
    # engine consulted ZERO additional times.
    second = await runtime.handle_delivery(
        inbound_event_id="tg:hd",
        channel="telegram",
        conversation_id=cid,
        message="black screen",
        projection_state=AI_ACTIVE,
    )
    assert isinstance(second, RecoveredTurnDelivery)
    assert second.turn == first.turn
    assert len(core.calls) == 1  # no second SupportCore call, ever
    assert len(await _recovery_rows(session_factory)) == 1
    assert await _count_messages(session_factory, cid) == 0


@pytest.mark.asyncio
@pytest.mark.db
async def test_db_runtime_introduces_no_new_persistence(
    session_factory, engine
):
    cid = await _conversation(session_factory)
    before = await _table_names(engine)

    await _executor(session_factory)._execute_persisted(
        conversation_id=cid,
        process=lambda previous: _turn(text="Recovered answer."),
        execution_id="tg:tables",
        projection_state=AI_ACTIVE,
    )
    replay = await _replayed_result(
        session_factory, cid=cid, execution_id="tg:tables"
    )
    core = _FakeCore(
        SupportTurnOutcome(
            status=TURN_EXECUTED, conversation_id=cid, persisted=replay
        )
    )
    runtime = _runtime(session_factory, core)

    # Every path of the boundary: claim + fresh(replay) + duplicate resolve.
    await runtime.handle_delivery(
        inbound_event_id="tg:tables",
        channel="telegram",
        conversation_id=cid,
        message="black screen",
        projection_state=AI_ACTIVE,
    )
    await runtime.resolve_duplicate(
        inbound_event_id="tg:tables", conversation_id=cid
    )

    # The schema is EXACTLY what it was: the boundary persists nothing new.
    assert await _table_names(engine) == before
    assert await _revision(session_factory, cid) == 1
