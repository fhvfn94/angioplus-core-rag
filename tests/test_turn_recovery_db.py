# -*- coding: utf-8 -*-
"""
HL-2c.3a -- durable turn recovery (DB-BACKED proofs).

REAL PostgreSQL (no SQLite, no in-memory double): the ONE thing these tests
must prove is TRANSACTIONAL behaviour, which an offline double cannot honestly
demonstrate.

  * the recovery record is written on the SAME session / transaction as the
    ProcedureState CAS -- and if that write FAILS, the CAS is rolled back with
    it (no crash window in which the procedure state advanced without its
    recovery record);
  * an idempotent replay loads the WINNER's STORED record VERBATIM (never one
    recomputed on the hit, never overwritten by a racing loser) and reproduces
    the ORIGINAL runtime turn -- including the SNAPSHOTTED projection inputs,
    which beat whatever state the replaying caller passes in;
  * a legacy caller that supplies no projection snapshot writes NO record, and
    its replay legitimately reports ``recovery is None`` (ABSENT != CORRUPT);
  * a damaged row fails LOUDLY; a record never crosses conversations and one
    ``execution_id`` can never be reused for another conversation.

Every test is marked ``db`` and therefore goes through the fail-closed
``tests/conftest.py`` guard; run with a dedicated ``TEST_DATABASE_URL``
(async DSN, NOT the production database):

  postgresql+asyncpg://angioplus_test:<pw>@127.0.0.1:5432/angioplus_test

e.g. (PowerShell)::

    $env:TEST_DATABASE_URL = "<that DSN>"
    python -m pytest tests/test_turn_recovery_db.py -q
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import select
from sqlalchemy import text as sql_text
from sqlalchemy.exc import IntegrityError

from support.domain import (
    ConversationMode,
    HandoffStatus,
    SupportStatus,
)
from support.models import TurnRecoveryRow
from support.procedure_state import ProcedureState
from support.procedure_state_store import (
    PersistedProcedureState,
    ProcedureStateStore,
)
from support.router import EscalateReason, RouteAction, RoutingDecision
from support.runtime_outcome import RuntimeTurn, adapt_turn_outcome
from support.session_service import SessionService
from support.turn_executor import PersistedTurnResult, SupportTurnExecutor
from support.turn_recovery import (
    TurnRecoveryCorruption,
    TurnRecoveryPayload,
    reproduce_runtime_turn,
)
from support.turn_recovery_store import TurnRecoveryStore

AI_ACTIVE = (ConversationMode.AI, SupportStatus.ACTIVE, None)
HUMAN_WAITING_OPERATOR = (
    ConversationMode.HUMAN,
    SupportStatus.ACTIVE,
    HandoffStatus.WAITING_OPERATOR,
)


# ============================================================
# HELPERS / DOUBLES
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


def _payload(*, execution_id, conversation_id, turn, state=AI_ACTIVE):
    return TurnRecoveryPayload.capture(
        execution_id=execution_id,
        conversation_id=conversation_id,
        turn=turn,
        projection_state=state,
    )


def _original(turn, state, conversation_id) -> RuntimeTurn:
    """The projection of the ORIGINAL (non-replay) turn -- the reference."""
    outcome = SimpleNamespace(
        status="TURN_EXECUTED",
        conversation_id=conversation_id,
        persisted=PersistedTurnResult(
            conversation_id=conversation_id,
            turn=turn,
            state=ProcedureState(),
            revision=1,
        ),
    )
    result = adapt_turn_outcome(outcome, state)
    assert isinstance(result, RuntimeTurn)
    return result


def _must_not_process(previous_state):
    raise AssertionError("PROCESS must not run on an idempotent replay")


class _StaleReadStore(ProcedureStateStore):
    """A REAL store whose READ hides the winner's ``execution_id``.

    Simulates the losing worker of a same-execution-id race: its READ still
    reports "no write under this id" (so it is NOT short-circuited before
    PROCESS), while ``save_cas`` runs for REAL against PostgreSQL and is
    therefore resolved by the REAL idempotency check -- the post-CAS race
    path.
    """

    async def read(self, conversation_id):
        stored = await super().read(conversation_id)
        if stored is None:
            return None
        return PersistedProcedureState(
            state=stored.state,
            revision=stored.revision,
            last_execution_id=None,
        )


class _ExplodingRecoveryStore:
    """Recovery store whose WRITE fails (to prove CAS atomicity)."""

    def __init__(self, session):
        self._session = session

    async def save(self, payload):
        raise RuntimeError("recovery write failed")

    async def read(self, *, execution_id, conversation_id):
        raise AssertionError("no recovery read expected in this test")


async def _make_conversation(session_factory, channel="telegram", uid="u1",
                            cid="chat1") -> str:
    async with session_factory() as session:
        conv = await SessionService(session).create(channel, uid, cid)
        await session.commit()
        return conv.conversation_id


def _exploding_store(session):
    return _ExplodingRecoveryStore(session)


async def _read_rows(session_factory) -> list[TurnRecoveryRow]:
    async with session_factory() as session:
        return list(
            (await session.execute(select(TurnRecoveryRow))).scalars().all()
        )


# ============================================================
# 1. WRITTEN ATOMICALLY WITH THE CAS
# ============================================================

@pytest.mark.asyncio
@pytest.mark.db
async def test_db_recovery_is_written_with_the_cas(session_factory):
    cid = await _make_conversation(session_factory)
    executor = SupportTurnExecutor(
        session_factory=session_factory, store_factory=ProcedureStateStore
    )
    turn = _turn()

    result = await executor._execute_persisted(
        conversation_id=cid,
        process=lambda previous: turn,
        execution_id="tg:1",
        projection_state=AI_ACTIVE,
    )

    # A NORMAL turn carries its own turn, NOT a recovery record.
    assert result.replayed is False
    assert result.recovery is None
    assert result.revision == 1

    rows = await _read_rows(session_factory)
    assert len(rows) == 1
    row = rows[0]
    assert row.execution_id == "tg:1"
    assert row.conversation_id == cid
    assert row.text == turn.assistant_text
    assert row.action == RouteAction.ESCALATE.value
    assert row.escalate_reason == EscalateReason.PROCEDURE_TERMINAL.value
    assert row.clarify_reason is None
    assert row.projection_mode == "ai"
    assert row.projection_support_status == "active"
    assert row.projection_handoff_status is None
    assert row.created_at is not None

    async with session_factory() as session:
        stored = await ProcedureStateStore(session).read(cid)
    assert stored.revision == 1
    assert stored.last_execution_id == "tg:1"


@pytest.mark.asyncio
@pytest.mark.db
async def test_db_no_projection_snapshot_writes_no_record(session_factory):
    """A LEGACY caller (id but no projection) must be completely unaffected."""
    cid = await _make_conversation(session_factory)
    executor = SupportTurnExecutor(
        session_factory=session_factory, store_factory=ProcedureStateStore
    )

    first = await executor._execute_persisted(
        conversation_id=cid,
        process=lambda previous: _turn(),
        execution_id="tg:legacy",
    )
    assert first.revision == 1
    assert await _read_rows(session_factory) == []

    replay = await executor._execute_persisted(
        conversation_id=cid,
        process=_must_not_process,
        execution_id="tg:legacy",
    )
    assert replay.replayed is True
    # ABSENT is legitimate (not corruption): nothing to recover.
    assert replay.recovery is None
    assert replay.revision == 1


@pytest.mark.asyncio
@pytest.mark.db
async def test_db_a_failed_recovery_write_rolls_back_the_cas(session_factory):
    """THE atomicity proof: the CAS cannot survive a failed recovery write.

    If the record were written in a SECOND transaction the procedure state
    would stay advanced here -- the exact W3 crash window this slice closes.
    """
    cid = await _make_conversation(session_factory)
    executor = SupportTurnExecutor(
        session_factory=session_factory,
        store_factory=ProcedureStateStore,
        recovery_store_factory=_exploding_store,
    )

    with pytest.raises(RuntimeError, match="recovery write failed"):
        await executor._execute_persisted(
            conversation_id=cid,
            process=lambda previous: _turn(),
            execution_id="tg:1",
            projection_state=AI_ACTIVE,
        )

    async with session_factory() as session:
        # NOTHING advanced: the CAS rolled back WITH the recovery record.
        assert await ProcedureStateStore(session).read(cid) is None
    assert await _read_rows(session_factory) == []


# ============================================================
# 2. REPLAY SERVES THE WINNER'S STORED RECORD
# ============================================================

@pytest.mark.asyncio
@pytest.mark.db
async def test_db_replay_reproduces_the_winner_turn_from_the_record(
    session_factory,
):
    cid = await _make_conversation(session_factory)
    executor = SupportTurnExecutor(
        session_factory=session_factory, store_factory=ProcedureStateStore
    )
    winner_turn = _turn(text="The winner's original answer.")

    first = await executor._execute_persisted(
        conversation_id=cid,
        process=lambda previous: winner_turn,
        execution_id="tg:dup",
        projection_state=AI_ACTIVE,
    )
    assert first.replayed is False

    # The duplicate delivery: PROCESS must NOT run, and the caller's OWN
    # (now human-owned) live state must NOT influence the recovery output.
    replay = await executor._execute_persisted(
        conversation_id=cid,
        process=_must_not_process,
        execution_id="tg:dup",
        projection_state=HUMAN_WAITING_OPERATOR,
    )

    assert replay.replayed is True
    assert replay.turn is None
    assert replay.revision == first.revision
    assert replay.recovery is not None
    assert replay.recovery.text == winner_turn.assistant_text
    # The WINNER's snapshot, not the replayer's.
    assert replay.recovery.projection_mode == "ai"
    assert replay.recovery.conversation_state() == AI_ACTIVE

    # ...and the ORIGINAL turn is reproduced exactly, from the record alone.
    reproduced = reproduce_runtime_turn(replay.recovery)
    assert reproduced == _original(winner_turn, AI_ACTIVE, cid)
    assert reproduced.text == winner_turn.assistant_text
    assert reproduced.decision == "ESCALATE"
    assert reproduced.reason == EscalateReason.PROCEDURE_TERMINAL.value
    # `AI_ACTIVE` yields ESCALATE_NEW; the replayer's HUMAN state would have
    # yielded USER_MESSAGE_HUMAN -- the snapshot wins.
    assert reproduced.fsm_event.value == "ESCALATE_NEW"
    assert reproduced.case_action.value == "CREATE_NEW"

    # Nothing advanced and the single record is untouched.
    async with session_factory() as session:
        stored = await ProcedureStateStore(session).read(cid)
    assert stored.revision == 1
    rows = await _read_rows(session_factory)
    assert len(rows) == 1
    assert rows[0].text == winner_turn.assistant_text


@pytest.mark.asyncio
@pytest.mark.db
async def test_db_post_cas_race_keeps_the_winner_record(session_factory):
    """A loser of a SAME-id race discards its turn and reads the winner's."""
    cid = await _make_conversation(session_factory)
    winner_turn = _turn(text="The winner's original answer.")

    winner = SupportTurnExecutor(
        session_factory=session_factory, store_factory=ProcedureStateStore
    )
    await winner._execute_persisted(
        conversation_id=cid,
        process=lambda previous: winner_turn,
        execution_id="tg:race",
        projection_state=AI_ACTIVE,
    )

    loser_turn = _turn(text="The LOSER's recomputed answer.")
    loser = SupportTurnExecutor(
        session_factory=session_factory, store_factory=_StaleReadStore
    )
    result = await loser._execute_persisted(
        conversation_id=cid,
        process=lambda previous: loser_turn,
        execution_id="tg:race",
        projection_state=AI_ACTIVE,
    )

    # The loser's recomputed turn is DISCARDED in favour of the winner's.
    assert result.replayed is True
    assert result.turn is None
    assert result.revision == 1
    assert result.recovery is not None
    assert result.recovery.text == winner_turn.assistant_text
    assert result.recovery.text != loser_turn.assistant_text

    rows = await _read_rows(session_factory)
    assert len(rows) == 1
    assert rows[0].text == winner_turn.assistant_text


# ============================================================
# 3. CORRUPT / MIS-SCOPED / REUSED RECORDS
# ============================================================

@pytest.mark.asyncio
@pytest.mark.db
async def test_db_a_damaged_record_fails_closed(session_factory):
    cid = await _make_conversation(session_factory)
    executor = SupportTurnExecutor(
        session_factory=session_factory, store_factory=ProcedureStateStore
    )
    await executor._execute_persisted(
        conversation_id=cid,
        process=lambda previous: _turn(),
        execution_id="tg:1",
        projection_state=AI_ACTIVE,
    )

    async with session_factory() as session:
        await session.execute(
            sql_text(
                "UPDATE turn_recovery SET action = 'NOT_AN_ACTION' "
                "WHERE execution_id = 'tg:1'"
            )
        )
        await session.commit()

    # A record that EXISTS but cannot be interpreted is NEVER degraded to
    # ``None`` (that would silently hide durable damage).
    with pytest.raises(TurnRecoveryCorruption):
        await executor._execute_persisted(
            conversation_id=cid,
            process=_must_not_process,
            execution_id="tg:1",
            projection_state=AI_ACTIVE,
        )


@pytest.mark.asyncio
@pytest.mark.db
async def test_db_a_record_never_crosses_conversations(session_factory):
    cid_a = await _make_conversation(session_factory, uid="a", cid="chat-a")
    cid_b = await _make_conversation(session_factory, uid="b", cid="chat-b")
    turn = _turn()

    async with session_factory() as session:
        await TurnRecoveryStore(session).save(
            _payload(execution_id="evt-1", conversation_id=cid_a, turn=turn)
        )
        await session.commit()

    async with session_factory() as session:
        store = TurnRecoveryStore(session)
        # The OWNER conversation reads it back verbatim.
        assert (
            await store.read(execution_id="evt-1", conversation_id=cid_a)
        ).text == turn.assistant_text
        # Another conversation must NOT be served this turn.
        with pytest.raises(TurnRecoveryCorruption):
            await store.read(execution_id="evt-1", conversation_id=cid_b)
        # A genuinely ABSENT id is ``None``, never corruption.
        assert (
            await store.read(execution_id="evt-absent", conversation_id=cid_b)
        ) is None

        # One execution id may key AT MOST ONE conversation's record: reusing
        # it cannot silently overwrite the winner's record.
        with pytest.raises(IntegrityError):
            await TurnRecoveryStore(session).save(
                _payload(
                    execution_id="evt-1", conversation_id=cid_b, turn=turn
                )
            )
            await session.commit()
