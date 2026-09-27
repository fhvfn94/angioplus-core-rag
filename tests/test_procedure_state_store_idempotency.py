"""
HL-2b -- idempotent compare-and-swap in ``ProcedureStateStore.save_cas``.

OFFLINE (no database)
    A scripted session double records every statement the store issues and
    returns canned scalar results, so the store's DECISION LOGIC is proven
    without PostgreSQL:

      * a fresh ``execution_id`` is written atomically with the state (the
        INSERT / UPDATE carries it) and replaces the attempt log;
      * repeating the SAME ``execution_id`` is an idempotent no-op: it
        returns ``applied=False`` with the CURRENT persisted state (loaded
        from the DB) and the attempt log is NOT touched;
      * a DIFFERENT ``execution_id`` with a stale revision still raises
        ``ProcedureStateConflict``;
      * omitting ``execution_id`` keeps the legacy behaviour exactly.

DB-BACKED (``@pytest.mark.db``; fail-closed guard in conftest)
    The REAL store against PostgreSQL proves the invariants end to end:
    no double advance, no attempt-log rewrite, conflict for a different id,
    and global uniqueness of ``last_execution_id``.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql.dml import Delete, Insert, Update
from sqlalchemy.sql.selectable import Select

from support.models import ProcedureStateAttemptRow, ProcedureStateRow
from support.observation import ProblemStatus, StepOutcome
from support.procedure_state import ProcedureState, StepAttempt
from support.procedure_state_store import (
    COLD_START_REVISION,
    CasResult,
    ProcedureStateConflict,
    ProcedureStateStore,
)
from support.session_service import SessionService


# ============================================================
# HELPERS
# ============================================================

def _state_a() -> ProcedureState:
    return ProcedureState(
        issue="black screen",
        procedure_id="BLACK_SCREEN",
        procedure_version=1,
        last_recommended_step="CHECK_POWER",
        step_attempts=(StepAttempt("CHECK_POWER", StepOutcome.FAILED, 0),),
        problem_status=ProblemStatus.ONGOING,
        schema_version=1,
    )


def _state_b() -> ProcedureState:
    return ProcedureState(
        issue="black screen",
        procedure_id="BLACK_SCREEN",
        procedure_version=1,
        last_recommended_step="CHECK_CABLE",
        step_attempts=(
            StepAttempt("CHECK_POWER", StepOutcome.FAILED, 0),
            StepAttempt("CHECK_CABLE", StepOutcome.FAILED, 1),
        ),
        problem_status=ProblemStatus.ONGOING,
        schema_version=1,
    )


def _row_from(
    state: ProcedureState, revision: int, execution_id=None
) -> ProcedureStateRow:
    """A detached row image carrying exactly ``state`` (never session-added)."""
    return ProcedureStateRow(
        conversation_id="chat-1",
        issue=state.issue,
        procedure_id=state.procedure_id,
        procedure_version=state.procedure_version,
        last_recommended_step=state.last_recommended_step,
        problem_status=state.problem_status.value,
        pending_clarification=state.pending_clarification,
        clarification_topic=state.clarification_topic,
        schema_version=state.schema_version,
        revision=revision,
        last_execution_id=execution_id,
    )


def _attempt_row(step_id: str, outcome: StepOutcome, sequence: int):
    """A detached attempt-log row image (never session-added)."""
    return ProcedureStateAttemptRow(
        conversation_id="chat-1",
        sequence=sequence,
        step_id=step_id,
        outcome=outcome.value,
    )


def _attempts_of(state: ProcedureState):
    return [
        _attempt_row(a.step_id, a.outcome, a.sequence)
        for a in state.step_attempts
    ]


class _FakeResult:
    """Duck-typed SQLAlchemy Result covering both read shapes."""

    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value

    def scalars(self):
        return self

    def all(self):
        return list(self._value or [])


class _ScriptedSession:
    """Session double: dispatches on statement type, records the sequence.

    ``insert`` / ``update`` are the scripted scalar results for those DML
    statements. ``rows`` is a FIFO of ``ProcedureStateRow`` images returned by
    state-row SELECTs; ``attempts`` is a FIFO of attempt-row LISTS returned by
    attempt-log SELECTs (the store reloads the PERSISTED state when it
    resolves an idempotent hit).
    """

    def __init__(self, *, insert=None, update=None, rows=(), attempts=()):
        self._insert = insert
        self._update = update
        self._rows = list(rows)
        self._attempts = list(attempts)
        self.statements: list[str] = []
        self.added: list = []
        self.flushes = 0

    async def execute(self, statement):
        if isinstance(statement, Insert):
            self.statements.append("insert")
            return _FakeResult(self._insert)
        if isinstance(statement, Update):
            self.statements.append("update")
            return _FakeResult(self._update)
        if isinstance(statement, Delete):
            self.statements.append("delete")
            return _FakeResult(None)
        if isinstance(statement, Select):
            self.statements.append("select")
            if (
                ProcedureStateAttemptRow.__table__
                in set(statement.get_final_froms())
            ):
                value = self._attempts.pop(0) if self._attempts else []
            else:
                value = self._rows.pop(0) if self._rows else None
            return _FakeResult(value)
        raise AssertionError(f"unexpected statement: {statement!r}")

    def add_all(self, rows):
        self.statements.append("add_all")
        self.added.extend(rows)

    async def flush(self):
        self.flushes += 1


# ============================================================
# OFFLINE: cold start
# ============================================================

@pytest.mark.asyncio
async def test_cold_start_applies_and_writes_execution_id():
    session = _ScriptedSession(insert=COLD_START_REVISION)

    result = await ProcedureStateStore(session).save_cas(
        "chat-1", _state_a(), expected_revision=None, execution_id="tg:1"
    )

    assert result.applied is True
    assert result.revision == COLD_START_REVISION
    assert result.state == _state_a()
    # Applied -> the attempt log WAS replaced.
    assert "insert" in session.statements
    assert "delete" in session.statements
    assert "add_all" in session.statements


@pytest.mark.asyncio
async def test_cold_start_repeat_same_execution_id_is_a_noop():
    session = _ScriptedSession(
        insert=None,
        rows=[_row_from(_state_a(), revision=1, execution_id="tg:1")],
        attempts=[_attempts_of(_state_a())],
    )

    result = await ProcedureStateStore(session).save_cas(
        "chat-1", _state_b(), expected_revision=None, execution_id="tg:1"
    )

    assert result.applied is False
    assert result.revision == 1               # current revision, not advanced
    assert result.state == _state_a()          # PERSISTED state, not candidate
    assert result.state != _state_b()
    assert session.statements == ["insert", "select", "select"]
    # Invariant: the attempt log is NEVER rewritten on an idempotent hit.
    assert "delete" not in session.statements
    assert "add_all" not in session.statements


@pytest.mark.asyncio
async def test_cold_start_conflict_different_execution_id_raises():
    session = _ScriptedSession(
        insert=None,
        rows=[_row_from(_state_a(), revision=1, execution_id="tg:other")],
    )

    with pytest.raises(ProcedureStateConflict):
        await ProcedureStateStore(session).save_cas(
            "chat-1", _state_b(), expected_revision=None, execution_id="tg:1"
        )

    assert "delete" not in session.statements
    assert "add_all" not in session.statements


# ============================================================
# OFFLINE: CAS update
# ============================================================

@pytest.mark.asyncio
async def test_update_applies_and_writes_execution_id():
    # Pre-check SELECT sees a DIFFERENT id -> not a replay -> real CAS write.
    session = _ScriptedSession(
        rows=[_row_from(_state_a(), revision=1, execution_id="tg:old")],
        update=2,
    )

    result = await ProcedureStateStore(session).save_cas(
        "chat-1", _state_b(), expected_revision=1, execution_id="tg:new"
    )

    assert result.applied is True
    assert result.revision == 2
    assert result.state == _state_b()
    assert session.statements == ["select", "update", "delete", "add_all"]


@pytest.mark.asyncio
async def test_update_repeat_same_execution_id_is_a_noop_before_cas():
    # The executor re-read the (already advanced) revision, so ``expected``
    # matches -- the pre-check must stop the second bump.
    session = _ScriptedSession(
        rows=[_row_from(_state_a(), revision=3, execution_id="tg:1")],
        attempts=[_attempts_of(_state_a())],
    )

    result = await ProcedureStateStore(session).save_cas(
        "chat-1", _state_b(), expected_revision=3, execution_id="tg:1"
    )

    assert result.applied is False
    assert result.revision == 3
    assert result.state == _state_a()          # PERSISTED state, not candidate
    assert session.statements == ["select", "select"]  # no UPDATE, no DELETE
    assert "update" not in session.statements
    assert "delete" not in session.statements


@pytest.mark.asyncio
async def test_update_stale_different_execution_id_raises():
    session = _ScriptedSession(
        rows=[
            _row_from(_state_a(), 2, "tg:winner"),  # pre-check: miss
            _row_from(_state_a(), 2, "tg:winner"),  # diagnostic: different
        ],
        update=None,  # CAS misses (stale revision)
    )

    with pytest.raises(ProcedureStateConflict):
        await ProcedureStateStore(session).save_cas(
            "chat-1", _state_a(), expected_revision=1, execution_id="tg:loser"
        )

    assert session.statements == ["select", "update", "select"]
    assert "delete" not in session.statements


@pytest.mark.asyncio
async def test_update_stale_same_execution_id_is_idempotent():
    # The SAME id won a race between our read and our CAS: the pre-check
    # missed, the UPDATE missed, but the diagnostic finds our own id.
    session = _ScriptedSession(
        rows=[
            _row_from(_state_a(), 2, "tg:other"),  # pre-check: miss
            _row_from(_state_a(), 2, "tg:1"),      # diagnostic: match
        ],
        attempts=[_attempts_of(_state_a())],
        update=None,
    )

    result = await ProcedureStateStore(session).save_cas(
        "chat-1", _state_b(), expected_revision=1, execution_id="tg:1"
    )

    assert result.applied is False
    assert result.revision == 2
    assert result.state == _state_a()   # the WINNER's persisted state
    assert result.state != _state_b()
    assert "delete" not in session.statements
    assert "add_all" not in session.statements


# ============================================================
# OFFLINE: backward compatibility (no execution_id)
# ============================================================

@pytest.mark.asyncio
async def test_without_execution_id_applies_legacy_shape():
    session = _ScriptedSession(update=2)

    result = await ProcedureStateStore(session).save_cas(
        "chat-1", _state_b(), expected_revision=1
    )

    assert result.applied is True
    assert result.revision == 2
    assert result.state == _state_b()
    # No pre-check SELECT is issued when no execution id is supplied.
    assert session.statements == ["update", "delete", "add_all"]


@pytest.mark.asyncio
async def test_without_execution_id_stale_revision_still_conflicts():
    session = _ScriptedSession(
        rows=[_row_from(_state_a(), 5, "tg:1")], update=None
    )

    with pytest.raises(ProcedureStateConflict):
        await ProcedureStateStore(session).save_cas(
            "chat-1", _state_a(), expected_revision=1
        )


# ============================================================
# DB-BACKED
# ============================================================

async def _make_conversation(
    session_factory, channel="telegram", uid="u1", cid="chat1"
) -> str:
    async with session_factory() as session:
        conv = await SessionService(session).create(channel, uid, cid)
        await session.commit()
        return conv.conversation_id


@pytest.mark.asyncio
@pytest.mark.db
async def test_db_same_execution_id_never_advances_twice(session_factory):
    cid = await _make_conversation(session_factory)

    async with session_factory() as session:
        first = await ProcedureStateStore(session).save_cas(
            cid, _state_a(), expected_revision=None, execution_id="tg:1"
        )
        await session.commit()
    assert first.applied is True
    assert first.revision == 1

    # The SAME execution id is redelivered at the SAME (re-read) revision:
    # it must be an idempotent no-op that hands back the PERSISTED state.
    async with session_factory() as session:
        second = await ProcedureStateStore(session).save_cas(
            cid, _state_b(), expected_revision=1, execution_id="tg:1"
        )
        await session.commit()
    assert second.applied is False
    assert second.revision == 1
    assert second.state == _state_a()   # PERSISTED, not the _state_b candidate

    async with session_factory() as check:
        stored = await ProcedureStateStore(check).read(cid)
        attempts = (
            await check.execute(
                select(ProcedureStateAttemptRow)
                .where(ProcedureStateAttemptRow.conversation_id == cid)
                .order_by(ProcedureStateAttemptRow.sequence)
            )
        ).scalars().all()

    assert stored.revision == 1
    assert stored.state == _state_a()            # winner's state intact
    assert stored.last_execution_id == "tg:1"
    assert [(a.sequence, a.step_id) for a in attempts] == [(0, "CHECK_POWER")]


@pytest.mark.asyncio
@pytest.mark.db
async def test_db_different_execution_id_stale_revision_conflicts(session_factory):
    cid = await _make_conversation(session_factory)

    async with session_factory() as session:
        await ProcedureStateStore(session).save_cas(
            cid, _state_a(), expected_revision=None, execution_id="tg:1"
        )
        await session.commit()
    async with session_factory() as session:
        applied = await ProcedureStateStore(session).save_cas(
            cid, _state_b(), expected_revision=1, execution_id="tg:2"
        )
        await session.commit()
    assert applied.applied is True
    assert applied.revision == 2

    async with session_factory() as session:
        with pytest.raises(ProcedureStateConflict):
            await ProcedureStateStore(session).save_cas(
                cid, _state_a(), expected_revision=1, execution_id="tg:3"
            )
        await session.rollback()

    async with session_factory() as check:
        stored = await ProcedureStateStore(check).read(cid)
    assert stored.revision == 2
    assert stored.state == _state_b()
    assert stored.last_execution_id == "tg:2"


@pytest.mark.asyncio
@pytest.mark.db
async def test_db_without_execution_id_keep_legacy_behaviour(session_factory):
    cid = await _make_conversation(session_factory)

    async with session_factory() as session:
        await ProcedureStateStore(session).save_cas(
            cid, _state_a(), expected_revision=None
        )
        await session.commit()
    async with session_factory() as session:
        result = await ProcedureStateStore(session).save_cas(
            cid, _state_b(), expected_revision=1
        )
        await session.commit()
    assert result.applied is True
    assert result.revision == 2

    async with session_factory() as check:
        stored = await ProcedureStateStore(check).read(cid)
    assert stored.revision == 2
    assert stored.last_execution_id is None

    # A stale revision with no execution id is still a plain conflict.
    async with session_factory() as session:
        with pytest.raises(ProcedureStateConflict):
            await ProcedureStateStore(session).save_cas(
                cid, _state_a(), expected_revision=1
            )
        await session.rollback()


@pytest.mark.asyncio
@pytest.mark.db
async def test_db_last_execution_id_is_globally_unique(session_factory):
    cid_1 = await _make_conversation(session_factory, cid="chat1")
    cid_2 = await _make_conversation(session_factory, cid="chat2")

    async with session_factory() as session:
        await ProcedureStateStore(session).save_cas(
            cid_1, _state_a(), expected_revision=None, execution_id="tg:1"
        )
        await session.commit()

    # The SAME execution id must not advance a SECOND conversation.
    async with session_factory() as session:
        with pytest.raises(IntegrityError):
            await ProcedureStateStore(session).save_cas(
                cid_2, _state_a(), expected_revision=None, execution_id="tg:1"
            )
        await session.rollback()
