"""
ProcedureStateStore integration tests (PostgreSQL).

Exercises the compare-and-swap (CAS) persistence contract introduced in
Slice 9 -- the write path that the frozen Slice 8 ``ProcedureStateService``
does not provide:

  * cold start inserts the first state with ``revision == 1``;
  * a CAS update bumps the revision and round-trips every field;
  * a STALE revision raises :class:`ProcedureStateConflict` and writes
    nothing (the stored state and attempt log are untouched);
  * a simultaneous cold start creates exactly ONE row (the loser conflicts,
    never overwrites);
  * a failure mid-write leaves NO partial state (state row + attempt log
    are atomic);
  * the frozen ``ProcedureStateService`` shares the SAME semantic model
    (its writes land at ``revision == 1``).

DB-backed: marked ``@pytest.mark.db`` (fail-closed safety guard in conftest).
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from support.models import ProcedureStateAttemptRow, ProcedureStateRow
from support.observation import ProblemStatus, StepOutcome
from support.procedure_state import ProcedureState, StepAttempt
from support.procedure_state_service import ProcedureStateService
from support.procedure_state_store import (
    COLD_START_REVISION,
    ProcedureStateConflict,
    ProcedureStateStore,
)
from support.session_service import SessionService


pytestmark = [pytest.mark.asyncio, pytest.mark.db]


# ============================================================
# HELPERS
# ============================================================

async def _make_conversation(
    session_factory, channel="telegram", uid="u1", cid="chat1"
) -> str:
    async with session_factory() as session:
        conv = await SessionService(session).create(channel, uid, cid)
        await session.commit()
        return conv.conversation_id


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


class _FailingAfterAttemptsDeleteStore(ProcedureStateStore):
    """A store that fails AFTER deleting the attempt log.

    Used to prove atomicity: the CAS UPDATE and the attempt-log delete must
    both roll back together, leaving no partial write.
    """

    async def _replace_attempts(self, conversation_id, state) -> None:
        await self._session.execute(
            delete(ProcedureStateAttemptRow).where(
                ProcedureStateAttemptRow.conversation_id == conversation_id
            )
        )
        raise RuntimeError("simulated write failure")


# ============================================================
# READ
# ============================================================

async def test_read_returns_none_when_absent(session_factory):
    cid = await _make_conversation(session_factory)
    async with session_factory() as session:
        assert await ProcedureStateStore(session).read(cid) is None
    async with session_factory() as check:
        rows = (
            await check.execute(select(ProcedureStateRow))
        ).scalars().all()
    assert rows == []


# ============================================================
# COLD START
# ============================================================

async def test_cold_start_insert_sets_revision_one(session_factory):
    assert COLD_START_REVISION == 1
    cid = await _make_conversation(session_factory)

    async with session_factory() as session:
        result = await ProcedureStateStore(session).save_cas(
            cid, _state_a(), expected_revision=None
        )
        await session.commit()

    assert result.applied is True
    assert result.revision == 1
    assert result.state == _state_a()
    async with session_factory() as check:
        stored = await ProcedureStateStore(check).read(cid)
    assert stored is not None
    assert stored.revision == 1
    assert stored.state == _state_a()


# ============================================================
# CAS UPDATE
# ============================================================

async def test_cas_update_increments_revision_and_round_trips(session_factory):
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
    assert stored.state == _state_b()
    # replace-as-a-set: exactly two attempt rows, in sequence order.
    assert [a.sequence for a in stored.state.step_attempts] == [0, 1]
    assert [a.step_id for a in stored.state.step_attempts] == [
        "CHECK_POWER",
        "CHECK_CABLE",
    ]


async def test_stale_revision_raises_conflict_and_does_not_overwrite(
    session_factory,
):
    cid = await _make_conversation(session_factory)
    async with session_factory() as session:
        await ProcedureStateStore(session).save_cas(
            cid, _state_a(), expected_revision=None
        )
        await session.commit()
    # Advance to revision 2.
    async with session_factory() as session:
        await ProcedureStateStore(session).save_cas(
            cid, _state_b(), expected_revision=1
        )
        await session.commit()

    # A writer that still believes the revision is 1 must lose.
    async with session_factory() as session:
        with pytest.raises(ProcedureStateConflict):
            await ProcedureStateStore(session).save_cas(
                cid, _state_a(), expected_revision=1
            )
        await session.rollback()

    async with session_factory() as check:
        stored = await ProcedureStateStore(check).read(cid)
    assert stored.revision == 2
    assert stored.state == _state_b()  # winner's write intact, not clobbered


# ============================================================
# COLD-START RACE
# ============================================================

async def test_simultaneous_cold_start_creates_single_row(session_factory):
    cid = await _make_conversation(session_factory)

    async def writer():
        async with session_factory() as session:
            try:
                result = await ProcedureStateStore(session).save_cas(
                    cid, _state_a(), expected_revision=None
                )
                await session.commit()
                return result.revision
            except ProcedureStateConflict:
                await session.rollback()
                return "conflict"

    results = await asyncio.gather(writer(), writer())

    assert results.count("conflict") == 1
    assert [r for r in results if r != "conflict"] == [1]

    async with session_factory() as check:
        stored = await ProcedureStateStore(check).read(cid)
        rows = (
            await check.execute(select(ProcedureStateRow))
        ).scalars().all()
    assert len(rows) == 1
    assert stored.revision == 1
    assert stored.state == _state_a()


# ============================================================
# ATOMICITY
# ============================================================

async def test_failed_write_leaves_no_partial_state(session_factory):
    cid = await _make_conversation(session_factory)
    async with session_factory() as session:
        await ProcedureStateStore(session).save_cas(
            cid, _state_a(), expected_revision=None
        )
        await session.commit()

    async with session_factory() as session:
        with pytest.raises(RuntimeError):
            await _FailingAfterAttemptsDeleteStore(session).save_cas(
                cid, _state_b(), expected_revision=1
            )
        await session.rollback()

    async with session_factory() as check:
        stored = await ProcedureStateStore(check).read(cid)
        attempt_rows = (
            await check.execute(select(ProcedureStateAttemptRow))
        ).scalars().all()
    # State row unchanged (revision not bumped, old state intact).
    assert stored.revision == 1
    assert stored.state == _state_a()
    # Attempt log unchanged (still the winner/_state_a log).
    assert [(a.sequence, a.step_id) for a in attempt_rows] == [(0, "CHECK_POWER")]


# ============================================================
# SHARED SEMANTIC MODEL WITH THE FROZEN SERVICE
# ============================================================

async def test_frozen_service_write_lands_at_revision_one(session_factory):
    cid = await _make_conversation(session_factory)

    # The frozen Slice 8 adapter never sets ``revision``.
    async with session_factory() as session:
        await ProcedureStateService(session).save(cid, _state_a())
        await session.commit()

    async with session_factory() as check:
        stored = await ProcedureStateStore(check).read(cid)
    assert stored.revision == 1

    # ...and the CAS can continue from that row on the SAME model.
    async with session_factory() as session:
        result = await ProcedureStateStore(session).save_cas(
            cid, _state_b(), expected_revision=1
        )
        await session.commit()
    assert result.applied is True
    assert result.revision == 2


async def test_revision_below_one_rejected_by_check(session_factory):
    cid = await _make_conversation(session_factory)
    async with session_factory() as session:
        session.add(ProcedureStateRow(conversation_id=cid, revision=0))
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()
