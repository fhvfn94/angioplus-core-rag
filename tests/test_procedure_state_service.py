"""
ProcedureStateService integration tests (PostgreSQL).

Exercises the durable projection of the pure
:class:`~support.procedure_state.ProcedureState` value object:

  * lossless ``save`` -> ``handle`` round trips (including the attempt log),
  * upsert / replace-as-a-set semantics (no double-append),
  * DB CHECK constraints and ON DELETE CASCADE,
  * FOR UPDATE serialization of concurrent writers,
  * the ACCEPTANCE property: a state persisted on one turn can be reloaded and
    drives the next REAL orchestrator turn identically.

DB-backed: marked ``@pytest.mark.db`` (fail-closed safety guard in conftest).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from support.models import (
    Conversation,
    ProcedureStateAttemptRow,
    ProcedureStateRow,
)
from support.observation import ProblemStatus, StepOutcome
from support.orchestrator import process_support_turn
from support.procedure import load_procedure
from support.procedure_state import (
    ProcedureState,
    StepAttempt,
    initial_state,
)
from support.procedure_state_service import ProcedureStateService
from support.session_service import SessionService


pytestmark = [pytest.mark.asyncio, pytest.mark.db]

FIXTURES = Path(__file__).parent / "fixtures" / "procedures"


# ============================================================
# HELPERS
# ============================================================

def _black_screen():
    return load_procedure(FIXTURES / "black_screen.yaml")


async def _make_conversation(session, channel="telegram", uid="u1", cid="chat1"):
    svc = SessionService(session)
    conv = await svc.create(channel, uid, cid)
    await session.commit()
    return conv


def _populated_state() -> ProcedureState:
    """A fully-populated state so the round trip touches every column."""
    return ProcedureState(
        issue="black screen",
        procedure_id="BLACK_SCREEN",
        procedure_version=1,
        last_recommended_step="CHECK_CABLE",
        step_attempts=(
            StepAttempt("CHECK_POWER", StepOutcome.FAILED, 0),
            StepAttempt("CHECK_CABLE", StepOutcome.SKIPPED, 1),
        ),
        problem_status=ProblemStatus.ONGOING,
        pending_clarification=True,
        clarification_topic="CHECK_CABLE",
        schema_version=1,
    )


def _obs_json(
    *,
    scope="in_scope",
    actions=(),
    problem_status=None,
    human_requested=False,
    clarification_facts=(),
    new_issue=None,
) -> str:
    """A strictly-valid analyzer JSON object (exact schema)."""
    return json.dumps(
        {
            "scope": scope,
            "reported_actions": [
                {"step_id": sid, "outcome": outcome} for sid, outcome in actions
            ],
            "problem_status": problem_status,
            "human_requested": human_requested,
            "clarification_facts": list(clarification_facts),
            "new_issue": new_issue,
        }
    )


def _analyzer(raw: str):
    """An AnalyzerProvider returning a fixed raw JSON string."""
    return lambda prompt, history: raw


# ============================================================
# ROUND TRIP
# ============================================================

async def test_handle_returns_none_and_creates_nothing_when_absent(session):
    conv = await _make_conversation(session)
    svc = ProcedureStateService(session)

    assert await svc.handle(conv.conversation_id) is None
    rows = (await session.execute(select(ProcedureStateRow))).scalars().all()
    assert rows == []


async def test_save_then_handle_round_trips_every_field(session):
    conv = await _make_conversation(session)
    svc = ProcedureStateService(session)
    original = _populated_state()

    await svc.save(conv.conversation_id, original)
    await session.commit()

    loaded = await svc.handle(conv.conversation_id)
    assert loaded == original  # frozen-dataclass equality, attempts included


async def test_saving_initial_state_round_trips_to_initial_state(session):
    conv = await _make_conversation(session)
    svc = ProcedureStateService(session)

    await svc.save(conv.conversation_id, initial_state())
    await session.commit()

    assert await svc.handle(conv.conversation_id) == initial_state()


async def test_attempts_are_returned_in_sequence_order(session):
    conv = await _make_conversation(session)
    svc = ProcedureStateService(session)
    state = ProcedureState(
        step_attempts=(
            StepAttempt("A", StepOutcome.FAILED, 0),
            StepAttempt("B", StepOutcome.SUCCEEDED, 1),
            StepAttempt("C", StepOutcome.DONE_UNKNOWN, 2),
        )
    )

    await svc.save(conv.conversation_id, state)
    await session.commit()

    loaded = await svc.handle(conv.conversation_id)
    assert [a.sequence for a in loaded.step_attempts] == [0, 1, 2]
    assert [a.step_id for a in loaded.step_attempts] == ["A", "B", "C"]
    assert [a.outcome for a in loaded.step_attempts] == [
        StepOutcome.FAILED,
        StepOutcome.SUCCEEDED,
        StepOutcome.DONE_UNKNOWN,
    ]


async def test_repeated_save_of_identical_state_does_not_double_append(session):
    conv = await _make_conversation(session)
    svc = ProcedureStateService(session)
    state = _populated_state()

    await svc.save(conv.conversation_id, state)
    await session.commit()
    await svc.save(conv.conversation_id, state)
    await session.commit()

    loaded = await svc.handle(conv.conversation_id)
    assert loaded == state
    attempt_rows = (
        await session.execute(select(ProcedureStateAttemptRow))
    ).scalars().all()
    assert len(attempt_rows) == 2  # replace-as-a-set, not append


async def test_save_replaces_previous_attempt_log(session):
    conv = await _make_conversation(session)
    svc = ProcedureStateService(session)

    await svc.save(
        conv.conversation_id,
        ProcedureState(
            step_attempts=(StepAttempt("A", StepOutcome.FAILED, 0),)
        ),
    )
    await session.commit()

    await svc.save(
        conv.conversation_id,
        ProcedureState(
            step_attempts=(StepAttempt("B", StepOutcome.SUCCEEDED, 0),)
        ),
    )
    await session.commit()

    loaded = await svc.handle(conv.conversation_id)
    assert [a.step_id for a in loaded.step_attempts] == ["B"]


# ============================================================
# DB CONSTRAINTS
# ============================================================

async def test_forbidden_outcome_rejected_by_check(session):
    conv = await _make_conversation(session)
    svc = ProcedureStateService(session)
    await svc.save(conv.conversation_id, initial_state())
    await session.commit()

    session.add(
        ProcedureStateAttemptRow(
            conversation_id=conv.conversation_id,
            sequence=0,
            step_id="CHECK_POWER",
            outcome="bogus",  # not a StepOutcome value
        )
    )
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()


async def test_forbidden_problem_status_rejected_by_check(session):
    conv = await _make_conversation(session)
    svc = ProcedureStateService(session)
    await svc.save(conv.conversation_id, initial_state())
    await session.commit()

    row = (
        await session.execute(
            select(ProcedureStateRow).where(
                ProcedureStateRow.conversation_id == conv.conversation_id
            )
        )
    ).scalar_one()
    row.problem_status = "bogus"  # not a ProblemStatus value
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()


async def test_duplicate_attempt_sequence_rejected(session):
    conv = await _make_conversation(session)
    svc = ProcedureStateService(session)
    await svc.save(conv.conversation_id, initial_state())
    await session.commit()

    session.add_all(
        [
            ProcedureStateAttemptRow(
                conversation_id=conv.conversation_id,
                sequence=0,
                step_id="A",
                outcome="failed",
            ),
            ProcedureStateAttemptRow(
                conversation_id=conv.conversation_id,
                sequence=0,  # duplicate (conversation_id, sequence)
                step_id="B",
                outcome="failed",
            ),
        ]
    )
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()


# ============================================================
# CASCADE
# ============================================================

async def test_deleting_conversation_cascades_to_state_and_attempts(session):
    conv = await _make_conversation(session)
    svc = ProcedureStateService(session)
    await svc.save(conv.conversation_id, _populated_state())
    await session.commit()

    await session.execute(
        delete(Conversation).where(
            Conversation.conversation_id == conv.conversation_id
        )
    )
    await session.commit()

    assert await svc.handle(conv.conversation_id) is None
    state_rows = (await session.execute(select(ProcedureStateRow))).scalars().all()
    attempt_rows = (
        await session.execute(select(ProcedureStateAttemptRow))
    ).scalars().all()
    assert state_rows == []
    assert attempt_rows == []


# ============================================================
# CONCURRENCY (FOR UPDATE serialization)
# ============================================================

async def test_concurrent_saves_are_serialized(session_factory):
    async with session_factory() as setup:
        conv = await SessionService(setup).create("telegram", "u1", "chat1")
        await ProcedureStateService(setup).save(
            conv.conversation_id, initial_state()
        )
        await setup.commit()
        conversation_id = conv.conversation_id

    target = ProcedureState(
        step_attempts=(
            StepAttempt("CHECK_POWER", StepOutcome.FAILED, 0),
            StepAttempt("CHECK_CABLE", StepOutcome.FAILED, 1),
        )
    )

    async def writer():
        async with session_factory() as s:
            await ProcedureStateService(s).save(conversation_id, target)
            await s.commit()

    # Without FOR UPDATE the two replace-as-a-set writers would race on the
    # unique (conversation_id, sequence); locking serializes them.
    await asyncio.gather(writer(), writer())

    async with session_factory() as check:
        loaded = await ProcedureStateService(check).handle(conversation_id)
    assert loaded is not None
    assert [a.sequence for a in loaded.step_attempts] == [0, 1]
    assert [a.step_id for a in loaded.step_attempts] == [
        "CHECK_POWER",
        "CHECK_CABLE",
    ]


# ============================================================
# ACCEPTANCE: persisted state drives the next real orchestrator turn
# ============================================================

async def test_persisted_state_drives_next_orchestrator_turn(session):
    procedure = _black_screen()
    conv = await _make_conversation(session)
    svc = ProcedureStateService(session)

    # Turn 1: cold start recommends the entry step and is persisted.
    r1 = process_support_turn(
        message="my screen is black",
        procedure=procedure,
        analyzer_provider=_analyzer(_obs_json()),
    )
    assert r1.decision.next_step == "CHECK_POWER"
    await svc.save(conv.conversation_id, r1.state)
    await session.commit()

    # The DB round trip is lossless and carries the cursor continuity fact.
    loaded = await svc.handle(conv.conversation_id)
    assert loaded == r1.state
    assert loaded.last_recommended_step == "CHECK_POWER"

    # Turn 2 continues from the RELOADED state (proves persistence drives it).
    r2 = process_support_turn(
        message="checked the power, still black",
        state=loaded,
        procedure=procedure,
        analyzer_provider=_analyzer(
            _obs_json(
                actions=[("CHECK_POWER", "failed")],
                problem_status="ongoing",
            )
        ),
    )
    assert r2.decision.next_step == "CHECK_CABLE"
    assert [a.step_id for a in r2.state.step_attempts] == ["CHECK_POWER"]
    await svc.save(conv.conversation_id, r2.state)
    await session.commit()

    # The attempt log survived the second round trip.
    loaded2 = await svc.handle(conv.conversation_id)
    assert loaded2 == r2.state
    assert [a.step_id for a in loaded2.step_attempts] == ["CHECK_POWER"]
    assert loaded2.last_recommended_step == "CHECK_CABLE"
