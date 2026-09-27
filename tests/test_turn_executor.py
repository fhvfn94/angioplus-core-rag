"""
SupportTurnExecutor integration tests (PostgreSQL).

Exercises the Slice 9 durable application boundary end to end, using the
REAL frozen orchestrator (analyzer port injected, no network):

  * cold start persists the first state at ``revision == 1`` and returns the
    orchestrator's :class:`TurnResult` plus the persisted coordinates;
  * multi-turn continuity grows the attempt log and bumps the revision;
  * PROCESS performs NO database work (no open transaction across provider
    calls);
  * a restart (fresh executor instance) reloads and continues from the DB;
  * a stale expected revision surfaces as :class:`ProcedureStateConflict`
    with no overwrite;
  * the cursor / pending-clarification / procedure identity are persisted
    exactly as the orchestrator produced them;
  * ESCALATE / RESOLVE persist the procedure state WITHOUT any FSM or Tracker
    side effect;
  * ``history`` is an opaque caller-supplied pass-through;
  * a mid-write DB failure leaves NO partial state.

DB-backed: marked ``@pytest.mark.db`` (fail-closed safety guard in conftest).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import delete, select

from support.models import (
    Conversation,
    InboundEvent,
    ProcedureStateAttemptRow,
    ProcessedEvent,
)
from support.procedure import load_procedure
from support.procedure_state_store import (
    ProcedureStateConflict,
    ProcedureStateStore,
)
from support.router import EscalateReason, RouteAction
from support.session_service import SessionService
from support.turn_executor import SupportTurnExecutor


pytestmark = [pytest.mark.asyncio, pytest.mark.db]

FIXTURES = Path(__file__).parent / "fixtures" / "procedures"


# ============================================================
# HELPERS
# ============================================================

def _black_screen():
    return load_procedure(FIXTURES / "black_screen.yaml")


async def _make_conversation(
    session_factory, channel="telegram", uid="u1", cid="chat1"
) -> str:
    async with session_factory() as session:
        conv = await SessionService(session).create(channel, uid, cid)
        await session.commit()
        return conv.conversation_id


def _obs_json(
    *,
    scope="in_scope",
    actions=(),
    problem_status=None,
    human_requested=False,
    clarification_facts=(),
    new_issue=None,
    request_kind="problem",
) -> str:
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
            "request_kind": request_kind,
        }
    )


def _analyzer(raw: str):
    return lambda prompt, history: raw


class _FailingAfterAttemptsDeleteStore(ProcedureStateStore):
    """Fails AFTER deleting the attempt log (atomicity probe)."""

    async def _replace_attempts(self, conversation_id, state) -> None:
        await self._session.execute(
            delete(ProcedureStateAttemptRow).where(
                ProcedureStateAttemptRow.conversation_id == conversation_id
            )
        )
        raise RuntimeError("simulated write failure")


async def _read(session_factory, cid):
    async with session_factory() as session:
        return await ProcedureStateStore(session).read(cid)


# ============================================================
# COLD START / MULTI-TURN CONTINUITY
# ============================================================

async def test_cold_start_persists_first_turn_at_revision_one(session_factory):
    procedure = _black_screen()
    cid = await _make_conversation(session_factory)
    executor = SupportTurnExecutor(session_factory=session_factory)

    result = await executor.execute_persisted_turn(
        conversation_id=cid,
        message="my screen is black",
        procedure=procedure,
        analyzer_provider=_analyzer(_obs_json()),
    )

    assert result.conversation_id == cid
    assert result.revision == 1
    assert result.turn.decision.action is RouteAction.RECOMMEND_STEP
    assert result.turn.decision.next_step == "CHECK_POWER"
    assert result.state.last_recommended_step == "CHECK_POWER"
    # Result exposes the orchestrator's own TurnResult verbatim.
    assert result.turn.state == result.state

    stored = await _read(session_factory, cid)
    assert stored.revision == 1
    assert stored.state == result.state


async def test_multi_turn_continuity_grows_attempt_log(session_factory):
    procedure = _black_screen()
    cid = await _make_conversation(session_factory)
    executor = SupportTurnExecutor(session_factory=session_factory)

    r1 = await executor.execute_persisted_turn(
        conversation_id=cid,
        message="my screen is black",
        procedure=procedure,
        analyzer_provider=_analyzer(_obs_json()),
    )
    assert r1.revision == 1

    r2 = await executor.execute_persisted_turn(
        conversation_id=cid,
        message="checked the power, still black",
        procedure=procedure,
        analyzer_provider=_analyzer(
            _obs_json(
                actions=[("CHECK_POWER", "failed")],
                problem_status="ongoing",
            )
        ),
    )
    assert r2.revision == 2
    assert r2.turn.decision.next_step == "CHECK_CABLE"
    assert [a.step_id for a in r2.state.step_attempts] == ["CHECK_POWER"]
    assert r2.state.last_recommended_step == "CHECK_CABLE"

    # Reload: every revision is a durable, lossless snapshot.
    stored = await _read(session_factory, cid)
    assert stored.revision == 2
    assert stored.state == r2.state


async def test_restart_reloads_and_continues_from_db(session_factory):
    procedure = _black_screen()
    cid = await _make_conversation(session_factory)

    first = SupportTurnExecutor(session_factory=session_factory)
    await first.execute_persisted_turn(
        conversation_id=cid,
        message="my screen is black",
        procedure=procedure,
        analyzer_provider=_analyzer(_obs_json()),
    )

    # A brand-new executor (process restart) has no in-memory state.
    restarted = SupportTurnExecutor(session_factory=session_factory)
    r2 = await restarted.execute_persisted_turn(
        conversation_id=cid,
        message="checked the power, still black",
        procedure=procedure,
        analyzer_provider=_analyzer(
            _obs_json(
                actions=[("CHECK_POWER", "failed")],
                problem_status="ongoing",
            )
        ),
    )
    assert r2.revision == 2
    assert r2.turn.decision.next_step == "CHECK_CABLE"
    assert [a.step_id for a in r2.state.step_attempts] == ["CHECK_POWER"]


# ============================================================
# NO DB TRANSACTION ACROSS PROCESSING
# ============================================================

async def test_process_runs_without_open_db_transaction(session_factory, engine):
    procedure = _black_screen()
    cid = await _make_conversation(session_factory)
    executor = SupportTurnExecutor(session_factory=session_factory)

    observed: dict = {}

    def probe(prompt, history):
        observed["checkedout"] = engine.pool.checkedout()
        return _obs_json()

    await executor.execute_persisted_turn(
        conversation_id=cid,
        message="my screen is black",
        procedure=procedure,
        analyzer_provider=probe,
    )

    # The analyzer (PROCESS) ran with zero pooled connections checked out:
    # the read transaction was closed before it, the write opened after it.
    assert observed["checkedout"] == 0


# ============================================================
# STALE REVISION / CONFLICT
# ============================================================

async def test_stale_revision_raises_conflict_without_overwrite(session_factory):
    procedure = _black_screen()
    cid = await _make_conversation(session_factory)

    seed = SupportTurnExecutor(session_factory=session_factory)
    r1 = await seed.execute_persisted_turn(
        conversation_id=cid,
        message="my screen is black",
        procedure=procedure,
        analyzer_provider=_analyzer(_obs_json()),
    )
    assert r1.revision == 1

    class _CompetingWriterStore(ProcedureStateStore):
        """A concurrent turn wins the race between READ and WRITE.

        The competing revision is committed from a SEPARATE transaction (its
        own connection), simulating a different turn that advanced the state
        while this one was in PROCESS; the executor's CAS must then lose.
        """

        async def save_cas(self, conversation_id, state, *, expected_revision):
            async with session_factory() as other:
                await ProcedureStateStore(other).save_cas(
                    conversation_id, r1.state, expected_revision=expected_revision
                )
                await other.commit()
            return await super().save_cas(
                conversation_id, state, expected_revision=expected_revision
            )

    executor = SupportTurnExecutor(
        session_factory=session_factory, store_factory=_CompetingWriterStore
    )
    with pytest.raises(ProcedureStateConflict):
        await executor.execute_persisted_turn(
            conversation_id=cid,
            message="checked the power, still black",
            procedure=procedure,
            analyzer_provider=_analyzer(
                _obs_json(
                    actions=[("CHECK_POWER", "failed")],
                    problem_status="ongoing",
                )
            ),
        )

    stored = await _read(session_factory, cid)
    assert stored.revision == 2  # the competing writer won; no clobber
    assert stored.state == r1.state


# ============================================================
# PERSISTED FIELDS: CURSOR / CLARIFICATION / IDENTITY
# ============================================================

async def test_procedure_identity_is_persisted(session_factory):
    procedure = _black_screen()
    cid = await _make_conversation(session_factory)
    executor = SupportTurnExecutor(session_factory=session_factory)

    result = await executor.execute_persisted_turn(
        conversation_id=cid,
        message="my screen is black",
        procedure=procedure,
        analyzer_provider=_analyzer(_obs_json()),
    )
    assert result.state.procedure_id == "BLACK_SCREEN"
    assert result.state.procedure_version == procedure.version

    stored = await _read(session_factory, cid)
    assert stored.state.procedure_id == "BLACK_SCREEN"
    assert stored.state.procedure_version == procedure.version


async def test_clarify_persists_pending_clarification(session_factory):
    procedure = _black_screen()
    cid = await _make_conversation(session_factory)
    executor = SupportTurnExecutor(session_factory=session_factory)

    result = await executor.execute_persisted_turn(
        conversation_id=cid,
        message="something is wrong",
        procedure=procedure,
        analyzer_provider=_analyzer(
            _obs_json(problem_status="ongoing")  # ongoing, nothing to attribute
        ),
    )
    assert result.turn.decision.action is RouteAction.CLARIFY
    assert result.state.pending_clarification is True

    stored = await _read(session_factory, cid)
    assert stored.state.pending_clarification is True


# ============================================================
# ESCALATE / RESOLVE: NO FSM OR TRACKER SIDE EFFECT
# ============================================================

async def _assert_no_fsm_or_tracker_side_effect(session_factory, cid):
    async with session_factory() as check:
        conv = (
            await check.execute(
                select(Conversation).where(Conversation.conversation_id == cid)
            )
        ).scalar_one()
        assert conv.conversation_mode == "ai"
        assert conv.support_status == "active"
        assert conv.handoff_status is None
        assert conv.external_case_id is None
        assert conv.operator_provider is None

        processed = (await check.execute(select(ProcessedEvent))).scalars().all()
        inbound = (await check.execute(select(InboundEvent))).scalars().all()
        assert processed == []
        assert inbound == []


async def test_escalate_persists_state_without_fsm_or_tracker_effect(
    session_factory,
):
    procedure = _black_screen()
    cid = await _make_conversation(session_factory)
    executor = SupportTurnExecutor(session_factory=session_factory)

    await executor.execute_persisted_turn(
        conversation_id=cid,
        message="my screen is black",
        procedure=procedure,
        analyzer_provider=_analyzer(_obs_json()),
    )
    await executor.execute_persisted_turn(
        conversation_id=cid,
        message="checked power, still black",
        procedure=procedure,
        analyzer_provider=_analyzer(
            _obs_json(
                actions=[("CHECK_POWER", "failed")], problem_status="ongoing"
            )
        ),
    )
    r3 = await executor.execute_persisted_turn(
        conversation_id=cid,
        message="reseated the cable, still black",
        procedure=procedure,
        analyzer_provider=_analyzer(
            _obs_json(
                actions=[("CHECK_CABLE", "failed")], problem_status="ongoing"
            )
        ),
    )

    assert r3.revision == 3
    assert r3.turn.decision.action is RouteAction.ESCALATE
    assert r3.turn.decision.escalate_reason is EscalateReason.PROCEDURE_TERMINAL
    assert r3.turn.escalation_reason is EscalateReason.PROCEDURE_TERMINAL

    stored = await _read(session_factory, cid)
    assert stored.revision == 3
    assert [a.step_id for a in stored.state.step_attempts] == [
        "CHECK_POWER",
        "CHECK_CABLE",
    ]

    await _assert_no_fsm_or_tracker_side_effect(session_factory, cid)


async def test_resolve_persists_state_without_fsm_or_tracker_effect(
    session_factory,
):
    procedure = _black_screen()
    cid = await _make_conversation(session_factory)
    executor = SupportTurnExecutor(session_factory=session_factory)

    await executor.execute_persisted_turn(
        conversation_id=cid,
        message="my screen is black",
        procedure=procedure,
        analyzer_provider=_analyzer(_obs_json()),
    )
    result = await executor.execute_persisted_turn(
        conversation_id=cid,
        message="it works now, thanks",
        procedure=procedure,
        analyzer_provider=_analyzer(_obs_json(problem_status="resolved")),
    )

    assert result.turn.decision.action is RouteAction.RESOLVE
    assert result.revision == 2

    stored = await _read(session_factory, cid)
    assert stored.state.problem_status.value == "resolved"

    await _assert_no_fsm_or_tracker_side_effect(session_factory, cid)


# ============================================================
# HISTORY PASS-THROUGH
# ============================================================

async def test_history_is_passed_through_opaquely(session_factory):
    procedure = _black_screen()
    cid = await _make_conversation(session_factory)
    executor = SupportTurnExecutor(session_factory=session_factory)

    history = [
        {"role": "user", "text": "hi"},
        {"role": "assistant", "text": "hello, how can I help?"},
    ]
    seen: dict = {}

    def probe(prompt, history_arg):
        seen["history"] = list(history_arg)
        return _obs_json()

    await executor.execute_persisted_turn(
        conversation_id=cid,
        message="my screen is black",
        procedure=procedure,
        analyzer_provider=probe,
        history=history,
    )
    assert seen["history"] == history


# ============================================================
# ATOMICITY
# ============================================================

async def test_failed_write_leaves_no_partial_state(session_factory):
    procedure = _black_screen()
    cid = await _make_conversation(session_factory)

    ok = SupportTurnExecutor(session_factory=session_factory)
    r1 = await ok.execute_persisted_turn(
        conversation_id=cid,
        message="my screen is black",
        procedure=procedure,
        analyzer_provider=_analyzer(_obs_json()),
    )
    assert r1.revision == 1

    failing = SupportTurnExecutor(
        session_factory=session_factory,
        store_factory=_FailingAfterAttemptsDeleteStore,
    )
    with pytest.raises(RuntimeError):
        await failing.execute_persisted_turn(
            conversation_id=cid,
            message="checked power, still black",
            procedure=procedure,
            analyzer_provider=_analyzer(
                _obs_json(
                    actions=[("CHECK_POWER", "failed")],
                    problem_status="ongoing",
                )
            ),
        )

    stored = await _read(session_factory, cid)
    assert stored.revision == 1
    assert stored.state == r1.state
