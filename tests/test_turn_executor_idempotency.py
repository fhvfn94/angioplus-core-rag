"""
HL-2b -- idempotent turn execution at the ``SupportTurnExecutor`` boundary.

Two layers of proof, both about the ONE shared durable lifecycle
(``SupportTurnExecutor._execute_persisted``: READ -> (maybe short-circuit)
-> PROCESS -> CAS WRITE):

OFFLINE (no database)
    A fake store + fake session factory drive the lifecycle directly, so the
    executor's OWN logic is proven without PostgreSQL:

      * a fresh execution forwards the OPAQUE ``execution_id`` to the store;
      * repeating the SAME ``execution_id`` short-circuits BEFORE PROCESS and
        returns the ACTUALLY PERSISTED state (never a recomputed candidate);
      * when the CAS loses a race to the SAME ``execution_id`` AFTER PROCESS,
        the recomputed turn/state is DISCARDED and the PERSISTED state is
        returned (``applied=False``) -- correct WITHOUT single-flight;
      * a DIFFERENT ``execution_id`` is not short-circuited;
      * omitting ``execution_id`` keeps the EXACT legacy ``save_cas`` call
        shape (backward compatibility);
      * provider work (PROCESS) runs with NO session open (no DB transaction
        is held across the provider calls);
      * a ``ProcedureStateConflict`` is propagated unchanged (no retry).

DB-BACKED (``@pytest.mark.db``; fail-closed guard in conftest)
    The REAL ``ProcedureStateStore`` is driven through the executor with a
    fake PROCESS, simulating a crash between the CAS commit and the outer
    finalization and then replaying the SAME execution id:

      * the first execution advances the revision exactly once;
      * the replay advances NOTHING (revision unchanged, attempt log
        unchanged) and returns the persisted state with ``replayed=True``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import select

from support.models import ProcedureStateAttemptRow
from support.observation import ProblemStatus, StepOutcome
from support.procedure_state import ProcedureState, StepAttempt
from support.procedure_state_store import (
    COLD_START_REVISION,
    CasResult,
    PersistedProcedureState,
    ProcedureStateConflict,
    ProcedureStateStore,
)
from support.session_service import SessionService
from support.turn_executor import PersistedTurnResult, SupportTurnExecutor


# ============================================================
# FIXTURES / DOUBLES (offline)
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


class _RecordingSession:
    def __init__(self, factory: "_RecordingSessionFactory") -> None:
        self._factory = factory

    async def __aenter__(self):
        self._factory.open += 1
        return self

    async def __aexit__(self, *exc_info):
        self._factory.open -= 1
        return False

    async def commit(self):
        self._factory.commits += 1

    async def rollback(self):
        self._factory.rollbacks += 1


class _RecordingSessionFactory:
    """Session-factory double that tracks open/uncommitted sessions."""

    def __init__(self) -> None:
        self.open = 0
        self.opened = 0
        self.commits = 0
        self.rollbacks = 0

    def __call__(self) -> _RecordingSession:
        self.opened += 1
        return _RecordingSession(self)


class _FakeStore:
    """Store double: records the identity read and the CAS call.

    ``applied`` / ``persisted_state`` let a test script a POST-CAS idempotent
    hit: the store reports ``applied=False`` and hands back the state that
    the winner ACTUALLY persisted (never the caller's candidate).
    """

    def __init__(
        self,
        *,
        stored=None,
        save_result=1,
        conflict=None,
        applied=True,
        persisted_state=None,
    ) -> None:
        self.stored = stored
        self.save_result = save_result
        self.conflict = conflict
        self.applied = applied
        self.persisted_state = persisted_state
        self.read_calls: list[str] = []
        self.save_calls: list[dict] = []

    async def read(self, conversation_id):
        self.read_calls.append(conversation_id)
        return self.stored

    async def save_cas(self, conversation_id, state, **kwargs):
        self.save_calls.append(
            {"conversation_id": conversation_id, "state": state, **kwargs}
        )
        if self.conflict is not None:
            raise self.conflict
        persisted = (
            self.persisted_state
            if self.persisted_state is not None
            else state
        )
        return CasResult(
            revision=self.save_result, applied=self.applied, state=persisted
        )


class _FakeRecoveryStore:
    """``TurnRecoveryStore`` double: NO records (these callers never project).

    ``save`` is an ASSERTION, not a no-op: none of the offline lifecycle
    tests supplies a ``projection_state``, so NO recovery record may ever be
    written. ``read`` returns ``None`` -- the legitimate ABSENT case.
    """

    async def save(self, payload):
        raise AssertionError(
            "no recovery record without a projection snapshot"
        )

    async def read(self, *, execution_id, conversation_id):
        return None


def _executor(store: _FakeStore) -> tuple[SupportTurnExecutor, _RecordingSessionFactory]:
    sessions = _RecordingSessionFactory()
    executor = SupportTurnExecutor(
        session_factory=sessions,
        store_factory=lambda session: store,
        recovery_store_factory=lambda session: _FakeRecoveryStore(),
    )
    return executor, sessions


# ============================================================
# OFFLINE: the shared durable lifecycle
# ============================================================

@pytest.mark.asyncio
async def test_first_execution_forwards_execution_id_and_applies():
    store = _FakeStore(stored=None, save_result=1)
    executor, sessions = _executor(store)

    candidate = _state_a()
    turn = SimpleNamespace(state=candidate)
    observed_open_during_process: list[int] = []

    def _process(previous_state):
        # Invariant: PROCESS runs with NO session open (no DB tx held).
        observed_open_during_process.append(sessions.open)
        assert previous_state is None  # unbound conversation
        return turn

    result = await executor._execute_persisted(
        conversation_id="chat-1", process=_process, execution_id="tg:1"
    )

    assert isinstance(result, PersistedTurnResult)
    assert result.replayed is False
    assert result.turn is turn
    assert result.state is candidate
    assert result.revision == 1

    # The OPAQUE execution id is forwarded verbatim; the store knows nothing
    # about its origin.
    assert store.save_calls[0]["execution_id"] == "tg:1"
    assert store.save_calls[0]["expected_revision"] is None

    assert observed_open_during_process == [0]
    assert sessions.commits == 1
    assert sessions.open == 0


@pytest.mark.asyncio
async def test_repeat_execution_short_circuits_before_processing():
    persisted = _state_b()
    store = _FakeStore(
        stored=PersistedProcedureState(
            state=persisted, revision=5, last_execution_id="tg:1"
        )
    )
    executor, sessions = _executor(store)

    processed: list = []

    def _process(previous_state):
        processed.append(previous_state)
        raise AssertionError("PROCESS must not run on an idempotent hit")

    result = await executor._execute_persisted(
        conversation_id="chat-1", process=_process, execution_id="tg:1"
    )

    assert result.replayed is True
    assert result.turn is None
    assert result.state is persisted  # the PERSISTED state, not a candidate
    assert result.revision == 5

    assert processed == []            # PROCESS skipped entirely
    assert store.save_calls == []     # no CAS attempted
    assert sessions.opened == 1       # only the identity READ
    assert sessions.commits == 0


@pytest.mark.asyncio
async def test_different_execution_id_is_not_short_circuited():
    persisted = _state_a()
    store = _FakeStore(
        stored=PersistedProcedureState(
            state=persisted, revision=2, last_execution_id="tg:OLD"
        ),
        save_result=3,
    )
    executor, _sessions = _executor(store)

    candidate = _state_b()
    turn = SimpleNamespace(state=candidate)
    seen: list = []

    def _process(previous_state):
        seen.append(previous_state)
        return turn

    result = await executor._execute_persisted(
        conversation_id="chat-1", process=_process, execution_id="tg:NEW"
    )

    assert seen == [persisted]           # PROCESS ran against the persisted state
    assert result.replayed is False
    assert result.state is candidate
    assert store.save_calls[0]["execution_id"] == "tg:NEW"
    assert store.save_calls[0]["expected_revision"] == 2


@pytest.mark.asyncio
async def test_omitting_execution_id_keeps_the_legacy_call_shape():
    store = _FakeStore(stored=None, save_result=COLD_START_REVISION)
    executor, _sessions = _executor(store)

    turn = SimpleNamespace(state=_state_a())

    result = await executor._execute_persisted(
        conversation_id="chat-1", process=lambda previous: turn
    )

    assert result.replayed is False
    # No ``execution_id`` keyword at all -> byte-identical legacy save_cas call.
    assert store.save_calls[0] == {
        "conversation_id": "chat-1",
        "state": turn.state,
        "expected_revision": None,
    }


@pytest.mark.asyncio
async def test_absent_execution_id_never_short_circuits_on_a_stored_id():
    # Even when the row already carries an execution id, a caller that does
    # NOT supply one must not be treated as a replay (legacy behaviour).
    store = _FakeStore(
        stored=PersistedProcedureState(
            state=_state_a(), revision=4, last_execution_id="tg:1"
        ),
        save_result=5,
    )
    executor, _sessions = _executor(store)

    turn = SimpleNamespace(state=_state_b())
    result = await executor._execute_persisted(
        conversation_id="chat-1", process=lambda previous: turn
    )

    assert result.replayed is False
    assert "execution_id" not in store.save_calls[0]


@pytest.mark.asyncio
async def test_conflict_is_propagated_without_retry():
    store = _FakeStore(
        stored=PersistedProcedureState(state=_state_a(), revision=1),
        conflict=ProcedureStateConflict("cas miss"),
    )
    executor, sessions = _executor(store)

    turn = SimpleNamespace(state=_state_b())

    with pytest.raises(ProcedureStateConflict):
        await executor._execute_persisted(
            conversation_id="chat-1",
            process=lambda previous: turn,
            execution_id="tg:1",
        )

    assert len(store.save_calls) == 1   # exactly ONE attempt, no retry
    assert sessions.rollbacks == 1
    assert sessions.commits == 0


@pytest.mark.asyncio
async def test_post_cas_same_id_race_discards_recomputed_state():
    """Worker B LOSES the same-execution-id race at the CAS (post-PROCESS).

    B reads the row (its execution id is still OLD, so B is NOT
    short-circuited), runs PROCESS and computes ``state_B``; by the time B's
    CAS runs, worker A has already persisted the SAME ``execution_id`` with
    ``state_A``. The store reports ``applied=False`` and returns the
    PERSISTED ``state_A``. The executor MUST discard the recomputed
    ``state_B``/turn and return the persisted winner.

    No external single-flight is involved: the correctness comes from the
    typed :class:`CasResult` alone.
    """
    store = _FakeStore(
        stored=PersistedProcedureState(
            state=_state_a(), revision=7, last_execution_id="tg:OLD"
        ),
        save_result=8,                # A advanced 7 -> 8
        applied=False,                # ...so B's CAS did NOT apply
        persisted_state=_state_a(),   # the PERSISTED winner state
    )
    executor, sessions = _executor(store)

    recomputed = _state_b()
    recomputed_turn = SimpleNamespace(state=recomputed)
    processed: list = []

    def _process(previous_state):
        processed.append(previous_state)
        return recomputed_turn

    result = await executor._execute_persisted(
        conversation_id="chat-1", process=_process, execution_id="tg:SAME"
    )

    # The post-CAS path DID run PROCESS (unlike the pre-PROCESS short-circuit).
    assert processed == [_state_a()]
    assert len(store.save_calls) == 1
    assert store.save_calls[0]["execution_id"] == "tg:SAME"
    assert sessions.commits == 1

    # The recomputed loser turn/state is DISCARDED...
    assert result.replayed is True
    assert result.turn is None
    assert result.state is not recomputed
    assert result.state != recomputed
    # ...and the PERSISTED winner state/revision is returned instead.
    assert result.state == _state_a()
    assert result.revision == 8


# ============================================================
# OFFLINE: public entry points forward the opaque id
# ============================================================

@pytest.mark.asyncio
async def test_public_entry_points_forward_execution_id(monkeypatch):
    captured: dict = {}

    async def _fake_execute_persisted(
        self, *, conversation_id, process, execution_id=None
    ):
        captured["conversation_id"] = conversation_id
        captured["execution_id"] = execution_id
        return "sentinel"

    monkeypatch.setattr(
        SupportTurnExecutor, "_execute_persisted", _fake_execute_persisted
    )

    executor = SupportTurnExecutor(session_factory=lambda: None)

    forwarded = await executor.execute_persisted_turn(
        conversation_id="chat-1",
        message="black screen",
        procedure=object(),
        analyzer_provider=object(),
        execution_id="web:uuid",
    )
    assert forwarded == "sentinel"
    assert captured == {"conversation_id": "chat-1", "execution_id": "web:uuid"}

    forwarded = await executor.execute_persisted_turn_from_observation(
        conversation_id="chat-2",
        observation=object(),
        procedure=object(),
    )
    assert forwarded == "sentinel"
    assert captured == {"conversation_id": "chat-2", "execution_id": None}


# ============================================================
# DB-BACKED: the REAL store, driven through the executor
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
async def test_db_crash_window_replay_advances_nothing(session_factory):
    cid = await _make_conversation(session_factory)
    executor = SupportTurnExecutor(
        session_factory=session_factory, store_factory=ProcedureStateStore
    )

    first_turn = SimpleNamespace(state=_state_a())

    # 1. First execution: CAS commits revision 1 with the execution id.
    first = await executor._execute_persisted(
        conversation_id=cid,
        process=lambda previous: first_turn,
        execution_id="tg:42",
    )
    assert first.replayed is False
    assert first.revision == 1

    # 2. Simulate a crash AFTER the CAS commit but BEFORE the caller's outer
    #    finalization, then the SAME event is redelivered: the executor
    #    short-circuits and returns the PERSISTED state, writing nothing.
    replayed = await executor._execute_persisted(
        conversation_id=cid,
        process=lambda previous: (_ for _ in ()).throw(
            AssertionError("PROCESS must not run on replay")
        ),
        execution_id="tg:42",
    )

    assert replayed.replayed is True
    assert replayed.turn is None
    assert replayed.revision == 1
    assert replayed.state == _state_a()

    # Nothing advanced, and the attempt log was NOT rewritten.
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
    assert stored.state == _state_a()
    assert stored.last_execution_id == "tg:42"
    assert [(a.sequence, a.step_id) for a in attempts] == [(0, "CHECK_POWER")]


@pytest.mark.asyncio
@pytest.mark.db
async def test_db_executor_stale_different_execution_id_conflicts(session_factory):
    cid = await _make_conversation(session_factory)

    seed = SupportTurnExecutor(
        session_factory=session_factory, store_factory=ProcedureStateStore
    )
    first = await seed._execute_persisted(
        conversation_id=cid,
        process=lambda previous: SimpleNamespace(state=_state_a()),
        execution_id="tg:first",
    )
    assert first.revision == 1

    class _CompetingWriterStore(ProcedureStateStore):
        """Advances the row from a SEPARATE transaction between READ and CAS.

        This makes the executor's CAS genuinely stale (its READ saw revision
        1, but the row is already at revision 2), so a DIFFERENT execution id
        must lose with a conflict rather than overwrite the winner.
        """

        async def save_cas(
            self, conversation_id, state, *, expected_revision, execution_id=None
        ):
            async with session_factory() as other:
                await ProcedureStateStore(other).save_cas(
                    conversation_id,
                    _state_b(),
                    expected_revision=expected_revision,
                    execution_id="tg:winner",
                )
                await other.commit()
            return await super().save_cas(
                conversation_id,
                state,
                expected_revision=expected_revision,
                execution_id=execution_id,
            )

    executor = SupportTurnExecutor(
        session_factory=session_factory, store_factory=_CompetingWriterStore
    )
    with pytest.raises(ProcedureStateConflict):
        await executor._execute_persisted(
            conversation_id=cid,
            process=lambda previous: SimpleNamespace(state=_state_b()),
            execution_id="tg:loser",
        )

    async with session_factory() as check:
        stored = await ProcedureStateStore(check).read(cid)
    assert stored.revision == 2
    assert stored.state == _state_b()
    assert stored.last_execution_id == "tg:winner"
