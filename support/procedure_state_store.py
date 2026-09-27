"""
ProcedureStateStore -- revision-aware (compare-and-swap) persistence for
``support.procedure_state.ProcedureState``.

WHY A SECOND PERSISTENCE ADAPTER
--------------------------------
:class:`support.procedure_state_service.ProcedureStateService` (Slice 8) is
the lossless load/save adapter for the SAME two tables, but it is a
last-writer-wins upsert guarded only by a ``SELECT ... FOR UPDATE``: it has
no notion of "the state I read is the state I am replacing". Slice 9 needs
OPTIMISTIC concurrency control that (a) never holds a row lock across the
LLM/network processing window and (b) refuses to overwrite a state written
by a concurrent turn. That is a different write contract, so it lives in a
dedicated store rather than mutating the frozen Slice 8 adapter.

This module adds ONLY concurrency; it is a load/save ADAPTER. It never
reduces, never routes, never mutates the domain state, and is NOT imported
by any production runtime path (server.py, orchestrator.py, n8n, Tracker).

CONCURRENCY MODEL (compare-and-swap)
------------------------------------
The revision is a monotonic per-conversation counter:

  * cold start (no row) -> ``INSERT ... ON CONFLICT DO NOTHING`` with
    ``revision = 1``; a racing creator makes the INSERT a no-op and the
    loser raises :class:`ProcedureStateConflict` instead of overwriting.
  * existing row -> ``UPDATE ... WHERE conversation_id = :cid
    AND revision = :expected RETURNING revision``; zero rows means the
    expected revision no longer matches (stale read / concurrent writer)
    and the loser raises :class:`ProcedureStateConflict`.

A conditional UPDATE is itself atomic, so the CAS replaces the
``FOR UPDATE`` serialization: only the winner proceeds to replace the
attempt log, and every loser aborts before touching it. INVARIANT: every
writer must go through this store (``ProcedureStateService.save`` is not
wired to any runtime path), so no other writer can race the attempt log.

IDEMPOTENT REPLAY (Slice HL-2b)
-------------------------------
An OPTIONAL, OPAQUE ``execution_id`` may be supplied to :meth:`save_cas`.
When it is, the id is written ATOMICALLY with the state (same INSERT /
UPDATE -- no second transaction), and a repeat of the SAME id is an
idempotent no-op: NOTHING is written, no revision is bumped, the attempt log
is NOT rewritten, and the CURRENT PERSISTED state (loaded from the DB) is
returned with ``applied=False``. This lets the store tell an idempotent
replay apart from a genuine stale-writer conflict (the latter still raises
:class:`ProcedureStateConflict`). The store never interprets the id and
knows nothing about where it came from; callers that supply none keep the
previous behaviour exactly (the column stays NULL).

FK PRECONDITION: ``procedure_states.conversation_id`` references
``conversations.conversation_id``. This store therefore REQUIRES an existing
conversation row; it does NOT create one (conversation/channel lifecycle is
a runtime concern). A missing conversation surfaces as the underlying
``IntegrityError`` from the write, propagated unchanged.

Transaction-agnostic: it runs in the caller's ``AsyncSession`` and the
caller commits/rolls back. On conflict the store raises WITHOUT rolling
back, so the caller decides (mirrors the other ``*Service`` adapters).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from support.models import ProcedureStateAttemptRow, ProcedureStateRow
from support.observation import ProblemStatus, StepOutcome
from support.procedure_state import ProcedureState, StepAttempt


# The revision of a conversation's FIRST persisted state.
COLD_START_REVISION = 1


class ProcedureStateConflict(Exception):
    """The persisted state changed under us (optimistic CAS miss).

    Raised by :meth:`ProcedureStateStore.save_cas` when the expected
    revision no longer matches the stored row (a concurrent writer got
    there first) or when a concurrent cold-start created the row first.
    Nothing is written and nothing is overwritten; the caller decides how
    to react (no automatic retry here).
    """


@dataclass(frozen=True)
class PersistedProcedureState:
    """A persisted :class:`ProcedureState` plus its write revision.

    ``last_execution_id`` is the OPAQUE idempotency identity of the LAST
    write (see :meth:`ProcedureStateStore.save_cas`); it is ``None`` when no
    execution id was supplied and carries no channel/runtime meaning.
    """

    state: ProcedureState
    revision: int
    last_execution_id: Optional[str] = None


@dataclass(frozen=True)
class CasResult:
    """The outcome of ONE :meth:`ProcedureStateStore.save_cas` attempt.

    ``applied`` is the discriminator an ``int``-only return would drop:

      * ``applied=True``  -- THIS call wrote the row. ``revision`` is the
        NEW persisted revision and ``state`` is exactly what was persisted.
      * ``applied=False`` -- THIS call did NOT write: another write of the
        SAME ``execution_id`` already advanced the row. ``revision`` is the
        CURRENT persisted revision and ``state`` is the CURRENT persisted
        state LOADED FROM THE DB -- never the caller's candidate state.
    """

    revision: int
    applied: bool
    state: ProcedureState


class ProcedureStateStore:
    """Revision-aware load / CAS-save of one conversation's state."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------
    # READ
    # ------------------------------------------------------------

    async def read(
        self, conversation_id: str
    ) -> Optional[PersistedProcedureState]:
        """Return the persisted state + revision, or ``None`` if never saved.

        Read-only and lock-free: it takes no row lock, so the caller may
        safely close the read transaction before doing LLM/network work.
        """
        result = await self._session.execute(
            select(ProcedureStateRow).where(
                ProcedureStateRow.conversation_id == conversation_id
            )
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return PersistedProcedureState(
            state=await self._to_state(row),
            revision=row.revision,
            last_execution_id=row.last_execution_id,
        )

    # ------------------------------------------------------------
    # WRITE (compare-and-swap)
    # ------------------------------------------------------------

    async def save_cas(
        self,
        conversation_id: str,
        state: ProcedureState,
        *,
        expected_revision: Optional[int],
        execution_id: Optional[str] = None,
    ) -> CasResult:
        """Persist ``state`` iff it still matches ``expected_revision``.

        ``expected_revision is None`` means "I read no row; create the first
        one". Otherwise the row must still be at ``expected_revision``.

        ``execution_id`` is an OPTIONAL, OPAQUE idempotency identity. When it
        is supplied it is written atomically with the state, and a REPEAT of
        the same id is an idempotent no-op: NOTHING is written (no revision
        bump, no attempt-log rewrite) and the CURRENT persisted state is
        returned with ``applied=False``. When it is omitted (``None``) the
        call behaves exactly as before.

        Returns a :class:`CasResult`: ``applied=True`` carries the NEW
        revision and the state that was persisted; ``applied=False`` carries
        the CURRENT revision and the CURRENT persisted state loaded from the
        DB (NEVER the caller's candidate). Raises
        :class:`ProcedureStateConflict` on a genuine CAS miss (a DIFFERENT or
        absent execution id) and writes NOTHING (neither the state row nor
        the attempt log). Transaction-agnostic: the caller commits.
        """
        if expected_revision is None:
            result = await self._insert_cold_start(
                conversation_id, state, execution_id
            )
        else:
            result = await self._update_cas(
                conversation_id, state, expected_revision, execution_id
            )

        # Only a REAL CAS write replaces the attempt log; an idempotent hit
        # (``applied=False``) NEVER rewrites the attempt history.
        if result.applied:
            await self._replace_attempts(conversation_id, state)
        return result

    async def _insert_cold_start(
        self,
        conversation_id: str,
        state: ProcedureState,
        execution_id: Optional[str],
    ) -> CasResult:
        statement = (
            pg_insert(ProcedureStateRow)
            .values(
                conversation_id=conversation_id,
                issue=state.issue,
                procedure_id=state.procedure_id,
                procedure_version=state.procedure_version,
                last_recommended_step=state.last_recommended_step,
                problem_status=state.problem_status.value,
                pending_clarification=state.pending_clarification,
                clarification_topic=state.clarification_topic,
                schema_version=state.schema_version,
                revision=COLD_START_REVISION,
                last_execution_id=execution_id,
            )
            .on_conflict_do_nothing(index_elements=["conversation_id"])
            .returning(ProcedureStateRow.revision)
        )
        result = await self._session.execute(statement)
        inserted_revision = result.scalar_one_or_none()
        if inserted_revision is not None:
            return CasResult(
                revision=inserted_revision, applied=True, state=state
            )
        # The row already exists. If it was created by THIS SAME execution id
        # (a replayed / raced cold start) the write is an idempotent no-op;
        # otherwise it is a genuine concurrent cold start -> conflict. Never
        # fall back to an UPDATE (that would silently overwrite the winner).
        return await self._resolve_idempotent_or_conflict(
            conversation_id,
            execution_id,
            miss_message=(
                f"procedure state for {conversation_id!r} was created "
                "concurrently"
            ),
        )

    async def _update_cas(
        self,
        conversation_id: str,
        state: ProcedureState,
        expected_revision: int,
        execution_id: Optional[str],
    ) -> CasResult:
        # Idempotency short-circuit BEFORE the CAS: the executor re-reads the
        # (already advanced) revision, so ``expected_revision`` would match --
        # yet THIS execution must not bump the revision a second time.
        if execution_id is not None:
            row = await self._load_row(conversation_id)
            if row is not None and row.last_execution_id == execution_id:
                return CasResult(
                    revision=row.revision,
                    applied=False,
                    state=await self._to_state(row),
                )

        statement = (
            update(ProcedureStateRow)
            .where(
                ProcedureStateRow.conversation_id == conversation_id,
                ProcedureStateRow.revision == expected_revision,
            )
            .values(
                issue=state.issue,
                procedure_id=state.procedure_id,
                procedure_version=state.procedure_version,
                last_recommended_step=state.last_recommended_step,
                problem_status=state.problem_status.value,
                pending_clarification=state.pending_clarification,
                clarification_topic=state.clarification_topic,
                schema_version=state.schema_version,
                revision=ProcedureStateRow.revision + 1,
                last_execution_id=execution_id,
                updated_at=func.now(),
            )
            .returning(ProcedureStateRow.revision)
        )
        result = await self._session.execute(statement)
        new_revision = result.scalar_one_or_none()
        if new_revision is not None:
            return CasResult(revision=new_revision, applied=True, state=state)
        # Stale revision: a genuine concurrent writer (conflict) OR the SAME
        # execution id that won a race between our read and this CAS
        # (idempotent no-op).
        return await self._resolve_idempotent_or_conflict(
            conversation_id,
            execution_id,
            miss_message=(
                f"procedure state for {conversation_id!r} is no longer at "
                f"revision {expected_revision}"
            ),
        )

    async def _load_row(
        self, conversation_id: str
    ) -> Optional[ProcedureStateRow]:
        result = await self._session.execute(
            select(ProcedureStateRow).where(
                ProcedureStateRow.conversation_id == conversation_id
            )
        )
        return result.scalar_one_or_none()

    async def _resolve_idempotent_or_conflict(
        self,
        conversation_id: str,
        execution_id: Optional[str],
        *,
        miss_message: str,
    ) -> CasResult:
        """Tell an idempotent replay apart from a genuine CAS conflict.

        Consulted ONLY on a CAS-miss path. A row whose ``last_execution_id``
        equals this call's ``execution_id`` means the SAME execution already
        advanced the state -> idempotent no-op (return the CURRENT persisted
        state/revision, write nothing). Anything else (a different id, or no
        id at all) is a real conflict and raises
        :class:`ProcedureStateConflict`.
        """
        row = await self._load_row(conversation_id)
        if (
            row is not None
            and execution_id is not None
            and row.last_execution_id == execution_id
        ):
            return CasResult(
                revision=row.revision,
                applied=False,
                state=await self._to_state(row),
            )
        raise ProcedureStateConflict(miss_message)

    async def _replace_attempts(
        self, conversation_id: str, state: ProcedureState
    ) -> None:
        """Replace the attempt log wholesale (replace-as-a-set)."""
        await self._session.execute(
            delete(ProcedureStateAttemptRow).where(
                ProcedureStateAttemptRow.conversation_id == conversation_id
            )
        )
        if state.step_attempts:
            self._session.add_all(
                [
                    ProcedureStateAttemptRow(
                        conversation_id=conversation_id,
                        sequence=attempt.sequence,
                        step_id=attempt.step_id,
                        outcome=attempt.outcome.value,
                    )
                    for attempt in state.step_attempts
                ]
            )
        await self._session.flush()

    # ------------------------------------------------------------
    # PROJECTION
    # ------------------------------------------------------------

    async def _to_state(self, row: ProcedureStateRow) -> ProcedureState:
        result = await self._session.execute(
            select(ProcedureStateAttemptRow)
            .where(
                ProcedureStateAttemptRow.conversation_id == row.conversation_id
            )
            .order_by(ProcedureStateAttemptRow.sequence)
        )
        attempts = tuple(
            StepAttempt(
                step_id=attempt_row.step_id,
                outcome=StepOutcome(attempt_row.outcome),
                sequence=attempt_row.sequence,
            )
            for attempt_row in result.scalars().all()
        )
        return ProcedureState(
            issue=row.issue,
            procedure_id=row.procedure_id,
            procedure_version=row.procedure_version,
            last_recommended_step=row.last_recommended_step,
            step_attempts=attempts,
            problem_status=ProblemStatus(row.problem_status),
            pending_clarification=row.pending_clarification,
            clarification_topic=row.clarification_topic,
            schema_version=row.schema_version,
        )


__all__ = [
    "COLD_START_REVISION",
    "CasResult",
    "PersistedProcedureState",
    "ProcedureStateConflict",
    "ProcedureStateStore",
]
