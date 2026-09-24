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
    """A persisted :class:`ProcedureState` plus its write revision."""

    state: ProcedureState
    revision: int


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
    ) -> int:
        """Persist ``state`` iff it still matches ``expected_revision``.

        ``expected_revision is None`` means "I read no row; create the first
        one". Otherwise the row must still be at ``expected_revision``.

        Returns the NEW revision (``1`` on cold start, else
        ``expected_revision + 1``). Raises :class:`ProcedureStateConflict`
        on a CAS miss and writes NOTHING (neither the state row nor the
        attempt log). Transaction-agnostic: the caller commits.
        """
        if expected_revision is None:
            new_revision = await self._insert_cold_start(
                conversation_id, state
            )
        else:
            new_revision = await self._update_cas(
                conversation_id, state, expected_revision
            )

        # Only the CAS winner reaches here, so the attempt log cannot race.
        await self._replace_attempts(conversation_id, state)
        return new_revision

    async def _insert_cold_start(
        self, conversation_id: str, state: ProcedureState
    ) -> int:
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
            )
            .on_conflict_do_nothing(index_elements=["conversation_id"])
            .returning(ProcedureStateRow.revision)
        )
        result = await self._session.execute(statement)
        inserted_revision = result.scalar_one_or_none()
        if inserted_revision is None:
            # A concurrent cold start created the row first. Never fall back
            # to an UPDATE (that would silently overwrite the winner).
            raise ProcedureStateConflict(
                f"procedure state for {conversation_id!r} was created "
                "concurrently"
            )
        return inserted_revision

    async def _update_cas(
        self,
        conversation_id: str,
        state: ProcedureState,
        expected_revision: int,
    ) -> int:
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
                updated_at=func.now(),
            )
            .returning(ProcedureStateRow.revision)
        )
        result = await self._session.execute(statement)
        new_revision = result.scalar_one_or_none()
        if new_revision is None:
            raise ProcedureStateConflict(
                f"procedure state for {conversation_id!r} is no longer at "
                f"revision {expected_revision}"
            )
        return new_revision

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
    "PersistedProcedureState",
    "ProcedureStateConflict",
    "ProcedureStateStore",
]
