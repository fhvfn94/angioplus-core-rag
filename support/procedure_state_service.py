"""
ProcedureStateService -- persistence for ``support.procedure_state``.

The pure reducer keeps troubleshooting progress in an immutable
:class:`~support.procedure_state.ProcedureState` (a frozen dataclass with a
tuple of :class:`~support.procedure_state.StepAttempt`). That value object is
not itself persistable, so this service projects it onto two additive tables:

  * ``procedure_states``          -- the CURRENT state, one row per conversation
  * ``procedure_state_attempts``  -- the append-only attempt log

It is a load/save ADAPTER only: it never reduces, never routes, never mutates
the domain state, and it is NOT imported by any production runtime path yet
(server.py, orchestrator.py, n8n, Tracker). It mirrors the other ``*Service``
adapters (``SessionService``, ``MessageService``, ...): the service takes an
``AsyncSession``, runs in the caller's transaction, and the caller commits.

Why not name this ``SupportStateService``? That name is already taken by the
conversation-FSM read model in :mod:`support.state_service` (the
mode/status/handoff triple). This service persists the *orthogonal* procedure
axis (see the State/Router split in :mod:`support.procedure_state`), so it is
named after the value object it stores.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from support.models import ProcedureStateAttemptRow, ProcedureStateRow
from support.observation import ProblemStatus, StepOutcome
from support.procedure_state import ProcedureState, StepAttempt


class ProcedureStateService:
    """Load / persist the :class:`ProcedureState` of one conversation."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------
    # READS
    # ------------------------------------------------------------

    async def handle(
        self,
        conversation_id: str,
        *,
        for_update: bool = False,
    ) -> Optional[ProcedureState]:
        """Return the persisted state, or ``None`` if it was never saved.

        Read-only. ``for_update=True`` takes a row lock for a caller that is
        about to write within the same transaction (mirrors
        :meth:`support.session_service.SessionService.get_active`).
        """
        row = await self._state_row(conversation_id, for_update=for_update)
        if row is None:
            return None
        return await self._to_state(row)

    # ------------------------------------------------------------
    # WRITES
    # ------------------------------------------------------------

    async def save(self, conversation_id: str, state: ProcedureState) -> None:
        """Upsert the current state and replace its attempt log, in place.

        Transaction-agnostic: the caller commits. A ``SELECT ... FOR UPDATE``
        serializes concurrent writers for the same conversation, so the
        replace-as-a-set of the attempt log is race-safe within the DB.
        """
        row = await self._state_row(conversation_id, for_update=True)
        if row is None:
            row = ProcedureStateRow(conversation_id=conversation_id)
            self._session.add(row)

        row.issue = state.issue
        row.procedure_id = state.procedure_id
        row.procedure_version = state.procedure_version
        row.last_recommended_step = state.last_recommended_step
        row.problem_status = state.problem_status.value
        row.pending_clarification = state.pending_clarification
        row.clarification_topic = state.clarification_topic
        row.schema_version = state.schema_version

        # Flush the parent first so the attempts' FK is satisfiable, whether
        # the state row is brand new or already present.
        await self._session.flush()

        await self._session.execute(
            delete(ProcedureStateAttemptRow).where(
                ProcedureStateAttemptRow.conversation_id == conversation_id
            )
        )
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
    # INTERNALS
    # ------------------------------------------------------------

    async def _state_row(
        self,
        conversation_id: str,
        *,
        for_update: bool,
    ) -> Optional[ProcedureStateRow]:
        statement = select(ProcedureStateRow).where(
            ProcedureStateRow.conversation_id == conversation_id
        )
        if for_update:
            statement = statement.with_for_update()
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

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
