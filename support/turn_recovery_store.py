"""
TurnRecoveryStore -- the NARROW persistence of the durable turn recovery
record (HL-2c.3a).

One responsibility: read / write ONE row of ``turn_recovery`` keyed by the
opaque ``execution_id``. It is deliberately NOT a generic repository, event
store or replay service, and it is COMPLETELY SEPARATE from
:class:`support.procedure_state_store.ProcedureStateStore` (which owns
``ProcedureState`` + CAS + ``last_execution_id`` only and knows nothing about
recovery or ``ConversationState``).

ATOMICITY
---------
``save`` writes on the CALLER'S session and does NOT commit: the executor
calls it inside the SAME transaction as the ProcedureState CAS, so the CAS and
the recovery record commit or roll back together. There is no crash window in
which the procedure state advanced but the recovery record for that
``execution_id`` is missing, and if this write fails the whole transaction --
including the CAS -- rolls back.

INTEGRITY
---------
``read`` distinguishes the two failure modes:

  * ABSENT -- no row for that ``execution_id`` (legitimate: the execution did
    not record recovery, e.g. a legacy caller that supplied no projection
    snapshot, or a caller not using idempotency at all). Represented
    EXPLICITLY as ``None``.
  * CORRUPT -- a row exists but is partial / unparseable, or belongs to a
    DIFFERENT conversation. Raises :class:`TurnRecoveryCorruption`: never
    silently coerced to ``None``.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from support.models import TurnRecoveryRow
from support.turn_recovery import TurnRecoveryCorruption, TurnRecoveryPayload


class TurnRecoveryStore:
    """Read / write the durable recovery record for one ``execution_id``."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, payload: TurnRecoveryPayload) -> None:
        """INSERT the recovery record on the caller's session (NO commit).

        The caller owns the transaction boundary. A duplicate ``execution_id``
        (which the executor's ``applied`` guard makes impossible) fails loudly
        at the database and rolls the transaction back, rather than silently
        rewriting an existing winner's record.
        """
        await self._session.execute(
            insert(TurnRecoveryRow).values(
                execution_id=payload.execution_id,
                conversation_id=payload.conversation_id,
                text=payload.text,
                action=payload.action,
                escalate_reason=payload.escalate_reason,
                clarify_reason=payload.clarify_reason,
                projection_mode=payload.projection_mode,
                projection_support_status=payload.projection_support_status,
                projection_handoff_status=payload.projection_handoff_status,
            )
        )

    async def read(
        self, *, execution_id: str, conversation_id: str
    ) -> Optional[TurnRecoveryPayload]:
        """Load the recovery record for ``execution_id`` (or ``None``).

        ``None`` means ABSENT (legitimate). A record that exists but does not
        belong to ``conversation_id``, or that is partial / unparseable, raises
        :class:`TurnRecoveryCorruption` -- recovery can never be served for
        another conversation's turn, and damage is never hidden.
        """
        row = (
            await self._session.execute(
                select(TurnRecoveryRow).where(
                    TurnRecoveryRow.execution_id == execution_id
                )
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        if row.conversation_id != conversation_id:
            raise TurnRecoveryCorruption(
                f"recovery record {execution_id!r} belongs to conversation "
                f"{row.conversation_id!r}, not {conversation_id!r}."
            )
        try:
            payload = TurnRecoveryPayload(
                execution_id=row.execution_id,
                conversation_id=row.conversation_id,
                text=row.text,
                action=row.action,
                projection_mode=row.projection_mode,
                projection_support_status=row.projection_support_status,
                escalate_reason=row.escalate_reason,
                clarify_reason=row.clarify_reason,
                projection_handoff_status=row.projection_handoff_status,
            )
            # Validate the closed vocabularies NOW (a corrupt row must fail at
            # the persistence boundary, not deep inside a later projection).
            payload.conversation_state()
            payload.routing_decision()
        except TurnRecoveryCorruption as exc:
            raise TurnRecoveryCorruption(
                f"durable recovery record {execution_id!r} is partial or "
                f"malformed: {exc}"
            ) from exc
        return payload


__all__ = ["TurnRecoveryStore"]
