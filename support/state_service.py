"""
SupportStateService -- applies FSM transitions transactionally.

This is the ONLY place that mutates conversation state in response to
domain events. It uses PostgreSQL row-level locking (SELECT ... FOR UPDATE,
via SQLAlchemy) so that two distinct events racing on the same
conversation cannot both succeed.

Transaction discipline (approved design):
  * TX1 (claim) and TX2 (apply) are short and separate.
  * The LLM / retrieval call happens OUTSIDE any DB transaction; this
    service never calls the LLM.
  * Inside TX2: SELECT ... FOR UPDATE, re-read state, validate the
    transition, apply, commit. A transition that became invalid because a
    concurrent event changed the state is rejected as a no-op.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from support.domain import (
    ConversationMode,
    ConversationState,
    HandoffStatus,
    InvalidTransition,
    SupportEvent,
    SupportStatus,
    is_transition_allowed,
    next_state,
)
from support.models import Conversation


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class TransitionOutcome:
    applied: bool
    state: Optional[ConversationState]
    reason: str = ""


def _to_state(row: Conversation) -> ConversationState:
    return (
        ConversationMode(row.conversation_mode),
        SupportStatus(row.support_status),
        HandoffStatus(row.handoff_status) if row.handoff_status else None,
    )


class SupportStateService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def handle(
        self,
        event: SupportEvent,
        conversation_id: str,
    ) -> TransitionOutcome:
        """
        Apply ``event`` to the conversation identified by ``conversation_id``.

        ``START_NEW_SESSION`` must be handled by SessionService
        (state-independent, close+open); passing it here is an error.
        """
        # Lock the row for the duration of TX2.
        result = await self._session.execute(
            select(Conversation)
            .where(Conversation.conversation_id == conversation_id)
            .with_for_update()
        )
        row = result.scalar_one_or_none()
        if row is None:
            return TransitionOutcome(
                applied=False, state=None, reason="conversation_not_found"
            )

        current = _to_state(row)

        if not is_transition_allowed(current, event):
            # Another event already moved the conversation; reject as no-op.
            return TransitionOutcome(
                applied=False, state=current, reason="invalid_transition"
            )

        try:
            new_state = next_state(current, event)
        except InvalidTransition as exc:
            return TransitionOutcome(
                applied=False, state=current, reason=exc.reason or "invalid_transition"
            )

        row.conversation_mode = new_state[0].value
        row.support_status = new_state[1].value
        row.handoff_status = new_state[2].value if new_state[2] else None
        row.updated_at = _now()

        if event is SupportEvent.TRACKER_RESOLVED:
            row.resolution_requested_at = _now()

        await self._session.flush()
        return TransitionOutcome(applied=True, state=new_state)
