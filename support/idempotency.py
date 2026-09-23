"""
IdempotencyService -- inbound-event and provider-event dedup.

Two distinct concerns, two distinct tables:

  * ``inbound_events``      -- deliveries from the user's channel
                              (Telegram update, web request, email, voice turn).
  * ``processed_events``    -- callbacks from external providers (Tracker).

The claim pattern relies on the primary key as the guard: the first INSERT
wins; a duplicate raises IntegrityError and is reported as not-claimed.

This service does NOT run the LLM or the FSM. It only claims / finalizes
and reads cached turn metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from support.models import InboundEvent, ProcessedEvent, Message
from support.schemas import ClaimResult, ProcessedTurnResult


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ============================================================
# PER-CHANNEL KEY BUILDERS
# ============================================================

def telegram_event_id(update_id: int | str) -> str:
    return f"tg:{update_id}"


def telegram_message_id(chat_id: int | str, message_id: int | str) -> str:
    return f"tg:{chat_id}:{message_id}"


def web_event_id(request_uuid: str) -> str:
    return f"web:{request_uuid}"


def email_event_id(message_id: str) -> str:
    return f"email:{message_id}"


def voice_event_id(call_id: str, turn_id: str) -> str:
    return f"voice:{call_id}:{turn_id}"


class IdempotencyService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------
    # INBOUND EVENTS (user -> backend)
    # ------------------------------------------------------------

    async def claim(
        self,
        inbound_event_id: str,
        channel: str,
    ) -> ClaimResult:
        """
        Try to claim an inbound event.

        Returns ClaimResult(claimed=True) for the first delivery.
        For a duplicate, returns claimed=False with the cached outcome and
        (if the turn finished) the cached decision metadata.

        A crash after this transaction commits intentionally leaves the row
        in ``received`` for Phase 1; stale-claim recovery is not implemented.
        """
        row = InboundEvent(
            inbound_event_id=inbound_event_id,
            channel=channel,
            outcome="received",
        )
        self._session.add(row)
        try:
            await self._session.flush()
        except IntegrityError:
            await self._session.rollback()
            return await self._load_duplicate(inbound_event_id)

        return ClaimResult(claimed=True, outcome="received")

    async def _load_duplicate(
        self,
        inbound_event_id: str,
    ) -> ClaimResult:
        result = await self._session.execute(
            select(InboundEvent).where(
                InboundEvent.inbound_event_id == inbound_event_id
            )
        )
        row = result.scalar_one_or_none()
        if row is None:
            # Extremely unlikely (row deleted between flush and select).
            return ClaimResult(claimed=False, outcome="duplicate", prior=None)

        prior: Optional[ProcessedTurnResult] = None
        if row.assistant_message_id is not None:
            text_result = await self._session.execute(
                select(Message.text).where(Message.id == row.assistant_message_id)
            )
            answer = text_result.scalar_one_or_none()
            prior = ProcessedTurnResult(
                answer=answer,
                decision=row.decision,
                reason=row.reason,
                case_action=row.case_action,
                conversation_id=row.conversation_id,
            )
        elif row.conversation_id is not None:
            prior = ProcessedTurnResult(
                answer=None,
                decision=row.decision,
                reason=row.reason,
                case_action=row.case_action,
                conversation_id=row.conversation_id,
            )

        return ClaimResult(claimed=False, outcome=row.outcome, prior=prior)

    async def finalize(
        self,
        inbound_event_id: str,
        outcome: str,
        conversation_id: Optional[str],
        assistant_message_id: Optional[int],
        decision: Optional[str] = None,
        reason: Optional[str] = None,
        case_action: Optional[str] = None,
    ) -> None:
        """Store the turn result metadata (answer text stays in messages)."""
        result = await self._session.execute(
            select(InboundEvent).where(
                InboundEvent.inbound_event_id == inbound_event_id
            )
        )
        row = result.scalar_one_or_none()
        if row is None:
            return
        row.outcome = outcome
        row.conversation_id = conversation_id
        row.assistant_message_id = assistant_message_id
        row.decision = decision
        row.reason = reason
        row.case_action = case_action
        row.processed_at = _now()
        await self._session.flush()

    # ------------------------------------------------------------
    # PROVIDER EVENTS (external callbacks, e.g. Tracker)
    # ------------------------------------------------------------

    async def claim_provider_event(
        self,
        event_key: str,
        source: str,
        external_case_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> bool:
        """
        Claim an external provider event. Returns True if newly claimed,
        False if already processed.
        """
        row = ProcessedEvent(
            event_key=event_key,
            source=source,
            external_case_id=external_case_id,
            conversation_id=conversation_id,
        )
        self._session.add(row)
        try:
            await self._session.flush()
        except IntegrityError:
            await self._session.rollback()
            return False
        return True
