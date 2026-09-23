"""
SessionService -- lifecycle of support conversations.

Responsibility: create / look up / close conversations, and atomically
close-and-open a new session (START_NEW_SESSION).

Does NOT: interpret messages, apply FSM events, write messages, or talk
to any network.

All methods take an ``AsyncSession`` so the caller controls transaction
boundaries.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from support.domain import (
    CaseAlreadyAttached,
    ConversationMode,
    HandoffStatus,
    SupportStatus,
)
from support.models import Conversation
from support.schemas import ConversationView


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _view(row: Conversation) -> ConversationView:
    return ConversationView(
        conversation_id=row.conversation_id,
        channel=row.channel,
        external_user_id=row.external_user_id,
        external_conversation_id=row.external_conversation_id,
        conversation_mode=row.conversation_mode,
        support_status=row.support_status,
        handoff_status=row.handoff_status,
        operator_provider=row.operator_provider,
        external_case_id=row.external_case_id,
    )


def new_conversation_id(channel: str, external_user_id: str) -> str:
    """Generate a business key, similar to the n8n pattern."""
    return f"{channel}-{external_user_id}-{int(_now().timestamp() * 1000)}"


class SessionService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------
    # READS
    # ------------------------------------------------------------

    async def get_active(
        self,
        channel: str,
        external_user_id: str,
        external_conversation_id: str,
        *,
        for_update: bool = False,
    ) -> Optional[ConversationView]:
        statement = select(Conversation).where(
            Conversation.channel == channel,
            Conversation.external_user_id == external_user_id,
            Conversation.external_conversation_id == external_conversation_id,
            Conversation.support_status == SupportStatus.ACTIVE.value,
        )
        if for_update:
            statement = statement.with_for_update()
        result = await self._session.execute(statement)
        row = result.scalar_one_or_none()
        return _view(row) if row else None

    async def get_by_conversation_id(
        self,
        conversation_id: str,
    ) -> Optional[ConversationView]:
        result = await self._session.execute(
            select(Conversation).where(
                Conversation.conversation_id == conversation_id
            )
        )
        row = result.scalar_one_or_none()
        return _view(row) if row else None

    async def get_by_external_case_id(
        self,
        external_case_id: str,
    ) -> Optional[ConversationView]:
        result = await self._session.execute(
            select(Conversation)
            .where(Conversation.external_case_id == external_case_id)
            .order_by(Conversation.id.desc())
        )
        row = result.scalars().first()
        return _view(row) if row else None

    async def list_resolution_timeouts(
        self,
        *,
        timeout_seconds: int,
        now: Optional[datetime] = None,
    ) -> list[ConversationView]:
        """Conversations waiting for a resolution confirmation past the SLA.

        Returns the active conversations whose ``handoff_status`` is
        ``waiting_resolution_confirmation`` and whose
        ``resolution_requested_at`` is older than ``now - timeout_seconds``.
        Read-only: the caller applies the ``RESOLUTION_TIMEOUT`` FSM event.
        """
        moment = now or _now()
        cutoff = moment - timedelta(seconds=timeout_seconds)
        result = await self._session.execute(
            select(Conversation).where(
                Conversation.support_status == SupportStatus.ACTIVE.value,
                Conversation.handoff_status
                == HandoffStatus.WAITING_RESOLUTION_CONFIRMATION.value,
                Conversation.resolution_requested_at.is_not(None),
                Conversation.resolution_requested_at < cutoff,
            )
        )
        return [_view(row) for row in result.scalars().all()]

    async def get_waiting_resolution(
        self,
        channel: str,
        external_user_id: str,
    ) -> Optional[ConversationView]:
        """The active conversation awaiting a resolution confirmation.

        PostgreSQL replacement for the legacy n8n Data Table lookup
        ``Find Resolution Conversation`` (match on channel + external_user_id
        + handoff_status = waiting_resolution_confirmation + active).
        Read-only; newest row wins if history ever contains duplicates.
        """
        result = await self._session.execute(
            select(Conversation)
            .where(
                Conversation.channel == channel,
                Conversation.external_user_id == external_user_id,
                Conversation.support_status == SupportStatus.ACTIVE.value,
                Conversation.handoff_status
                == HandoffStatus.WAITING_RESOLUTION_CONFIRMATION.value,
            )
            .order_by(Conversation.id.desc())
        )
        row = result.scalars().first()
        return _view(row) if row else None

    # ------------------------------------------------------------
    # WRITES
    # ------------------------------------------------------------

    async def create(
        self,
        channel: str,
        external_user_id: str,
        external_conversation_id: str,
        conversation_id: Optional[str] = None,
    ) -> ConversationView:
        """Create a fresh active AI-owned conversation."""
        row = Conversation(
            conversation_id=conversation_id
            or new_conversation_id(channel, external_user_id),
            channel=channel,
            external_user_id=external_user_id,
            external_conversation_id=external_conversation_id,
            conversation_mode=ConversationMode.AI.value,
            support_status=SupportStatus.ACTIVE.value,
            handoff_status=None,
        )
        self._session.add(row)
        await self._session.flush()
        return _view(row)

    async def get_or_create_active(
        self,
        channel: str,
        external_user_id: str,
        external_conversation_id: str,
    ) -> ConversationView:
        """Return the active conversation for the thread, or create one."""
        existing = await self.get_active(
            channel, external_user_id, external_conversation_id
        )
        if existing is not None:
            return existing
        return await self.create(
            channel, external_user_id, external_conversation_id
        )

    async def close(self, conversation_id: str) -> None:
        """Move a conversation to the single canonical closed state."""
        await self._session.execute(
            update(Conversation)
            .where(Conversation.conversation_id == conversation_id)
            .values(
                conversation_mode=ConversationMode.AI.value,
                support_status=SupportStatus.CLOSED.value,
                handoff_status=HandoffStatus.CLOSED.value,
                updated_at=_now(),
            )
        )

    async def close_and_open_new(
        self,
        channel: str,
        external_user_id: str,
        external_conversation_id: str,
    ) -> ConversationView:
        """
        START_NEW_SESSION, atomic and state-independent.

        In one transaction: close the current active conversation (if any)
        then open a fresh one. The partial unique active-session index forces
        the close-before-create ordering; doing it the other way fails loudly.
        """
        # Serialize starts for the same external thread, including the
        # first start when there is no row available to lock yet.
        lock_key = f"{channel}|{external_user_id}|{external_conversation_id}"
        await self._session.execute(
            select(func.pg_advisory_xact_lock(func.hashtextextended(lock_key, 0)))
        )

        active = await self.get_active(
            channel,
            external_user_id,
            external_conversation_id,
            for_update=True,
        )
        if active is not None:
            await self.close(active.conversation_id)
            # make sure the UPDATE is visible to the subsequent INSERT check
            await self._session.flush()

        return await self.create(
            channel,
            external_user_id,
            external_conversation_id,
            conversation_id=new_conversation_id(channel, external_user_id)
            + f"-{uuid.uuid4().hex[:6]}",
        )

    async def attach_case(
        self,
        conversation_id: str,
        external_case_id: str,
        operator_provider: Optional[str] = None,
    ) -> ConversationView:
        """
        Attach an external tracker case key to an existing conversation.

        This is the PostgreSQL source of truth for ``external_case_id`` --
        the backend equivalent of the legacy n8n Data Table write
        ``Save Tracker Case ID``. It ONLY mutates the case pointer: the FSM
        triple (conversation_mode / support_status / handoff_status) is left
        untouched, so no FSM invariant can be violated and no new
        conversation is ever created.

        Idempotency / retry safety:
          * unknown conversation           -> KeyError
          * same case already attached     -> no-op, returns current view
          * a DIFFERENT case already there -> CaseAlreadyAttached

        Runs in the caller's transaction; the caller commits.
        """
        result = await self._session.execute(
            select(Conversation)
            .where(Conversation.conversation_id == conversation_id)
            .with_for_update()
        )
        row = result.scalar_one_or_none()
        if row is None:
            raise KeyError(conversation_id)

        # Idempotent replays of the same escalation.
        if row.external_case_id == external_case_id:
            return _view(row)

        # Refuse to silently re-point an existing case to a new one.
        if row.external_case_id:
            raise CaseAlreadyAttached(conversation_id, row.external_case_id)

        row.external_case_id = external_case_id
        if operator_provider is not None:
            row.operator_provider = operator_provider
        row.updated_at = _now()
        await self._session.flush()
        return _view(row)
