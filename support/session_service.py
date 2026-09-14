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
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from support.domain import (
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
    ) -> Optional[ConversationView]:
        result = await self._session.execute(
            select(Conversation).where(
                Conversation.channel == channel,
                Conversation.external_user_id == external_user_id,
                Conversation.external_conversation_id == external_conversation_id,
                Conversation.support_status == SupportStatus.ACTIVE.value,
            )
        )
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
        active = await self.get_active(
            channel, external_user_id, external_conversation_id
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
