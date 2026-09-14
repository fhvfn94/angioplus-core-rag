"""
MessageService -- append-only message log and transcript building.

Responsibility: append messages and read a conversation transcript.

Does NOT: change conversation state, perform FSM logic, or touch the network.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from support.models import Message
from support.domain import MessageRole


class MessageService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def append(
        self,
        conversation_id: str,
        role: MessageRole | str,
        text: str,
        external_user_id: Optional[str] = None,
    ) -> int:
        """Append a message and return its id."""
        role_value = role.value if isinstance(role, MessageRole) else str(role)
        row = Message(
            conversation_id=conversation_id,
            external_user_id=external_user_id,
            role=role_value,
            text=text,
        )
        self._session.add(row)
        await self._session.flush()
        return row.id

    async def get_text(self, message_id: int) -> Optional[str]:
        result = await self._session.execute(
            select(Message.text).where(Message.id == message_id)
        )
        return result.scalar_one_or_none()

    async def list_for_conversation(
        self,
        conversation_id: str,
    ) -> list[Message]:
        result = await self._session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at, Message.id)
        )
        return list(result.scalars().all())

    async def build_transcript(self, conversation_id: str) -> str:
        """
        Build a human-readable transcript.

        Ports the intent of the n8n ``Build Conversation Transcript`` node,
        deterministically and testably.
        """
        rows = await self.list_for_conversation(conversation_id)
        if not rows:
            return ""

        labels = {
            MessageRole.USER.value: "Пользователь",
            MessageRole.ASSISTANT.value: "Ассистент",
            MessageRole.OPERATOR.value: "Оператор",
        }

        lines: list[str] = []
        for row in rows:
            label = labels.get(row.role, row.role)
            text = (row.text or "").strip()
            lines.append(f"{label}: {text}")
        return "\n\n".join(lines)
