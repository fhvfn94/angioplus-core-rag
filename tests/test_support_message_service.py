"""MessageService integration tests (PostgreSQL)."""

from __future__ import annotations

import pytest

from support.domain import MessageRole
from support.message_service import MessageService
from support.session_service import SessionService


pytestmark = [pytest.mark.asyncio, pytest.mark.db]


async def test_append_and_transcript_order(session):
    conv = await SessionService(session).create("telegram", "u1", "chat1")
    svc = MessageService(session)
    await svc.append(conv.conversation_id, MessageRole.USER, "привет")
    await svc.append(conv.conversation_id, MessageRole.ASSISTANT, "здравствуйте")
    await svc.append(conv.conversation_id, MessageRole.OPERATOR, "оператор здесь")
    await session.commit()

    transcript = await svc.build_transcript(conv.conversation_id)
    assert transcript == (
        "Пользователь: привет\n\n"
        "Ассистент: здравствуйте\n\n"
        "Оператор: оператор здесь"
    )


async def test_empty_transcript(session):
    conv = await SessionService(session).create("web", "u1", "s")
    await session.commit()
    assert await MessageService(session).build_transcript(conv.conversation_id) == ""


async def test_list_recent_for_conversation_is_bounded_and_isolated(session):
    conv1 = await SessionService(session).create("telegram", "u1", "chat1")
    conv2 = await SessionService(session).create("telegram", "u2", "chat2")

    svc = MessageService(session)

    await svc.append(conv1.conversation_id, MessageRole.USER, "старое сообщение")
    await svc.append(conv1.conversation_id, MessageRole.ASSISTANT, "старый ответ")
    await svc.append(conv1.conversation_id, MessageRole.USER, "чёрный экран")
    await svc.append(
        conv1.conversation_id,
        MessageRole.ASSISTANT,
        "перезапустите компьютер",
    )
    await svc.append(
        conv1.conversation_id,
        MessageRole.USER,
        "я уже два раза перезапускал",
    )

    # Сообщение другой conversation_id никогда не должно попасть в историю conv1.
    await svc.append(
        conv2.conversation_id,
        MessageRole.USER,
        "СЕКРЕТ ИЗ ДРУГОГО ДИАЛОГА",
    )

    await session.commit()

    rows = await svc.list_recent_for_conversation(
        conv1.conversation_id,
        limit=3,
    )

    assert [row.text for row in rows] == [
        "чёрный экран",
        "перезапустите компьютер",
        "я уже два раза перезапускал",
    ]

    assert all(row.conversation_id == conv1.conversation_id for row in rows)