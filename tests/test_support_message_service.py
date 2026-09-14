"""MessageService integration tests (PostgreSQL)."""

from __future__ import annotations

import pytest

from support.domain import MessageRole
from support.message_service import MessageService
from support.session_service import SessionService


pytestmark = pytest.mark.asyncio


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
