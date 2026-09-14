"""SessionService integration tests (PostgreSQL)."""

from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError

from support.domain import ConversationMode, HandoffStatus, SupportStatus
from support.models import Conversation
from support.session_service import SessionService


pytestmark = pytest.mark.asyncio


async def test_create_and_get_active(session):
    svc = SessionService(session)
    conv = await svc.create("telegram", "u1", "chat1")
    await session.commit()

    assert conv.conversation_mode == ConversationMode.AI.value
    assert conv.support_status == SupportStatus.ACTIVE.value
    assert conv.handoff_status is None

    found = await svc.get_active("telegram", "u1", "chat1")
    assert found is not None
    assert found.conversation_id == conv.conversation_id


async def test_partial_unique_rejects_second_active_same_thread(session):
    svc = SessionService(session)
    await svc.create("telegram", "u1", "chat1")
    await session.commit()

    with pytest.raises(IntegrityError):
        await svc.create("telegram", "u1", "chat1")
    await session.rollback()


async def test_same_user_multiple_threads_allowed(session):
    svc = SessionService(session)
    a = await svc.create("email", "u1", "thread-A")
    b = await svc.create("email", "u1", "thread-B")
    await session.commit()
    assert a.conversation_id != b.conversation_id

    found_a = await svc.get_active("email", "u1", "thread-A")
    found_b = await svc.get_active("email", "u1", "thread-B")
    assert found_a is not None and found_b is not None


async def test_close_then_reopen_allowed(session):
    svc = SessionService(session)
    conv = await svc.create("web", "u1", "sess")
    await session.commit()
    await svc.close(conv.conversation_id)
    await session.commit()

    reopened = await svc.create("web", "u1", "sess")
    await session.commit()
    assert reopened.conversation_id != conv.conversation_id


async def test_close_and_open_new_is_atomic(session):
    svc = SessionService(session)
    first = await svc.create("telegram", "u1", "chat1")
    await session.commit()

    new = await svc.close_and_open_new("telegram", "u1", "chat1")
    await session.commit()

    assert new.conversation_id != first.conversation_id

    old = await svc.get_by_conversation_id(first.conversation_id)
    assert old is not None
    assert old.support_status == SupportStatus.CLOSED.value
    assert old.handoff_status == HandoffStatus.CLOSED.value

    active = await svc.get_active("telegram", "u1", "chat1")
    assert active is not None
    assert active.conversation_id == new.conversation_id


async def test_start_new_session_from_human_state(session):
    svc = SessionService(session)
    conv = await svc.create("telegram", "u1", "chat1")
    await session.commit()
    # force a human-owned active state directly
    from sqlalchemy import select

    result = await session.execute(
        select(Conversation).where(
            Conversation.conversation_id == conv.conversation_id
        )
    )
    row = result.scalar_one()
    row.conversation_mode = ConversationMode.HUMAN.value
    row.handoff_status = HandoffStatus.WAITING_OPERATOR.value
    await session.commit()

    new = await svc.close_and_open_new("telegram", "u1", "chat1")
    await session.commit()
    assert new.support_status == SupportStatus.ACTIVE.value
    assert new.conversation_mode == ConversationMode.AI.value


@pytest.mark.parametrize(
    "mode,handoff",
    [
        (ConversationMode.AI.value, None),
        (ConversationMode.AI.value, HandoffStatus.WAITING_USER.value),
        (ConversationMode.HUMAN.value, HandoffStatus.WAITING_OPERATOR.value),
        (
            ConversationMode.HUMAN.value,
            HandoffStatus.WAITING_RESOLUTION_CONFIRMATION.value,
        ),
    ],
)
async def test_start_new_session_from_every_active_state(session, mode, handoff):
    svc = SessionService(session)
    conv = await svc.create("telegram", "u1", f"chat-{mode}-{handoff}")
    await session.commit()

    from sqlalchemy import select

    row = (
        await session.execute(
            select(Conversation).where(
                Conversation.conversation_id == conv.conversation_id
            )
        )
    ).scalar_one()
    row.conversation_mode = mode
    row.handoff_status = handoff
    await session.commit()

    new = await svc.close_and_open_new(
        "telegram", "u1", f"chat-{mode}-{handoff}"
    )
    await session.commit()

    assert new.conversation_mode == ConversationMode.AI.value
    assert new.support_status == SupportStatus.ACTIVE.value
    assert new.handoff_status is None


async def test_close_and_open_new_preserves_no_case_on_new_session(session):
    svc = SessionService(session)
    conv = await svc.create("telegram", "u1", "chat-case")
    await session.commit()

    from sqlalchemy import select

    row = (
        await session.execute(
            select(Conversation).where(
                Conversation.conversation_id == conv.conversation_id
            )
        )
    ).scalar_one()
    row.external_case_id = "CASE-123"
    row.conversation_mode = ConversationMode.AI.value
    row.handoff_status = HandoffStatus.WAITING_USER.value
    await session.commit()

    new = await svc.close_and_open_new("telegram", "u1", "chat-case")
    await session.commit()

    old = await svc.get_by_external_case_id("CASE-123")
    assert old is not None
    assert old.conversation_id == conv.conversation_id
    assert new.external_case_id is None
