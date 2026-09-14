"""SupportStateService integration tests (PostgreSQL), incl. concurrency."""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from support.domain import (
    ConversationMode,
    HandoffStatus,
    SupportEvent,
    SupportStatus,
)
from support.models import Conversation
from support.session_service import SessionService
from support.state_service import SupportStateService


pytestmark = pytest.mark.asyncio


async def _make_conversation(session, channel="telegram", uid="u1", cid="chat1"):
    svc = SessionService(session)
    conv = await svc.create(channel, uid, cid)
    await session.commit()
    return conv


async def test_escalate_new_moves_to_waiting_operator(session):
    conv = await _make_conversation(session)
    svc = SupportStateService(session)
    outcome = await svc.handle(SupportEvent.ESCALATE_NEW, conv.conversation_id)
    await session.commit()

    assert outcome.applied
    assert outcome.state == (
        ConversationMode.HUMAN,
        SupportStatus.ACTIVE,
        HandoffStatus.WAITING_OPERATOR,
    )


async def test_invalid_event_is_rejected_noop(session):
    conv = await _make_conversation(session)
    svc = SupportStateService(session)
    # USER_MESSAGE_HUMAN is not valid from ai/active/None
    outcome = await svc.handle(
        SupportEvent.USER_MESSAGE_HUMAN, conv.conversation_id
    )
    assert not outcome.applied
    assert outcome.reason == "invalid_transition"


async def test_waiting_user_round_trip(session):
    conv = await _make_conversation(session)
    svc = SupportStateService(session)

    await svc.handle(SupportEvent.ESCALATE_NEW, conv.conversation_id)
    await session.commit()

    await svc.handle(SupportEvent.TRACKER_NEED_INFO, conv.conversation_id)
    await session.commit()
    row = (
        await session.execute(
            select(Conversation).where(
                Conversation.conversation_id == conv.conversation_id
            )
        )
    ).scalar_one()
    assert row.conversation_mode == ConversationMode.AI.value
    assert row.handoff_status == HandoffStatus.WAITING_USER.value

    outcome = await svc.handle(
        SupportEvent.ESCALATE_RESUME, conv.conversation_id
    )
    await session.commit()
    assert outcome.state == (
        ConversationMode.HUMAN,
        SupportStatus.ACTIVE,
        HandoffStatus.WAITING_OPERATOR,
    )


async def test_resume_keeps_same_external_case_id(session):
    conv = await _make_conversation(session)
    result = await session.execute(
        select(Conversation).where(
            Conversation.conversation_id == conv.conversation_id
        )
    )
    row = result.scalar_one()
    row.external_case_id = "CASE-123"
    await session.commit()

    svc = SupportStateService(session)
    await svc.handle(SupportEvent.ESCALATE_NEW, conv.conversation_id)
    await session.commit()
    await svc.handle(SupportEvent.TRACKER_NEED_INFO, conv.conversation_id)
    await session.commit()
    outcome = await svc.handle(
        SupportEvent.ESCALATE_RESUME, conv.conversation_id
    )
    await session.commit()

    assert outcome.applied
    resumed = (
        await session.execute(
            select(Conversation).where(
                Conversation.conversation_id == conv.conversation_id
            )
        )
    ).scalar_one()
    assert resumed.external_case_id == "CASE-123"


async def test_concurrent_events_only_one_wins(engine, session_factory):
    """
    Two competing state-changing events from the same state must not both
    succeed. FOR UPDATE serializes them; the loser is a no-op.
    """
    # Seed a conversation in waiting_resolution_confirmation.
    from support.db import Base  # noqa: F401

    async with session_factory() as setup:
        svc = SessionService(setup)
        conv = await svc.create("telegram", "u1", "chat1")
        await setup.commit()
        state_svc = SupportStateService(setup)
        await state_svc.handle(SupportEvent.ESCALATE_NEW, conv.conversation_id)
        await state_svc.handle(SupportEvent.TRACKER_RESOLVED, conv.conversation_id)
        await setup.commit()
        conversation_id = conv.conversation_id

    results: list[bool] = []

    async def run_event(event):
        async with session_factory() as s:
            svc = SupportStateService(s)
            outcome = await svc.handle(event, conversation_id)
            await s.commit()
            results.append(outcome.applied)

    # USER_REJECTED_RESOLVED -> back to operator; RESOLUTION_TIMEOUT -> closed
    await asyncio.gather(
        run_event(SupportEvent.USER_REJECTED_RESOLVED),
        run_event(SupportEvent.RESOLUTION_TIMEOUT),
    )

    assert results.count(True) == 1
    assert results.count(False) == 1

    async with session_factory() as check:
        row = (
            await check.execute(
                select(Conversation).where(
                    Conversation.conversation_id == conversation_id
                )
            )
        ).scalar_one()
        # Whatever won, the state must be one of the two valid successors.
        assert (row.conversation_mode, row.support_status, row.handoff_status) in (
            (ConversationMode.HUMAN.value, SupportStatus.ACTIVE.value,
             HandoffStatus.WAITING_OPERATOR.value),
            (ConversationMode.AI.value, SupportStatus.CLOSED.value,
             HandoffStatus.CLOSED.value),
        )


async def test_forbidden_tristate_rejected_by_check(session):
    conv = await _make_conversation(session)
    result = await session.execute(
        select(Conversation).where(
            Conversation.conversation_id == conv.conversation_id
        )
    )
    row = result.scalar_one()
    # (ai, active, waiting_operator) is forbidden.
    row.handoff_status = HandoffStatus.WAITING_OPERATOR.value
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()


async def test_human_closed_closed_is_forbidden(session):
    conv = await _make_conversation(session)
    result = await session.execute(
        select(Conversation).where(
            Conversation.conversation_id == conv.conversation_id
        )
    )
    row = result.scalar_one()
    row.conversation_mode = ConversationMode.HUMAN.value
    row.support_status = SupportStatus.CLOSED.value
    row.handoff_status = HandoffStatus.CLOSED.value
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()
