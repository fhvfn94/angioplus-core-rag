"""IdempotencyService integration tests (PostgreSQL)."""

from __future__ import annotations

import pytest

from support.idempotency import (
    IdempotencyService,
    email_event_id,
    telegram_event_id,
    voice_event_id,
    web_event_id,
)
from support.message_service import MessageService
from support.session_service import SessionService
from support.domain import MessageRole


pytestmark = [pytest.mark.asyncio, pytest.mark.db]


@pytest.mark.filterwarnings(
    "ignore::pytest.PytestWarning:"
    "tests.test_support_idempotency"
)
def test_key_builders():
    assert telegram_event_id(123) == "tg:123"
    assert web_event_id("abc") == "web:abc"
    assert email_event_id("<m@x>") == "email:<m@x>"
    assert voice_event_id("call1", "turn2") == "voice:call1:turn2"


async def test_first_claim_wins_second_is_duplicate(session):
    idem = IdempotencyService(session)
    first = await idem.claim("tg:1", "telegram")
    await session.commit()
    assert first.claimed

    second = await idem.claim("tg:1", "telegram")
    assert not second.claimed
    assert second.outcome == "received"


async def test_duplicate_reconstructs_answer_and_metadata(session):
    svc = SessionService(session)
    conv = await svc.create("telegram", "u1", "chat1")
    msgs = MessageService(session)
    answer_id = await msgs.append(conv.conversation_id, MessageRole.ASSISTANT, "hi")
    await session.commit()

    idem = IdempotencyService(session)
    await idem.claim("tg:2", "telegram")
    await idem.finalize(
        inbound_event_id="tg:2",
        outcome="answered",
        conversation_id=conv.conversation_id,
        assistant_message_id=answer_id,
        decision="ANSWER",
        reason="RESOLVED_BY_L1",
        case_action="NONE",
    )
    await session.commit()

    dup = await idem.claim("tg:2", "telegram")
    assert not dup.claimed
    assert dup.prior is not None
    assert dup.prior.answer == "hi"
    assert dup.prior.decision == "ANSWER"
    assert dup.prior.reason == "RESOLVED_BY_L1"
    assert dup.prior.case_action == "NONE"


async def test_provider_event_claim_is_idempotent(session):
    idem = IdempotencyService(session)
    first = await idem.claim_provider_event("tracker:ANGIO-9:abc", "tracker_comment")
    await session.commit()
    assert first

    second = await idem.claim_provider_event("tracker:ANGIO-9:abc", "tracker_comment")
    assert not second
