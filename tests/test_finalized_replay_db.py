# -*- coding: utf-8 -*-
"""
HL-2c.3b.1 -- durable finalized replay (DB-BACKED proofs).

REAL PostgreSQL (no SQLite, no in-memory double): these tests prove the
DURABLE contract the runtime will rely on:

  * ``IdempotencyService.finalize`` persists the replay snapshot ATOMICALLY
    (the SAME transaction as the outcome), so a finalized row is never
    missing its snapshot;
  * ``IdempotencyService.load_finalized_replay`` returns the ACTUAL stored
    data and ``reproduce_finalized_response`` rebuilds the EXACT original
    body -- ``retrieval_used`` included, NOT hardcoded;
  * the answer text is loaded from its single durable source (``messages``),
    never duplicated into ``inbound_events``;
  * the replay reads ONLY; it creates no message and mutates no FSM state, and
    it does NOT re-derive ``external_case_id`` from the CURRENT conversation;
  * ABSENT and CORRUPT fail closed differently, and a LEGACY finalized row
    (snapshot never persisted) can never replay silently.

Every test is marked ``db`` and therefore goes through the fail-closed
``tests/conftest.py`` guard; run with a dedicated ``TEST_DATABASE_URL``:

  postgresql+asyncpg://angioplus_test:<pw>@127.0.0.1:5432/angioplus_test

e.g. (PowerShell)::

    $env:TEST_DATABASE_URL = "<that DSN>"
    python -m pytest tests/test_finalized_replay_db.py -q
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select
from sqlalchemy import text as sql_text

from support.domain import MessageRole
from support.finalized_replay import (
    FinalizedReplayCorruption,
    FinalizedReplayUnavailable,
    reproduce_finalized_response,
)
from support.idempotency import IdempotencyService
from support.message_service import MessageService
from support.models import Conversation, Message
from support.session_service import SessionService


# ============================================================
# HELPERS
# ============================================================

async def _conversation(session_factory, uid="u1", cid="chat1") -> str:
    async with session_factory() as session:
        conv = await SessionService(session).create("telegram", uid, cid)
        await session.commit()
        return conv.conversation_id


async def _assistant_message(session_factory, conversation_id, text) -> int:
    async with session_factory() as session:
        message_id = await MessageService(session).append(
            conversation_id, MessageRole.ASSISTANT, text
        )
        await session.commit()
        return message_id


async def _claim(session_factory, event_id) -> None:
    async with session_factory() as session:
        await IdempotencyService(session).claim(event_id, "telegram")
        await session.commit()


async def _finalize(
    session_factory,
    *,
    event_id,
    outcome,
    conversation_id,
    assistant_message_id,
    decision,
    reason,
    case_action,
    retrieval_used,
    conversation_mode,
    support_status,
    handoff_status=None,
    external_case_id=None,
) -> None:
    async with session_factory() as session:
        await IdempotencyService(session).finalize(
            event_id,
            outcome,
            conversation_id,
            assistant_message_id,
            decision=decision,
            reason=reason,
            case_action=case_action,
            retrieval_used=retrieval_used,
            conversation_mode=conversation_mode,
            support_status=support_status,
            handoff_status=handoff_status,
            external_case_id=external_case_id,
        )
        await session.commit()


async def _load(session_factory, event_id):
    async with session_factory() as session:
        return await IdempotencyService(session).load_finalized_replay(event_id)


async def _count_messages(session_factory, conversation_id) -> int:
    async with session_factory() as session:
        return int(
            (
                await session.execute(
                    select(func.count())
                    .select_from(Message)
                    .where(Message.conversation_id == conversation_id)
                )
            ).scalar_one()
        )


async def _conversation_triple(session_factory, conversation_id):
    async with session_factory() as session:
        row = (
            await session.execute(
                select(Conversation).where(
                    Conversation.conversation_id == conversation_id
                )
            )
        ).scalar_one()
        return (
            row.conversation_mode,
            row.support_status,
            row.handoff_status,
            row.external_case_id,
        )


def _answered_snapshot(*, retrieval_used, external_case_id):
    return dict(
        retrieval_used=retrieval_used,
        conversation_mode="ai",
        support_status="active",
        handoff_status=None,
        external_case_id=external_case_id,
    )


# ============================================================
# 1. FINALIZE PERSISTS THE SNAPSHOT; LOAD REPRODUCES IT EXACTLY
# ============================================================

@pytest.mark.asyncio
@pytest.mark.db
async def test_db_finalized_replay_reproduces_the_exact_body(session_factory):
    cid = await _conversation(session_factory)
    mid = await _assistant_message(session_factory, cid, "Reseat the cable.")

    await _claim(session_factory, "tg:1")
    await _finalize(
        session_factory,
        event_id="tg:1",
        outcome="answered",
        conversation_id=cid,
        assistant_message_id=mid,
        decision="ANSWER",
        reason="RESOLVED_BY_L1",
        case_action="NONE",
        **_answered_snapshot(retrieval_used=True, external_case_id=None),
    )

    snapshot = await _load(session_factory, "tg:1")
    response = reproduce_finalized_response(snapshot)

    assert response.ok is True
    assert response.answer == "Reseat the cable."
    assert response.retrieval_used is True  # NOT hardcoded to False
    assert response.decision == "ANSWER"
    assert response.reason == "RESOLVED_BY_L1"
    assert response.conversation_id == cid
    assert response.conversation_mode == "ai"
    assert response.support_status == "active"
    assert response.handoff_status is None
    assert response.case_action == "NONE"
    assert response.external_case_id is None
    assert response.deduplicated is True


@pytest.mark.asyncio
@pytest.mark.db
async def test_db_retrieval_used_false_is_preserved(session_factory):
    cid = await _conversation(session_factory)
    mid = await _assistant_message(session_factory, cid, "No search was used.")

    await _claim(session_factory, "tg:2")
    await _finalize(
        session_factory,
        event_id="tg:2",
        outcome="answered",
        conversation_id=cid,
        assistant_message_id=mid,
        decision="ANSWER",
        reason="RESOLVED_BY_L1",
        case_action="NONE",
        **_answered_snapshot(retrieval_used=False, external_case_id=None),
    )

    response = reproduce_finalized_response(await _load(session_factory, "tg:2"))
    assert response.retrieval_used is False


@pytest.mark.asyncio
@pytest.mark.db
async def test_db_legacy_finalize_leaves_the_snapshot_unset(session_factory):
    """A caller that passes no snapshot is completely unaffected (NULLs)."""
    cid = await _conversation(session_factory)
    mid = await _assistant_message(session_factory, cid, "Legacy answer.")

    await _claim(session_factory, "tg:legacy")
    async with session_factory() as session:
        await IdempotencyService(session).finalize(
            "tg:legacy",
            "answered",
            cid,
            mid,
            decision="ANSWER",
            reason="RESOLVED_BY_L1",
            case_action="NONE",
        )
        await session.commit()

    async with session_factory() as session:
        row = (
            await session.execute(
                sql_text(
                    "SELECT retrieval_used, conversation_mode, support_status, "
                    "handoff_status, external_case_id "
                    "FROM inbound_events WHERE inbound_event_id = 'tg:legacy'"
                )
            )
        ).one()
    assert tuple(row) == (None, None, None, None, None)

    # ...and replay fails CLOSED rather than guessing a lossy body.
    with pytest.raises(FinalizedReplayCorruption):
        await _load(session_factory, "tg:legacy")


# ============================================================
# 2. THE ANSWER LIVES IN MESSAGES (NOT DUPLICATED)
# ============================================================

@pytest.mark.asyncio
@pytest.mark.db
async def test_db_answer_is_loaded_from_the_message_not_duplicated(
    session_factory,
):
    cid = await _conversation(session_factory)
    mid = await _assistant_message(session_factory, cid, "Original text.")

    await _claim(session_factory, "tg:3")
    await _finalize(
        session_factory,
        event_id="tg:3",
        outcome="answered",
        conversation_id=cid,
        assistant_message_id=mid,
        decision="ANSWER",
        reason="RESOLVED_BY_L1",
        case_action="NONE",
        **_answered_snapshot(retrieval_used=False, external_case_id=None),
    )

    # There is NO ``answer`` column on ``inbound_events``: the text is linked,
    # not copied.
    async with session_factory() as session:
        columns = set(
            (
                await session.execute(
                    sql_text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'inbound_events'"
                    )
                )
            )
            .scalars()
            .all()
        )
    assert "answer" not in columns
    assert "assistant_message_id" in columns
    assert "retrieval_used" in columns

    # The resolved answer tracks the DURABLE message (its single source).
    assert (await _load(session_factory, "tg:3")).answer == "Original text."
    assert (await _load(session_factory, "tg:3")).assistant_message_id == mid

    async with session_factory() as session:
        await session.execute(
            sql_text("UPDATE messages SET text = 'Edited text.' WHERE id = :id"),
            {"id": mid},
        )
        await session.commit()

    assert (await _load(session_factory, "tg:3")).answer == "Edited text."


# ============================================================
# 3. REPLAY IS READ-ONLY AND STATE-INDEPENDENT
# ============================================================

@pytest.mark.asyncio
@pytest.mark.db
async def test_db_replay_creates_no_message_and_moves_no_fsm(
    session_factory,
):
    cid = await _conversation(session_factory)
    mid = await _assistant_message(session_factory, cid, "A stable answer.")

    await _claim(session_factory, "tg:4")
    await _finalize(
        session_factory,
        event_id="tg:4",
        outcome="answered",
        conversation_id=cid,
        assistant_message_id=mid,
        decision="ANSWER",
        reason="RESOLVED_BY_L1",
        case_action="NONE",
        **_answered_snapshot(retrieval_used=False, external_case_id=None),
    )

    before_messages = await _count_messages(session_factory, cid)
    before_state = await _conversation_triple(session_factory, cid)

    response = reproduce_finalized_response(await _load(session_factory, "tg:4"))

    assert response.answer == "A stable answer."
    assert await _count_messages(session_factory, cid) == before_messages
    assert await _conversation_triple(session_factory, cid) == before_state


@pytest.mark.asyncio
@pytest.mark.db
async def test_db_lifecycle_snapshot_beats_the_current_conversation(
    session_factory,
):
    cid = await _conversation(session_factory)
    mid = await _assistant_message(session_factory, cid, "Escalating now.")

    await _claim(session_factory, "tg:5")
    await _finalize(
        session_factory,
        event_id="tg:5",
        outcome="escalated",
        conversation_id=cid,
        assistant_message_id=mid,
        decision="ESCALATE",
        reason="STEPS_EXHAUSTED",
        case_action="CREATE_NEW",
        **_answered_snapshot(retrieval_used=False, external_case_id=None),
    )

    # The live conversation moves on AFTER finalization.
    async with session_factory() as session:
        await session.execute(
            sql_text(
                "UPDATE conversations SET conversation_mode = 'human', "
                "handoff_status = 'waiting_operator' WHERE conversation_id = :cid"
            ),
            {"cid": cid},
        )
        await session.commit()

    snapshot = await _load(session_factory, "tg:5")
    response = reproduce_finalized_response(snapshot)

    # The SNAPSHOT (AI/active/None) wins over the now-human conversation.
    assert response.conversation_mode == "ai"
    assert response.support_status == "active"
    assert response.handoff_status is None
    assert response.decision == "ESCALATE"
    assert response.case_action == "CREATE_NEW"


@pytest.mark.asyncio
@pytest.mark.db
async def test_db_replay_does_not_re_derive_external_case_id(session_factory):
    """The Tracker key is replayed from the SNAPSHOT, not the live row.

    An ``ESCALATE`` that pre-dated the case attach must not "gain" the case
    on replay, and vice versa.
    """
    cid = await _conversation(session_factory)
    mid = await _assistant_message(session_factory, cid, "Escalating now.")

    await _claim(session_factory, "tg:6")
    await _finalize(
        session_factory,
        event_id="tg:6",
        outcome="escalated",
        conversation_id=cid,
        assistant_message_id=mid,
        decision="ESCALATE",
        reason="STEPS_EXHAUSTED",
        case_action="CREATE_NEW",
        **_answered_snapshot(retrieval_used=False, external_case_id=None),
    )

    # A case is attached to the conversation AFTER the turn was finalized.
    async with session_factory() as session:
        await SessionService(session).attach_case(cid, "ANGIO-9")
        await session.commit()

    # The original body echoed no case; the replay must reproduce that.
    assert (await _load(session_factory, "tg:6")).external_case_id is None
    assert (
        reproduce_finalized_response(await _load(session_factory, "tg:6"))
    ).external_case_id is None


# ============================================================
# 4. ABSENT != CORRUPT
# ============================================================

@pytest.mark.asyncio
@pytest.mark.db
async def test_db_in_flight_event_is_unavailable(session_factory):
    await _claim(session_factory, "tg:in-flight")
    with pytest.raises(FinalizedReplayUnavailable):
        await _load(session_factory, "tg:in-flight")


@pytest.mark.asyncio
@pytest.mark.db
async def test_db_unknown_event_is_unavailable(session_factory):
    with pytest.raises(FinalizedReplayUnavailable):
        await _load(session_factory, "tg:never-seen")


@pytest.mark.asyncio
@pytest.mark.db
async def test_db_ai_turn_without_an_assistant_message_is_unavailable(
    session_factory,
):
    cid = await _conversation(session_factory)
    await _claim(session_factory, "tg:no-answer")
    # e.g. the START_NEW_SESSION bookkeeping row (answered, no message).
    async with session_factory() as session:
        await IdempotencyService(session).finalize(
            "tg:no-answer",
            "answered",
            cid,
            None,
            decision="START_NEW_SESSION",
            reason="START_NEW_SESSION",
            case_action="START_NEW_SESSION",
            retrieval_used=False,
            conversation_mode="ai",
            support_status="active",
            handoff_status=None,
        )
        await session.commit()

    with pytest.raises(FinalizedReplayUnavailable):
        await _load(session_factory, "tg:no-answer")


@pytest.mark.asyncio
@pytest.mark.db
async def test_db_legacy_finalized_row_without_snapshot_is_corruption(
    session_factory,
):
    cid = await _conversation(session_factory)
    mid = await _assistant_message(session_factory, cid, "Legacy answer.")
    await _claim(session_factory, "tg:legacy-row")

    # Simulate a pre-migration finalized row: outcome + metadata but NO
    # ``retrieval_used`` / lifecycle snapshot (they stay NULL).
    async with session_factory() as session:
        await session.execute(
            sql_text(
                "UPDATE inbound_events SET outcome = 'answered', "
                "conversation_id = :cid, assistant_message_id = :mid, "
                "decision = 'ANSWER', reason = 'RESOLVED_BY_L1', "
                "case_action = 'NONE' "
                "WHERE inbound_event_id = 'tg:legacy-row'"
            ),
            {"cid": cid, "mid": mid},
        )
        await session.commit()

    with pytest.raises(FinalizedReplayCorruption):
        await _load(session_factory, "tg:legacy-row")


# ============================================================
# 5. FORWARDED TURNS REPLAY EXACTLY
# ============================================================

@pytest.mark.asyncio
@pytest.mark.db
async def test_db_forwarded_finalized_replay(session_factory):
    cid = await _conversation(session_factory)
    await _claim(session_factory, "tg:fw")

    await _finalize(
        session_factory,
        event_id="tg:fw",
        outcome="forwarded",
        conversation_id=cid,
        assistant_message_id=None,
        decision="",
        reason="",
        case_action="HUMAN_FORWARD",
        retrieval_used=False,
        conversation_mode="human",
        support_status="active",
        handoff_status="waiting_operator",
        external_case_id="ANGIO-9",
    )

    response = reproduce_finalized_response(await _load(session_factory, "tg:fw"))
    assert response.answer == ""
    assert response.decision == ""
    assert response.reason == ""
    assert response.case_action == "HUMAN_FORWARD"
    assert response.conversation_mode == "human"
    assert response.handoff_status == "waiting_operator"
    assert response.external_case_id == "ANGIO-9"
    assert response.deduplicated is True
