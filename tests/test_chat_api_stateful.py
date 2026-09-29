"""
Stateful POST /api/chat tests (require a dedicated test database via
``TEST_DATABASE_URL``; fail-closed safety guard, see ``tests/conftest.py``;
the destructive schema reset is centralized there).

Verifies: escalation -> case_action, idempotent duplicate (no second LLM
call), and RESUME distinct from dedup.
"""

from __future__ import annotations

import asyncio

import pytest
from aiohttp.test_utils import TestClient, TestServer

import yandex.chat.server as server
from _db_safety import get_test_database_url
from support.domain import HandoffStatus, SupportEvent
from support.lifecycle_projection import CaseAction
from support.runtime_outcome import RuntimeTurn, RuntimeTurnKind
from support.runtime_service import FreshTurnDelivery, StatefulTurnRuntime


pytestmark = [pytest.mark.asyncio, pytest.mark.db]


@pytest.fixture(autouse=True)
def _patch_engine(monkeypatch):
    """Point the server's session factory at the test database."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(get_test_database_url())
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)

    monkeypatch.setattr(server, "get_session_factory", lambda: factory)
    yield


def _turn(
    *,
    conversation_id,
    text,
    decision,
    reason,
    case_action,
    fsm_event,
    outcome_status="TURN_EXECUTED",
):
    """A REAL ``RuntimeTurn`` for a genuine (engine-decided) turn."""
    return RuntimeTurn(
        kind=RuntimeTurnKind.USER_RESPONSE,
        outcome_status=outcome_status,
        conversation_id=conversation_id,
        text=text,
        decision=decision,
        reason=reason,
        case_action=case_action,
        fsm_event=fsm_event,
    )


def _escalate(*, message, history, conversation_id, projection_state, index):
    return _turn(
        conversation_id=conversation_id,
        text=f"answer-{index + 1}",
        decision="ESCALATE",
        reason="USER_REQUESTED_HUMAN",
        case_action=CaseAction.CREATE_NEW,
        fsm_event=SupportEvent.ESCALATE_NEW,
        outcome_status="ESCALATED",
    )


def _answer(*, message, history, conversation_id, projection_state, index):
    return _turn(
        conversation_id=conversation_id,
        text=f"answer-{index + 1}",
        decision="ANSWER",
        reason="RESOLVED_BY_L1",
        case_action=CaseAction.NONE,
        fsm_event=SupportEvent.USER_MESSAGE_AI,
        outcome_status="ANSWERED",
    )


def _resume_or_escalate(
    *, message, history, conversation_id, projection_state, index
):
    """Escalate a fresh conversation, but RESUME the SAME case once an
    operator has moved it to ``waiting_user`` (mirrors the projection)."""
    if projection_state[2] is HandoffStatus.WAITING_USER:
        return _turn(
            conversation_id=conversation_id,
            text=f"answer-{index + 1}",
            decision="ESCALATE",
            reason="USER_REQUESTED_HUMAN",
            case_action=CaseAction.RESUME,
            fsm_event=SupportEvent.ESCALATE_RESUME,
            outcome_status="ESCALATED",
        )
    return _escalate(
        message=message,
        history=history,
        conversation_id=conversation_id,
        projection_state=projection_state,
        index=index,
    )


class _ScriptedRuntime:
    """The ONE ``server._turn_runtime`` seam with a SCRIPTED engine.

    Everything EXCEPT the engine decision is the REAL
    :class:`~support.runtime_service.StatefulTurnRuntime`: the TX1 claim, the
    LEVEL A/B duplicate resolution and the finalized-snapshot replay are
    exercised for real. Only ``execute_fresh_turn`` is scripted, because a
    test must never make the external AI/network call.
    """

    def __init__(self, session_factory, respond):
        self._real = StatefulTurnRuntime(session_factory=session_factory)
        self._respond = respond
        self.fresh_calls: list = []

    async def claim(self, *, inbound_event_id, channel):
        return await self._real.claim(
            inbound_event_id=inbound_event_id, channel=channel
        )

    async def execute_fresh_turn(
        self,
        *,
        conversation_id,
        message,
        projection_state,
        history=(),
        execution_id=None,
    ):
        index = len(self.fresh_calls)
        self.fresh_calls.append(
            {
                "conversation_id": conversation_id,
                "message": message,
                "projection_state": projection_state,
                "history": list(history),
                "execution_id": execution_id,
            }
        )
        return FreshTurnDelivery(
            turn=self._respond(
                message=message,
                history=list(history),
                conversation_id=conversation_id,
                projection_state=projection_state,
                index=index,
            )
        )

    async def resolve_duplicate(self, *, inbound_event_id, conversation_id):
        return await self._real.resolve_duplicate(
            inbound_event_id=inbound_event_id,
            conversation_id=conversation_id,
        )


async def _client(monkeypatch, respond=_escalate):
    """Real handler + services; only the ENGINE DECISION is scripted.

    The runtime boundary is installed through the DOCUMENTED seam
    (``server._turn_runtime``); the legacy ``server.run_chat`` is never
    patched, because the stateful path must not use it at all.
    """
    factory = server.get_session_factory()

    runtime = _ScriptedRuntime(factory, respond)
    monkeypatch.setattr(server, "_turn_runtime", lambda _session_factory: runtime)

    app = server.create_app()
    return TestClient(TestServer(app)), runtime


async def test_escalation_creates_case(monkeypatch):
    client, runtime = await _client(monkeypatch)
    async with client:
        resp = await client.post(
            "/api/chat",
            json={
                "message": "нужен оператор",
                "channel": "telegram",
                "external_user_id": "u1",
                "external_conversation_id": "chat1",
                "inbound_event_id": "tg:100",
            },
        )
        data = await resp.json()

    assert resp.status == 200
    assert data["case_action"] == "CREATE_NEW"
    assert data["conversation_mode"] == "human"
    assert data["handoff_status"] == "waiting_operator"
    # The stateful fresh path runs the ONE runtime boundary exactly once
    # (and never the legacy ``server.run_chat``).
    assert len(runtime.fresh_calls) == 1


async def test_duplicate_does_not_call_llm_again(monkeypatch):
    client, runtime = await _client(monkeypatch)
    payload = {
        "message": "нужен оператор",
        "channel": "telegram",
        "external_user_id": "u1",
        "external_conversation_id": "chat1",
        "inbound_event_id": "tg:200",
    }
    async with client:
        first = await (await client.post("/api/chat", json=payload)).json()
        second = await (await client.post("/api/chat", json=payload)).json()

    # The engine ran EXACTLY once: the duplicate is served from durable state.
    assert len(runtime.fresh_calls) == 1
    assert first["deduplicated"] is None or first["deduplicated"] is False
    assert second["deduplicated"] is True
    assert second["answer"] == first["answer"]
    assert second["decision"] == first["decision"]
    assert second["case_action"] == first["case_action"]


async def test_resume_is_not_dedup(monkeypatch):
    """
    A NEW event while the conversation is waiting_user resumes the SAME case
    (case_action=RESUME), and is NOT treated as a duplicate.
    """
    client, _ = await _client(monkeypatch, _resume_or_escalate)
    async with client:
        # 1) escalate
        await client.post(
            "/api/chat",
            json={
                "message": "нужен оператор",
                "channel": "telegram",
                "external_user_id": "u1",
                "external_conversation_id": "chat1",
                "inbound_event_id": "tg:300",
            },
        )

    # Simulate the operator moving the case to need-info (waiting_user).
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from support.models import Conversation

    eng = create_async_engine(get_test_database_url())
    factory = async_sessionmaker(eng, expire_on_commit=False)
    async with factory() as s:
        row = (
            await s.execute(select(Conversation).order_by(Conversation.id))
        ).scalars().first()
        row.conversation_mode = "ai"
        row.handoff_status = "waiting_user"
        await s.commit()
    await eng.dispose()

    client2, _ = await _client(monkeypatch, _resume_or_escalate)
    async with client2:
        resp = await client2.post(
            "/api/chat",
            json={
                "message": "готов продолжить",
                "channel": "telegram",
                "external_user_id": "u1",
                "external_conversation_id": "chat1",
                "inbound_event_id": "tg:301",
            },
        )
        data = await resp.json()

    assert data["case_action"] == "RESUME"
    assert data.get("deduplicated") in (None, False)


async def test_stateful_request_requires_external_conversation_id(monkeypatch):
    client, runtime = await _client(monkeypatch)
    async with client:
        resp = await client.post(
            "/api/chat",
            json={
                "message": "hello",
                "channel": "telegram",
                "external_user_id": "u1",
            },
        )
        data = await resp.json()

    assert resp.status == 400
    assert data["error"] == "external_conversation_id_required"
    assert len(runtime.fresh_calls) == 0


async def test_session_start_creates_clean_conversation(monkeypatch):
    client, _ = await _client(monkeypatch)
    async with client:
        resp = await client.post(
            "/api/session/start",
            json={
                "channel": "telegram",
                "external_user_id": "u1",
                "external_conversation_id": "chat1",
                "inbound_event_id": "tg:start-1",
            },
        )
        data = await resp.json()

    assert resp.status == 200
    assert data == {
        "ok": True,
        "conversation_id": data["conversation_id"],
        "conversation_mode": "ai",
        "support_status": "active",
        "handoff_status": None,
        "deduplicated": False,
    }


async def test_session_start_duplicate_returns_same_conversation(monkeypatch):
    client, _ = await _client(monkeypatch)
    payload = {
        "channel": "telegram",
        "external_user_id": "u1",
        "external_conversation_id": "chat1",
        "inbound_event_id": "tg:start-2",
    }
    async with client:
        first = await (await client.post("/api/session/start", json=payload)).json()
        second = await (await client.post("/api/session/start", json=payload)).json()

    assert first["conversation_id"] == second["conversation_id"]
    assert second["deduplicated"] is True


async def test_concurrent_duplicate_session_start_creates_one_conversation(monkeypatch):
    client, _ = await _client(monkeypatch)
    payload = {
        "channel": "telegram",
        "external_user_id": "u1",
        "external_conversation_id": "chat1",
        "inbound_event_id": "tg:start-3",
    }
    async with client:
        responses = await asyncio.gather(
            client.post("/api/session/start", json=payload),
            client.post("/api/session/start", json=payload),
        )
        data = await asyncio.gather(*(response.json() for response in responses))

    assert {item["conversation_id"] for item in data}.__len__() == 1
    assert sorted(item["deduplicated"] for item in data) == [False, True]


async def test_distinct_session_start_events_are_sequential(monkeypatch):
    client, _ = await _client(monkeypatch)
    base = {
        "channel": "telegram",
        "external_user_id": "u1",
        "external_conversation_id": "chat1",
    }
    async with client:
        first, second = await asyncio.gather(
            client.post(
                "/api/session/start",
                json={**base, "inbound_event_id": "tg:start-4"},
            ),
            client.post(
                "/api/session/start",
                json={**base, "inbound_event_id": "tg:start-5"},
            ),
        )
        first_data, second_data = await asyncio.gather(
            first.json(), second.json()
        )

    assert first_data["conversation_id"] != second_data["conversation_id"]
    assert first_data["deduplicated"] is False
    assert second_data["deduplicated"] is False


async def test_stateful_turn_passes_previous_messages_as_history(monkeypatch):
    client, runtime = await _client(monkeypatch, _answer)

    async with client:
        first_resp = await client.post(
            "/api/chat",
            json={
                "message": "После запуска чёрный экран",
                "channel": "telegram",
                "external_user_id": "memory-user",
                "external_conversation_id": "memory-chat",
                "inbound_event_id": "tg:memory-1",
            },
        )

        second_resp = await client.post(
            "/api/chat",
            json={
                "message": "Я уже два раза перезапускал",
                "channel": "telegram",
                "external_user_id": "memory-user",
                "external_conversation_id": "memory-chat",
                "inbound_event_id": "tg:memory-2",
            },
        )

        assert first_resp.status == 200
        assert second_resp.status == 200

    assert len(runtime.fresh_calls) == 2

    assert runtime.fresh_calls[1]["message"] == "Я уже два раза перезапускал"

    assert runtime.fresh_calls[1]["history"] == [
        {
            "role": "user",
            "text": "После запуска чёрный экран",
        },
        {
            "role": "assistant",
            "text": "answer-1",
        },
    ]

    assert all(
        item["text"] != "Я уже два раза перезапускал"
        for item in runtime.fresh_calls[1]["history"]
    )