"""
Phase 1A acceptance test: normal Telegram AI conversation must be
PostgreSQL-backed through the EXISTING stateful backend.

This exercises the REAL stateful orchestration in
``yandex.chat.server._handle_stateful_turn`` (services, FSM, idempotency,
PostgreSQL persistence) while mocking ONLY the engine DECISION through the
documented ``server._turn_runtime`` seam (the external AI/network call).

Scenario (mirrors the Phase 1A acceptance criteria):

  /api/session/start (telegram / phase1a-user / phase1a-chat)
  /api/chat  "Что такое PACS?"                                  (tg:1)
  /api/chat  "Какие системные требования ...?"                  (tg:2)
  replay /api/chat "Какие системные требования ...?"            (tg:2)

After the scenario PostgreSQL must contain exactly ONE active
conversation for that thread and exactly FOUR messages attached to the
SAME backend conversation_id, in the role order
["user", "assistant", "user", "assistant"].

Requires a dedicated test database via ``TEST_DATABASE_URL`` (same
convention as tests/test_chat_api_stateful.py; fail-closed safety guard,
see ``tests/conftest.py``). The destructive schema reset is centralized
there.
"""

from __future__ import annotations

import pytest
from aiohttp.test_utils import TestClient, TestServer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    create_async_engine,
)

import yandex.chat.server as server
from support.domain import SupportEvent
from support.lifecycle_projection import CaseAction
from support.models import Conversation, Message
from support.runtime_outcome import RuntimeTurn, RuntimeTurnKind
from support.runtime_service import FreshTurnDelivery, StatefulTurnRuntime
from _db_safety import get_test_database_url


pytestmark = [pytest.mark.asyncio, pytest.mark.db]

CHANNEL = "telegram"
USER = "phase1a-user"
CHAT = "phase1a-chat"


@pytest.fixture(autouse=True)
def _patch_engine(monkeypatch):
    """Point the server's session factory at the test database."""
    engine = create_async_engine(get_test_database_url())
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)

    monkeypatch.setattr(server, "get_session_factory", lambda: factory)
    yield


class _ScriptedRuntime:
    """The ONE ``server._turn_runtime`` seam with a SCRIPTED engine.

    Everything EXCEPT the engine decision is the REAL
    :class:`~support.runtime_service.StatefulTurnRuntime`: the TX1 claim, the
    LEVEL A duplicate replay and the finalized snapshot are exercised for real.
    Only ``execute_fresh_turn`` is scripted, because a test must never make
    the external AI/network call.
    """

    def __init__(self, session_factory, *, decision, reason):
        self._real = StatefulTurnRuntime(session_factory=session_factory)
        self._decision = decision
        self._reason = reason
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
        self.fresh_calls.append({"message": message, "history": list(history)})
        return FreshTurnDelivery(
            turn=RuntimeTurn(
                kind=RuntimeTurnKind.USER_RESPONSE,
                outcome_status="ANSWERED",
                conversation_id=conversation_id,
                text=f"answer-{len(self.fresh_calls)}",
                decision=self._decision,
                reason=self._reason,
                case_action=CaseAction.NONE,
                fsm_event=SupportEvent.USER_MESSAGE_AI,
            )
        )

    async def resolve_duplicate(self, *, inbound_event_id, conversation_id):
        return await self._real.resolve_duplicate(
            inbound_event_id=inbound_event_id,
            conversation_id=conversation_id,
        )


async def _client(monkeypatch, decision="ANSWER", reason="RESOLVED_BY_L1"):
    """Real handler + services; only the ENGINE DECISION is scripted.

    The runtime boundary is installed through the DOCUMENTED seam
    (``server._turn_runtime``): the stateful path never calls the legacy
    ``server.run_chat``.
    """
    factory = server.get_session_factory()

    runtime = _ScriptedRuntime(factory, decision=decision, reason=reason)
    monkeypatch.setattr(server, "_turn_runtime", lambda _session_factory: runtime)

    app = server.create_app()
    return TestClient(TestServer(app)), runtime


async def _db_factory():
    engine = create_async_engine(get_test_database_url())
    factory = async_sessionmaker(engine, expire_on_commit=False)
    return engine, factory


async def test_phase1a_normal_ai_path_is_postgres_backed(monkeypatch):
    client, runtime = await _client(monkeypatch)

    async with client:
        # 1. Session start -------------------------------------------------
        start = await (
            await client.post(
                "/api/session/start",
                json={
                    "channel": CHANNEL,
                    "external_user_id": USER,
                    "external_conversation_id": CHAT,
                    "inbound_event_id": "tg:start-phase1a",
                },
            )
        ).json()
        assert start["ok"] is True
        session_conversation_id = start["conversation_id"]

        # 2. First AI turn -------------------------------------------------
        first = await (
            await client.post(
                "/api/chat",
                json={
                    "message": "Что такое PACS?",
                    "channel": CHANNEL,
                    "external_user_id": USER,
                    "external_conversation_id": CHAT,
                    "inbound_event_id": "tg:1",
                },
            )
        ).json()

        assert first["conversation_id"] == session_conversation_id
        assert first["conversation_mode"] == "ai"
        assert first["support_status"] == "active"
        assert first["deduplicated"] is False

        # 3. Second AI turn (different event, same thread) -----------------
        second = await (
            await client.post(
                "/api/chat",
                json={
                    "message": (
                        "Какие системные требования предъявляются "
                        "к программному обеспечению?"
                    ),
                    "channel": CHANNEL,
                    "external_user_id": USER,
                    "external_conversation_id": CHAT,
                    "inbound_event_id": "tg:2",
                },
            )
        ).json()

        assert second["conversation_id"] == session_conversation_id
        assert second["conversation_mode"] == "ai"

        # 5. Replay second request with the SAME inbound_event_id ----------
        replay = await (
            await client.post(
                "/api/chat",
                json={
                    "message": (
                        "Какие системные требования предъявляются "
                        "к программному обеспечению?"
                    ),
                    "channel": CHANNEL,
                    "external_user_id": USER,
                    "external_conversation_id": CHAT,
                    "inbound_event_id": "tg:2",
                },
            )
        ).json()

        assert replay["deduplicated"] is True
        # Replay returns the cached answer of the original turn.
        assert replay["answer"] == second["answer"]

    assert len(runtime.fresh_calls) == 2

    # 4. Inspect PostgreSQL directly ---------------------------------------
    engine, factory = await _db_factory()
    try:
        async with factory() as s:
            conversations = (
                await s.execute(
                    select(Conversation).where(
                        Conversation.channel == CHANNEL,
                        Conversation.external_user_id == USER,
                        Conversation.external_conversation_id == CHAT,
                        Conversation.support_status == "active",
                    )
                )
            ).scalars().all()

            assert len(conversations) == 1
            assert conversations[0].conversation_id == session_conversation_id

            messages = (
                await s.execute(
                    select(Message)
                    .where(Message.conversation_id == session_conversation_id)
                    .order_by(Message.created_at, Message.id)
                )
            ).scalars().all()

            assert [m.role for m in messages] == [
                "user",
                "assistant",
                "user",
                "assistant",
            ]
            assert all(
                m.conversation_id == session_conversation_id for m in messages
            )
            # The four texts are: the two user questions (verbatim) and the
            # two mocked assistant answers.
            assert messages[0].text == "Что такое PACS?"
            assert messages[2].text == (
                "Какие системные требования предъявляются "
                "к программному обеспечению?"
            )
            assert messages[1].text == first["answer"]
            assert messages[3].text == second["answer"]
    finally:
        await engine.dispose()
