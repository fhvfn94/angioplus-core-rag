# -*- coding: utf-8 -*-
"""
HL-2c.3b.3 -- POST /api/chat procedure-state CAS conflict.

When the ONE runtime boundary raises ``ProcedureStateConflict`` (the engine's
optimistic CAS lost a race), the transport must:

  * surface a STABLE ``409`` / ``{"error": "procedure_state_conflict"}``;
  * NOT retry and NOT run a second engine turn;
  * persist NOTHING -- no USER / ASSISTANT message, no FSM transition, and the
    inbound event is never falsely finalized.

Requires a dedicated test database via ``TEST_DATABASE_URL`` (same convention
as ``tests/test_chat_api_stateful.py``; fail-closed safety guard, see
``tests/conftest.py``; the destructive schema reset is centralized there).
"""

from __future__ import annotations

import pytest
from aiohttp.test_utils import TestClient, TestServer
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    create_async_engine,
)

import yandex.chat.server as server
from support.models import Conversation, Message
from support.procedure_state_store import ProcedureStateConflict
from support.runtime_service import StatefulTurnRuntime
from _db_safety import get_test_database_url


pytestmark = [pytest.mark.asyncio, pytest.mark.db]


@pytest.fixture(autouse=True)
def _patch_engine(monkeypatch):
    """Point the server's session factory at the test database."""
    engine = create_async_engine(get_test_database_url())
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)

    monkeypatch.setattr(server, "get_session_factory", lambda: factory)
    yield


async def _db_factory():
    engine = create_async_engine(get_test_database_url())
    factory = async_sessionmaker(engine, expire_on_commit=False)
    return engine, factory


class _ConflictRuntime:
    """``server._turn_runtime`` seam whose fresh engine ALWAYS conflicts.

    ``claim`` is the REAL one (the inbound event lifecycle must genuinely
    start), but ``execute_fresh_turn`` raises :class:`ProcedureStateConflict`
    exactly like a lost optimistic CAS would. Nothing is ever persisted.
    """

    def __init__(self, session_factory):
        self._real = StatefulTurnRuntime(session_factory=session_factory)
        self.fresh_calls = 0

    async def claim(self, *, inbound_event_id, channel):
        return await self._real.claim(
            inbound_event_id=inbound_event_id, channel=channel
        )

    async def execute_fresh_turn(self, **kwargs):
        self.fresh_calls += 1
        raise ProcedureStateConflict("lost the optimistic CAS race")

    async def resolve_duplicate(self, **kwargs):  # pragma: no cover
        raise AssertionError("no duplicate is expected in this scenario")


async def _client(monkeypatch):
    factory = server.get_session_factory()

    runtime = _ConflictRuntime(factory)
    monkeypatch.setattr(server, "_turn_runtime", lambda _session_factory: runtime)

    app = server.create_app()
    return TestClient(TestServer(app)), runtime


async def test_procedure_state_conflict_returns_409_and_persists_nothing(
    monkeypatch,
):
    client, runtime = await _client(monkeypatch)
    payload = {
        "message": "нужен оператор",
        "channel": "telegram",
        "external_user_id": "u-conflict",
        "external_conversation_id": "chat-conflict",
        "inbound_event_id": "tg:conflict-1",
    }

    async with client:
        resp = await client.post("/api/chat", json=payload)
        data = await resp.json()

        # A retry with the SAME inbound_event_id must NOT re-run the engine:
        # the event was claimed but never finalized, so it is served as an
        # in-flight duplicate instead.
        retry = await client.post("/api/chat", json=payload)
        retry_data = await retry.json()

    # The engine's optimistic CAS conflict is a STABLE, non-retried signal.
    assert resp.status == 409
    assert data == {"ok": False, "error": "procedure_state_conflict"}
    assert retry.status == 200
    assert retry_data["deduplicated"] is True
    assert runtime.fresh_calls == 1

    # NOTHING was persisted: no message, FSM unmoved.
    engine, factory = await _db_factory()
    try:
        async with factory() as s:
            message_count = await s.scalar(
                select(func.count()).select_from(Message)
            )
            conversation = (
                await s.execute(select(Conversation))
            ).scalars().first()
    finally:
        await engine.dispose()

    # A conversation row exists (it was resolved BEFORE the engine ran) but it
    # must be untouched AI/active with NO attached message.
    assert message_count == 0
    assert conversation is not None
    assert conversation.conversation_mode == "ai"
    assert conversation.support_status == "active"
    assert conversation.handoff_status is None
