"""
Stateful POST /api/chat tests (require PostgreSQL + TEST_DATABASE_URL).

Verifies: escalation -> case_action, idempotent duplicate (no second LLM
call), and RESUME distinct from dedup.
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

import yandex.chat.server as server
from support.db import Base


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "").strip()
pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _require_db():
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL is not set; skipping stateful API test")


@pytest.fixture(autouse=True)
def _patch_engine(monkeypatch):
    """Point the server's session factory at the test database."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(TEST_DATABASE_URL)
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)

    monkeypatch.setattr(server, "get_session_factory", lambda: factory)
    yield


@pytest_asyncio.fixture(autouse=True)
async def _schema():
    from sqlalchemy.ext.asyncio import create_async_engine

    eng = create_async_engine(TEST_DATABASE_URL)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    await eng.dispose()


async def _client(monkeypatch, decision="ESCALATE", reason="USER_REQUESTED_HUMAN"):
    calls = {"n": 0}

    async def fake_run_chat(message: str):
        calls["n"] += 1
        return {
            "answer": f"answer-{calls['n']}",
            "decision": decision,
            "reason": reason,
            "retrieval_used": False,
        }

    monkeypatch.setattr(server, "run_chat", fake_run_chat)
    app = server.create_app()
    return TestClient(TestServer(app)), calls


async def test_escalation_creates_case(monkeypatch):
    client, calls = await _client(monkeypatch)
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
    assert calls["n"] == 1


async def test_duplicate_does_not_call_llm_again(monkeypatch):
    client, calls = await _client(monkeypatch)
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

    assert calls["n"] == 1
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
    client, calls = await _client(monkeypatch)
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

    eng = create_async_engine(TEST_DATABASE_URL)
    factory = async_sessionmaker(eng, expire_on_commit=False)
    async with factory() as s:
        row = (
            await s.execute(select(Conversation).order_by(Conversation.id))
        ).scalars().first()
        row.conversation_mode = "ai"
        row.handoff_status = "waiting_user"
        await s.commit()
    await eng.dispose()

    client2, calls2 = await _client(monkeypatch)
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
    client, calls = await _client(monkeypatch)
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
    assert calls["n"] == 0
