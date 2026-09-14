"""
POST /api/chat backward-compatibility test.

The legacy payload {"message": "..."} must keep the exact old behaviour
and must NOT require a database (no dedup). We monkeypatch the LLM core.
"""

from __future__ import annotations

import pytest
from aiohttp.test_utils import TestClient, TestServer

import yandex.chat.server as server


@pytest.mark.asyncio
async def test_legacy_payload_shape(monkeypatch):
    async def fake_run_chat(message: str):
        return {
            "answer": f"echo: {message}",
            "decision": "ANSWER",
            "reason": "RESOLVED_BY_L1",
            "retrieval_used": False,
        }

    monkeypatch.setattr(server, "run_chat", fake_run_chat)

    app = server.create_app()
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/api/chat", json={"message": "hello"})
        assert resp.status == 200
        data = await resp.json()

    assert data["ok"] is True
    assert data["answer"] == "echo: hello"
    assert data["decision"] == "ANSWER"
    assert data["reason"] == "RESOLVED_BY_L1"
    assert data["retrieval_used"] is False
    # No stateful extras in legacy mode.
    assert "conversation_id" not in data
    assert "deduplicated" not in data


@pytest.mark.asyncio
async def test_message_required(monkeypatch):
    app = server.create_app()
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/api/chat", json={"message": ""})
        assert resp.status == 400
        data = await resp.json()
    assert data["error"] == "message_required"
