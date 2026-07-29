# -*- coding: utf-8 -*-
import pytest
from fastapi.testclient import TestClient

from app.main import app

API_KEY = "test-api-key"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", API_KEY)
    return TestClient(app)


def test_ask_without_api_key_is_unauthorized(client):
    response = client.post("/ask", json={"question": "test"})
    assert response.status_code == 401


def test_ask_with_wrong_api_key_is_unauthorized(client):
    response = client.post(
        "/ask",
        headers={"X-API-Key": "wrong"},
        json={"question": "test"},
    )
    assert response.status_code == 401


def test_ask_without_server_key_is_unavailable(monkeypatch):
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    response = TestClient(app).post("/ask", json={"question": "test"})
    assert response.status_code == 503


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "x" * 2001},
        {"question": "test", "top_k": 0},
        {"question": "test", "top_k": 1000},
    ],
)
def test_ask_rejects_invalid_payloads(client, payload):
    response = client.post(
        "/ask",
        headers={"X-API-Key": API_KEY},
        json=payload,
    )
    assert response.status_code == 422


def test_ask_does_not_leak_internal_errors(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.setattr("app.main.embed_query", boom)

    response = client.post(
        "/ask",
        headers={"X-API-Key": API_KEY},
        json={"question": "test"},
    )

    assert response.status_code == 500
    assert "secret internal detail" not in response.text


def test_health_needs_no_api_key(monkeypatch):
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    assert TestClient(app).get("/health").status_code == 200
