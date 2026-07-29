# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import httpx
import pytest

BOT_DIR = Path(__file__).resolve().parents[1] / "bot"
FAKE_TOKEN = "123456789:AAHfiqksKZ8WmR2zSjiQ7_v4-hlaHwMHnhs"


@pytest.fixture(scope="module")
def bot_main() -> ModuleType:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", FAKE_TOKEN)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.syspath_prepend(str(BOT_DIR))

        spec = importlib.util.spec_from_file_location(
            "angioplus_bot_main",
            BOT_DIR / "main.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        yield module

        del sys.modules[spec.name]


def call_ask_rag(
    bot_main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    handler,
) -> dict:
    async_client_cls = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        return async_client_cls(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(bot_main.httpx, "AsyncClient", client_factory)

    return asyncio.run(bot_main.ask_rag("вопрос"))


def test_ask_rag_returns_valid_payload(
    bot_main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"answer": "ответ", "sources": [{"file_name": "ifu.pdf"}]}

    data = call_ask_rag(
        bot_main,
        monkeypatch,
        lambda request: httpx.Response(200, json=payload),
    )

    assert data == payload


@pytest.mark.parametrize(
    "response_factory",
    [
        lambda request: httpx.Response(200, text="not json"),
        lambda request: httpx.Response(200, json=["unexpected"]),
        lambda request: httpx.Response(200, json={"sources": []}),
        lambda request: httpx.Response(200, json={"answer": "  "}),
        lambda request: httpx.Response(200, json={"answer": "ок", "sources": {}}),
    ],
)
def test_ask_rag_rejects_unusable_payloads(
    bot_main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    response_factory,
) -> None:
    with pytest.raises(bot_main.RagResponseError):
        call_ask_rag(bot_main, monkeypatch, response_factory)


def test_ask_rag_propagates_http_errors(
    bot_main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(httpx.HTTPStatusError):
        call_ask_rag(
            bot_main,
            monkeypatch,
            lambda request: httpx.Response(503, json={"detail": "down"}),
        )


def test_format_sources_tolerates_missing_fields(bot_main: ModuleType) -> None:
    text = bot_main.format_sources(
        [
            {
                "file_name": None,
                "section": None,
                "page_start": None,
                "page_end": None,
                "score": None,
            },
            {"file_name": "Q&A List ENG.xlsx", "section": "FAQ", "page_start": 12},
        ]
    )

    assert "None" not in text
    assert "Строка FAQ: 12" in text


def test_is_quota_error_detects_rate_limits(bot_main: ModuleType) -> None:
    assert bot_main.is_quota_error(RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert bot_main.is_quota_error(RuntimeError("quota exceeded"))
    assert not bot_main.is_quota_error(RuntimeError("invalid audio format"))
