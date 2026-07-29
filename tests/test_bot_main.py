# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import httpx
import pytest

TEST_BOT_TOKEN = "123456789:" + "A" * 35


@pytest.fixture()
def bot_main(monkeypatch):
    """Imports ``bot.main`` with the environment it requires at import time."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TEST_BOT_TOKEN)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("MAX_VOICE_DURATION_SECONDS", "120")
    monkeypatch.delenv("RAG_API_URL", raising=False)

    for name in ("bot.main", "main"):
        sys.modules.pop(name, None)

    module = importlib.import_module("bot.main")
    yield module

    sys.modules.pop("bot.main", None)


class FakeMessage:
    def __init__(self, text: str | None = None, voice=None, user_id: int | None = 7):
        self.text = text
        self.voice = voice
        self.from_user = SimpleNamespace(id=user_id) if user_id is not None else None
        self.answers: list[str] = []
        self.edits: list[str] = []

    async def answer(self, text: str):
        self.answers.append(text)
        return self

    async def edit_text(self, text: str):
        self.edits.append(text)
        return self


def make_voice(duration: int = 5, mime_type: str = "audio/ogg"):
    return SimpleNamespace(duration=duration, mime_type=mime_type)


def test_missing_bot_token_raises_at_import(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    sys.modules.pop("bot.main", None)

    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN is not set"):
        importlib.import_module("bot.main")

    sys.modules.pop("bot.main", None)


def test_missing_gemini_key_raises_at_import(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TEST_BOT_TOKEN)
    monkeypatch.setenv("GEMINI_API_KEY", "")
    sys.modules.pop("bot.main", None)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
        importlib.import_module("bot.main")

    sys.modules.pop("bot.main", None)


def test_format_sources_returns_empty_string_without_sources(bot_main):
    assert bot_main.format_sources([]) == ""


def test_format_sources_formats_document_sources(bot_main):
    text = bot_main.format_sources(
        [
            {
                "file_name": "IFU.pdf",
                "section": "11 Инсталляция",
                "page_start": 9,
                "page_end": 10,
                "score": 0.91,
            }
        ]
    )

    assert text.startswith("\n\nИсточники:")
    assert "IFU.pdf" in text
    assert "pages 9 to 10, score 0.91" in text


def test_format_sources_formats_faq_rows(bot_main):
    text = bot_main.format_sources(
        [
            {
                "file_name": "FAQ.xlsx",
                "section": "Flow",
                "page_start": 42,
                "page_end": 42,
                "score": 0.88,
            }
        ]
    )

    assert "Строка FAQ: 42, score 0.88" in text
    assert "pages" not in text


def test_format_sources_handles_missing_fields(bot_main):
    text = bot_main.format_sources([{}])

    assert "pages  to , score " in text


async def test_ask_rag_posts_question(bot_main, monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["content"] = request.content
        return httpx.Response(200, json={"answer": "ok", "sources": []})

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return original_client(*args, **kwargs)

    monkeypatch.setattr(bot_main.httpx, "AsyncClient", client_factory)

    assert await bot_main.ask_rag("вопрос") == {"answer": "ok", "sources": []}
    assert captured["url"] == bot_main.RAG_API_URL
    assert b"top_k" in captured["content"]


async def test_ask_rag_raises_for_error_status(bot_main, monkeypatch):
    transport = httpx.MockTransport(lambda request: httpx.Response(500))
    original_client = httpx.AsyncClient

    monkeypatch.setattr(
        bot_main.httpx,
        "AsyncClient",
        lambda *args, **kwargs: original_client(*args, transport=transport, **kwargs),
    )

    with pytest.raises(httpx.HTTPStatusError):
        await bot_main.ask_rag("вопрос")


async def test_send_rag_answer_rejects_empty_question(bot_main):
    message = FakeMessage()

    await bot_main.send_rag_answer(message, "   ")

    assert len(message.answers) == 1
    assert "Не удалось определить текст вопроса" in message.answers[0]


async def test_send_rag_answer_appends_sources(bot_main, monkeypatch):
    async def fake_ask_rag(question):
        assert question == "вопрос"
        return {
            "answer": "Ответ",
            "sources": [
                {
                    "file_name": "IFU.pdf",
                    "section": "1 Общее",
                    "page_start": 1,
                    "page_end": 2,
                    "score": 0.9,
                }
            ],
        }

    monkeypatch.setattr(bot_main, "ask_rag", fake_ask_rag)
    message = FakeMessage()

    await bot_main.send_rag_answer(message, " вопрос ")

    assert message.answers[0] == "Секунду, проверяю базу знаний..."
    assert message.answers[1].startswith("Ответ")
    assert "IFU.pdf" in message.answers[1]


async def test_send_rag_answer_uses_default_answer(bot_main, monkeypatch):
    monkeypatch.setattr(bot_main, "ask_rag", lambda question: _coro({}))
    message = FakeMessage()

    await bot_main.send_rag_answer(message, "вопрос")

    assert message.answers[1] == "Информация не найдена в документации."


async def test_send_rag_answer_truncates_long_answers(bot_main, monkeypatch):
    monkeypatch.setattr(
        bot_main,
        "ask_rag",
        lambda question: _coro({"answer": "x" * 5000, "sources": []}),
    )
    message = FakeMessage()

    await bot_main.send_rag_answer(message, "вопрос")

    final_text = message.answers[1]
    assert final_text.endswith("Ответ обрезан из-за лимита Telegram.")
    assert final_text.startswith("x" * 3900)


async def test_send_rag_answer_reports_http_status_error(bot_main, monkeypatch):
    def raiser(question):
        raise httpx.HTTPStatusError(
            "boom",
            request=httpx.Request("POST", "http://rag/ask"),
            response=httpx.Response(503),
        )

    monkeypatch.setattr(bot_main, "ask_rag", raiser)
    message = FakeMessage()

    await bot_main.send_rag_answer(message, "вопрос")

    assert message.answers[1] == "RAG API вернул ошибку HTTP 503."


async def test_send_rag_answer_reports_generic_error(bot_main, monkeypatch):
    def raiser(question):
        raise RuntimeError("network down")

    monkeypatch.setattr(bot_main, "ask_rag", raiser)
    message = FakeMessage()

    await bot_main.send_rag_answer(message, "вопрос")

    assert message.answers[1] == "Ошибка при обращении к RAG API: network down"


async def test_start_command_greets(bot_main):
    message = FakeMessage()

    await bot_main.start(message)

    assert "AngioPlus Core" in message.answers[0]


async def test_handle_text_question_delegates_to_rag(bot_main, monkeypatch):
    captured: dict = {}

    async def fake_send(message, question):
        captured["question"] = question

    monkeypatch.setattr(bot_main, "send_rag_answer", fake_send)

    await bot_main.handle_text_question(FakeMessage(text="  вопрос  "))

    assert captured["question"] == "вопрос"


async def test_handle_text_question_rejects_blank_text(bot_main):
    message = FakeMessage(text="   ")

    await bot_main.handle_text_question(message)

    assert message.answers == [
        "Пришли вопрос текстом или голосовым сообщением."
    ]


async def test_handle_unsupported_message(bot_main):
    message = FakeMessage()

    await bot_main.handle_unsupported_message(message)

    assert message.answers == [
        "Пришли вопрос текстом или голосовым сообщением."
    ]


async def test_handle_voice_without_voice_payload(bot_main):
    message = FakeMessage()

    await bot_main.handle_voice(message)

    assert message.answers == ["Не удалось получить голосовое сообщение."]


async def test_handle_voice_rejects_too_long_voice(bot_main):
    message = FakeMessage(voice=make_voice(duration=999))

    await bot_main.handle_voice(message)

    assert "слишком длинное" in message.answers[0]


async def test_handle_voice_transcribes_and_answers(bot_main, monkeypatch):
    captured: dict = {}

    monkeypatch.setattr(
        bot_main.stt_service,
        "transcribe_voice",
        lambda voice: _coro("расшифровка"),
    )

    async def fake_send(message, question):
        captured["question"] = question

    monkeypatch.setattr(bot_main, "send_rag_answer", fake_send)
    message = FakeMessage(voice=make_voice())

    await bot_main.handle_voice(message)

    assert message.edits[0] == "📝 Распознано:\nрасшифровка"
    assert captured["question"] == "расшифровка"


@pytest.mark.parametrize(
    "error_text, expected_fragment",
    [
        ("429 RESOURCE_EXHAUSTED", "превышен лимит Gemini API"),
        ("quota exceeded", "превышен лимит Gemini API"),
        ("connection reset", "Попробуй записать его ещё раз"),
    ],
)
async def test_handle_voice_reports_transcription_errors(
    bot_main,
    monkeypatch,
    error_text,
    expected_fragment,
):
    def raiser(voice):
        raise RuntimeError(error_text)

    monkeypatch.setattr(bot_main.stt_service, "transcribe_voice", raiser)
    message = FakeMessage(voice=make_voice())

    await bot_main.handle_voice(message)

    assert expected_fragment in message.edits[0]


async def test_main_starts_polling(bot_main, monkeypatch):
    captured: dict = {}

    async def fake_start_polling(bot):
        captured["bot"] = bot

    monkeypatch.setattr(bot_main.dp, "start_polling", fake_start_polling)

    await bot_main.main()

    assert captured["bot"] is bot_main.bot


async def _coro(value):
    return value
