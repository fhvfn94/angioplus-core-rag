# -*- coding: utf-8 -*-
from __future__ import annotations

from types import SimpleNamespace

import pytest

from bot.services.stt import STTService


class FakeModels:
    def __init__(self, text=None, error: Exception | None = None):
        self._text = text
        self._error = error
        self.calls: list[dict] = []

    def generate_content(self, *, model, contents):
        self.calls.append({"model": model, "contents": contents})
        if self._error is not None:
            raise self._error
        return SimpleNamespace(text=self._text)


class FakeGeminiClient:
    def __init__(self, text=None, error: Exception | None = None):
        self.models = FakeModels(text=text, error=error)


DOWNLOADED_FILE = object()


class FakeBot:
    """Mimics ``aiogram.Bot.download`` writing into the given destination."""

    def __init__(
        self,
        payload: bytes | None = b"audio-bytes",
        result=DOWNLOADED_FILE,
    ):
        self.payload = payload
        self.result = result
        self.calls: list[dict] = []

    async def download(self, file, destination, timeout):
        self.calls.append(
            {"file": file, "destination": destination, "timeout": timeout}
        )
        if self.payload:
            destination.write(self.payload)
        return self.result


def make_service(bot=None, *, text=None, error=None, download_timeout=60):
    service = STTService.__new__(STTService)
    service.bot = bot if bot is not None else FakeBot()
    service.gemini_audio_model = "gemini-audio"
    service.download_timeout = download_timeout
    service.gemini_client = FakeGeminiClient(text=text, error=error)
    return service


def make_voice(mime_type: str | None = "audio/ogg", duration: int = 3):
    return SimpleNamespace(mime_type=mime_type, duration=duration, file_id="fid")


def test_init_creates_gemini_client(monkeypatch):
    created: dict = {}

    class FakeClient:
        def __init__(self, api_key):
            created["api_key"] = api_key

    monkeypatch.setattr("bot.services.stt.genai.Client", FakeClient)

    bot = FakeBot()
    service = STTService(
        bot=bot,
        gemini_api_key="key",
        gemini_audio_model="gemini-audio",
        download_timeout=30,
    )

    assert created["api_key"] == "key"
    assert service.bot is bot
    assert service.gemini_audio_model == "gemini-audio"
    assert service.download_timeout == 30


def test_transcribe_audio_sync_returns_stripped_text():
    service = make_service(text="  расшифровка  ")

    assert service._transcribe_audio_sync(b"bytes", "audio/ogg") == "расшифровка"

    call = service.gemini_client.models.calls[0]
    assert call["model"] == "gemini-audio"
    prompt, audio_part = call["contents"]
    assert "AngioPlus Core" in prompt
    assert audio_part.inline_data.mime_type == "audio/ogg"
    assert audio_part.inline_data.data == b"bytes"


@pytest.mark.parametrize("text", [None, "", "   "])
def test_transcribe_audio_sync_rejects_empty_transcription(text):
    service = make_service(text=text)

    with pytest.raises(RuntimeError, match="empty transcription"):
        service._transcribe_audio_sync(b"bytes", "audio/ogg")


async def test_transcribe_voice_happy_path():
    bot = FakeBot(payload=b"audio-bytes")
    service = make_service(bot, text="привет")

    assert await service.transcribe_voice(make_voice()) == "привет"

    assert bot.calls[0]["timeout"] == 60
    assert service.gemini_client.models.calls[0]["contents"][1].inline_data.data == (
        b"audio-bytes"
    )


async def test_transcribe_voice_defaults_mime_type_when_missing():
    service = make_service(text="привет")

    await service.transcribe_voice(make_voice(mime_type=None))

    part = service.gemini_client.models.calls[0]["contents"][1]
    assert part.inline_data.mime_type == "audio/ogg"


async def test_transcribe_voice_raises_when_download_returns_none():
    service = make_service(FakeBot(result=None), text="привет")

    with pytest.raises(RuntimeError, match="could not be downloaded"):
        await service.transcribe_voice(make_voice())


async def test_transcribe_voice_raises_on_empty_file():
    service = make_service(FakeBot(payload=b""), text="привет")

    with pytest.raises(RuntimeError, match="empty"):
        await service.transcribe_voice(make_voice())


async def test_transcribe_voice_propagates_gemini_errors():
    service = make_service(error=RuntimeError("429 RESOURCE_EXHAUSTED"))

    with pytest.raises(RuntimeError, match="429"):
        await service.transcribe_voice(make_voice())
