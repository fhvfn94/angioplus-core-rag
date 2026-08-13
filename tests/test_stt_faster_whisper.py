# -*- coding: utf-8 -*-
"""Unit tests for the faster-whisper STT provider.

These tests DO NOT download or load a real Whisper model. They substitute a fake
`faster_whisper` module (and fake `WhisperModel`) via sys.modules so the library
is not required, while exercising the real STTService code paths.
"""
from __future__ import annotations

import io
import sys
import types
from types import SimpleNamespace

import pytest

from bot.services.stt import STTService


class FakeWhisperModel:
    """Configurable stand-in for faster_whisper.WhisperModel."""

    def __init__(self, model, device, compute_type):
        self.model_arg = model
        self.device_arg = device
        self.compute_type_arg = compute_type
        type(self).constructed_args.append((model, device, compute_type))
        self.transcribe_audio = None
        self.transcribe_language = None
        self.transcribe_initial_prompt = None

    def transcribe(self, audio, language=None, initial_prompt=None):
        self.transcribe_audio = audio
        self.transcribe_language = language
        self.transcribe_initial_prompt = initial_prompt
        return iter(type(self).segments_for_transcribe), {"language": language}


@pytest.fixture
def fake_whisper(monkeypatch):
    """Inject a fake faster_whisper module and record constructor calls."""
    # Fresh state per test.
    FakeWhisperModel.constructed_args = []
    FakeWhisperModel.segments_for_transcribe = [
        SimpleNamespace(text="Привет "),
        SimpleNamespace(text="мир"),
        SimpleNamespace(text="."),
    ]

    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    return FakeWhisperModel


def _make_service() -> STTService:
    # provider=faster_whisper -> gemini client is not constructed.
    return STTService(
        bot=None,
        gemini_api_key="unused",
        gemini_audio_model="unused",
        provider="faster_whisper",
    )


# --- provider selection ---

def test_faster_whisper_service_does_not_need_gemini_client():
    service = _make_service()
    assert service.provider == "faster_whisper"
    assert service.gemini_client is None


# --- model config forwarding ---

def test_whisper_model_config_forwards_model_device_compute(fake_whisper):
    service = _make_service()
    service.whisper_model = "whisper-small"
    service.whisper_device = "cuda"
    service.whisper_compute_type = "float16"

    model = service._load_whisper_model()

    assert fake_whisper.constructed_args == [
        ("whisper-small", "cuda", "float16")
    ]
    assert model is not None
    assert model.model_arg == "whisper-small"
    assert model.device_arg == "cuda"
    assert model.compute_type_arg == "float16"


# --- lazy load + reuse ---

def test_whisper_model_loaded_once_and_reused(fake_whisper):
    service = _make_service()

    first = service._load_whisper_model()
    second = service._load_whisper_model()

    assert len(fake_whisper.constructed_args) == 1
    assert first is second
    assert service._whisper_model is first


def test_model_not_loaded_until_first_request(fake_whisper):
    service = _make_service()
    # Constructing the service must NOT load any model.
    assert service._whisper_model is None
    assert fake_whisper.constructed_args == []


# --- transcription from segments ---

def test_transcription_joins_segments(fake_whisper):
    service = _make_service()

    text = service._transcribe_audio_faster_whisper(b"audio-bytes")

    assert text == "Привет мир."


def test_blocking_transcribe_receives_in_memory_audio_buffer(fake_whisper):
    service = _make_service()
    audio_bytes = b"fake-ogg-opus"
    model = service._load_whisper_model()

    service._transcribe_audio_faster_whisper(audio_bytes)

    # Must be an in-memory BytesIO (no temp file), not a path string.
    assert isinstance(model.transcribe_audio, io.BytesIO)
    assert model.transcribe_audio.getvalue() == audio_bytes
    # Empty STT_LANGUAGE -> None (auto-detect).
    assert model.transcribe_language is None


def test_empty_transcription_raises(fake_whisper):
    service = _make_service()
    fake_whisper.segments_for_transcribe = []

    with pytest.raises(
        RuntimeError, match="faster-whisper returned an empty transcription"
    ):
        service._transcribe_audio_faster_whisper(b"audio-bytes")


# --- language hint forwarding ---

def test_whisper_language_hint_is_forwarded(fake_whisper):
    service = _make_service()
    service.whisper_language = "ru"
    model = service._load_whisper_model()

    service._transcribe_audio_faster_whisper(b"audio-bytes")

    assert model.transcribe_language == "ru"


# --- initial prompt forwarding ---

def test_whisper_initial_prompt_is_forwarded(fake_whisper):
    service = _make_service()
    service.whisper_initial_prompt = "Специализированный медицинский словарь"
    model = service._load_whisper_model()

    service._transcribe_audio_faster_whisper(b"audio-bytes")

    assert model.transcribe_initial_prompt == (
        "Специализированный медицинский словарь"
    )


def test_whisper_initial_prompt_empty_becomes_none(fake_whisper):
    service = _make_service()
    # Empty (default) -> stored as None, so initial_prompt passed as None.
    service.whisper_initial_prompt = None
    model = service._load_whisper_model()

    service._transcribe_audio_faster_whisper(b"audio-bytes")

    assert service.whisper_initial_prompt is None
    assert model.transcribe_initial_prompt is None
