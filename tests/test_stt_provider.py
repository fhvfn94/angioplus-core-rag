# -*- coding: utf-8 -*-
"""Unit tests for STT provider selection.

These tests only exercise configuration / provider resolution logic. They do
NOT call the real Gemini API and do NOT download/load any Whisper model.
"""
from __future__ import annotations

import pytest

from bot.services.stt import (
    STT_DEFAULT_PROVIDER,
    SUPPORTED_STT_PROVIDERS,
    resolve_stt_provider,
)


@pytest.fixture
def clear_stt_provider(monkeypatch):
    """Remove STT_PROVIDER before each test so env is fully controlled."""
    monkeypatch.delenv("STT_PROVIDER", raising=False)
    yield


def test_default_provider_is_gemini(clear_stt_provider):
    assert resolve_stt_provider() == "gemini"
    assert STT_DEFAULT_PROVIDER == "gemini"
    assert SUPPORTED_STT_PROVIDERS == ("gemini",)


def test_explicit_gemini(clear_stt_provider, monkeypatch):
    monkeypatch.setenv("STT_PROVIDER", "gemini")
    assert resolve_stt_provider() == "gemini"


def test_provider_is_case_insensitive(clear_stt_provider, monkeypatch):
    monkeypatch.setenv("STT_PROVIDER", "GEMINI")
    assert resolve_stt_provider() == "gemini"


def test_provider_is_whitespace_trimmed(clear_stt_provider, monkeypatch):
    monkeypatch.setenv("STT_PROVIDER", "  gemini  ")
    assert resolve_stt_provider() == "gemini"


def test_empty_provider_value_uses_default(clear_stt_provider, monkeypatch):
    monkeypatch.setenv("STT_PROVIDER", "")
    assert resolve_stt_provider() == "gemini"


def test_unknown_provider_raises(clear_stt_provider, monkeypatch):
    monkeypatch.setenv("STT_PROVIDER", "faster_whisper")
    with pytest.raises(ValueError, match="Unknown STT_PROVIDER"):
        resolve_stt_provider()


def test_unknown_provider_message_lists_supported(
    clear_stt_provider, monkeypatch
):
    monkeypatch.setenv("STT_PROVIDER", "bogus")
    with pytest.raises(ValueError) as exc_info:
        resolve_stt_provider()
    assert "Supported values: gemini" in str(exc_info.value)
