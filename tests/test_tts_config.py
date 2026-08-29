# -*- coding: utf-8 -*-
"""Unit tests for voice/config.py TTS boundary config (Phase D2.3).

The TTS base URL is never hardcoded; TTS is disabled by default and only
enabled when TTS_BASE_URL is set. speaker/language/timeout/sample_rate have
sane defaults and can be overridden via env variables.
"""
from __future__ import annotations

from voice.config import (
    DEFAULT_TTS_LANGUAGE,
    DEFAULT_TTS_SAMPLE_RATE,
    DEFAULT_TTS_SPEAKER,
    DEFAULT_TTS_TIMEOUT_SECONDS,
    load_tts_config,
)


def test_tts_disabled_when_no_base_url(monkeypatch):
    monkeypatch.delenv("TTS_BASE_URL", raising=False)
    cfg = load_tts_config()
    assert cfg.enabled is False
    assert cfg.base_url is None
    # defaults still present
    assert cfg.speaker == DEFAULT_TTS_SPEAKER
    assert cfg.language == DEFAULT_TTS_LANGUAGE
    assert cfg.timeout_seconds == DEFAULT_TTS_TIMEOUT_SECONDS
    assert cfg.sample_rate == DEFAULT_TTS_SAMPLE_RATE


def test_tts_enabled_when_base_url_set(monkeypatch):
    monkeypatch.setenv("TTS_BASE_URL", "http://10.0.0.9:8001/")
    cfg = load_tts_config()
    assert cfg.enabled is True
    # trailing slash stripped
    assert cfg.base_url == "http://10.0.0.9:8001"


def test_tts_env_overrides(monkeypatch):
    monkeypatch.setenv("TTS_BASE_URL", "http://gpu:8001")
    monkeypatch.setenv("TTS_SPEAKER", "Ryan")
    monkeypatch.setenv("TTS_LANGUAGE", "Russian")
    monkeypatch.setenv("TTS_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("TTS_SAMPLE_RATE", "44100")
    cfg = load_tts_config()
    assert cfg.speaker == "Ryan"
    assert cfg.language == "Russian"
    assert cfg.timeout_seconds == 30
    assert cfg.sample_rate == 44100


def test_tts_invalid_timeout_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("TTS_BASE_URL", "http://gpu:8001")
    monkeypatch.setenv("TTS_TIMEOUT_SECONDS", "abc")
    cfg = load_tts_config()
    assert cfg.timeout_seconds == DEFAULT_TTS_TIMEOUT_SECONDS
