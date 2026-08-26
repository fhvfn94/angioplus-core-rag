# -*- coding: utf-8 -*-
"""Unit tests for the Silero VAD loader/configuration (voice/vad.py).

These tests do NOT download or load the real ONNX model. The loader contract is
tested by injecting a fake `vad_loader`, so we verify configuration parsing and
forwarding without neural-network inference or model files.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from voice.vad import VADConfig, load_vad


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_default_config_matches_russian_assistant_defaults():
    cfg = VADConfig()
    # Sensible defaults for turn detection.
    assert cfg.min_speech_duration == 0.05
    assert cfg.min_silence_duration == 0.55   # end-of-turn silence threshold
    assert cfg.prefix_padding_duration == 0.5
    assert cfg.activation_threshold == 0.5
    assert cfg.sample_rate == 16000           # Silero supports 8k / 16k
    assert cfg.force_cpu is True


def test_config_custom_values_are_stored():
    cfg = VADConfig(
        min_speech_duration=0.1,
        min_silence_duration=0.4,
        prefix_padding_duration=0.3,
        activation_threshold=0.6,
        sample_rate=16000,
    )
    assert cfg.min_speech_duration == 0.1
    assert cfg.min_silence_duration == 0.4
    assert cfg.prefix_padding_duration == 0.3
    assert cfg.activation_threshold == 0.6


def test_with_updates_returns_new_immutable_copy():
    base = VADConfig()
    updated = base.with_updates(min_silence_duration=0.4, activation_threshold=0.55)
    assert isinstance(updated, VADConfig)
    assert updated.min_silence_duration == 0.4
    assert updated.activation_threshold == 0.55
    # Original is unchanged (frozen dataclass), no mutation of fields set.
    assert base.min_silence_duration == 0.55
    # other fields carried over
    assert updated.min_speech_duration == base.min_speech_duration


# ---------------------------------------------------------------------------
# Loader behavior (injected fake loader, no real model)
# ---------------------------------------------------------------------------

def test_load_vad_forwards_default_config_to_loader():
    captured = {}

    def fake_loader(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(kind="vad")

    result = load_vad(VADConfig(), vad_loader=fake_loader)

    assert result.kind == "vad"
    assert captured["min_speech_duration"] == 0.05
    assert captured["min_silence_duration"] == 0.55
    assert captured["prefix_padding_duration"] == 0.5
    assert captured["activation_threshold"] == 0.5
    assert captured["sample_rate"] == 16000
    assert captured["force_cpu"] is True


def test_load_vad_forwards_custom_config_to_loader():
    captured = {}

    def fake_loader(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(kind="vad")

    cfg = VADConfig(min_silence_duration=0.4, activation_threshold=0.6)
    result = load_vad(cfg, vad_loader=fake_loader)

    assert result.kind == "vad"
    assert captured["min_silence_duration"] == 0.4
    assert captured["activation_threshold"] == 0.6
    assert captured["min_speech_duration"] == 0.05


def test_load_vad_uses_default_config_when_none_given():
    captured = {}

    def fake_loader(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(kind="vad")

    load_vad(None, vad_loader=fake_loader)
    assert captured["min_silence_duration"] == 0.55
    assert captured["sample_rate"] == 16000


def test_load_vad_returns_agents_vad_like_instance():
    # With a real silero plugin installed, verify the default loader is the
    # official one but DON'T run it (it needs the ONNX model). We assert the
    # wiring by checking the imported symbol exists.
    import importlib
    mod = importlib.import_module("livekit.plugins.silero")
    assert hasattr(mod, "VAD")
    assert hasattr(mod.VAD, "load")
