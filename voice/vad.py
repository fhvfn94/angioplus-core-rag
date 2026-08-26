# -*- coding: utf-8 -*-
"""Silero VAD loader/configuration for the voice component (Phase D1.3).

Deliberately small. Wraps the official `livekit.plugins.silero` VAD so the
rest of the code speaks to a plain `agents.vad.VAD` and exposes the tunables we
care about for turn detection in our Russian voice assistant.

This module contains no RAG, TTS, aiogram, or LiveKit Server code.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Callable

from livekit import agents

logger = logging.getLogger(__name__)

# Sensible defaults for a Russian-language voice assistant utterance/end-of-turn
# detection. These mirror the Silero plugin defaults: end-of-turn is declared
# after min_silence_duration of non-speech, with a short speech hold-in and a
# little leading padding so fast onsets/offsets are not chopped.
DEFAULT_VAD_CONFIG = dict(
    min_speech_duration=0.05,
    min_silence_duration=0.55,
    prefix_padding_duration=0.5,
    activation_threshold=0.5,
)


@dataclass(frozen=True)
class VADConfig:
    """Turn-detection tuning passed through to silero.VAD.load()."""

    min_speech_duration: float = 0.05
    min_silence_duration: float = 0.55
    prefix_padding_duration: float = 0.5
    activation_threshold: float = 0.5
    sample_rate: int = 16000
    force_cpu: bool = True

    def with_updates(self, **kwargs) -> "VADConfig":
        """Return a copy with only the given fields overridden."""
        return replace(self, **kwargs)


def load_vad(
    config: VADConfig | None = None,
    *,
    vad_loader: Callable[..., agents.vad.VAD] | None = None,
) -> agents.vad.VAD:
    """Build and load a Silero VAD instance.

    config:
        VADConfig tuning (defaults to DEFAULT_VAD_CONFIG semantics).
    vad_loader:
        Injected loader for tests (defaults to the official
        `livekit.plugins.silero.VAD.load`). This lets tests exercise the
        configuration/loader contract without downloading or running the ONNX
        model.

    Returns an `agents.vad.VAD` ready to create streaming VADStreams.
    """
    cfg = config or VADConfig()

    if vad_loader is None:
        from livekit.plugins import silero

        vad_loader = silero.VAD.load

    logger.info(
        "loading Silero VAD (min_speech=%.2f min_silence=%.2f "
        "prefix_pad=%.2f activation=%.2f sample_rate=%d)",
        cfg.min_speech_duration,
        cfg.min_silence_duration,
        cfg.prefix_padding_duration,
        cfg.activation_threshold,
        cfg.sample_rate,
    )

    return vad_loader(
        min_speech_duration=cfg.min_speech_duration,
        min_silence_duration=cfg.min_silence_duration,
        prefix_padding_duration=cfg.prefix_padding_duration,
        activation_threshold=cfg.activation_threshold,
        sample_rate=cfg.sample_rate,
        force_cpu=cfg.force_cpu,
    )
