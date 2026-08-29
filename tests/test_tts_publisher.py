# -*- coding: utf-8 -*-
"""Unit tests for voice/tts_publisher.py (D2.3) — PCM -> LiveKit frame split.

Only the pure ``pcm_to_frames`` helper is tested (imports only livekit.rtc,
no GPU / network). The live ``publish_tts_audio`` coroutine requires a real
RTC connection and is not exercised here.
"""
from __future__ import annotations

import numpy as np
import pytest

from voice.tts_publisher import pcm_to_frames


def _frames_summary(frames):
    """Return [(num_channels, sample_rate, samples_per_channel)] for each."""
    return [(f.num_channels, f.sample_rate, f.samples_per_channel)
            for f in frames]


def test_pcm_to_frames_preserves_total_samples_and_metadata():
    sr = 48000
    pcm = np.arange(4800, dtype=np.int16).tobytes()  # 4800 samples
    frames = list(pcm_to_frames(pcm, sr))
    total = sum(f.samples_per_channel for f in frames)
    assert total == 4800
    assert all(f.sample_rate == sr for f in frames)
    assert all(f.num_channels == 1 for f in frames)
    # default chunk ~= 100ms at 48kHz = 4800 samples -> one frame
    assert [f.samples_per_channel for f in frames] == [4800]
def test_pcm_to_frames_custom_chunk_size():
    pcm = np.arange(1000, dtype=np.int16).tobytes()
    frames = list(pcm_to_frames(pcm, 48000, samples_per_frame=300))
    sizes = [f.samples_per_channel for f in frames]
    assert sizes == [300, 300, 300, 100]
    assert sum(sizes) == 1000


def test_pcm_to_frames_last_frame_is_remainder():
    pcm = np.arange(250, dtype=np.int16).tobytes()
    frames = list(pcm_to_frames(pcm, 48000, samples_per_frame=100))
    sizes = [f.samples_per_channel for f in frames]
    assert sizes == [100, 100, 50]


def test_pcm_to_frames_invalid_rate_raises():
    with pytest.raises(ValueError):
        list(pcm_to_frames(b"\x00\x00" * 4, 0))


def test_pcm_to_frames_empty_pcm_yields_nothing():
    frames = list(pcm_to_frames(b"", 48000))
    assert frames == []

