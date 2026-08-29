# -*- coding: utf-8 -*-
"""Publish generated Qwen3-TTS audio into a LiveKit room (Phase D2.3).

The local LiveKit voice worker is a *programmatic* participant: it does NOT
use an AgentSession, so synthesized audio is played by creating a plain
``rtc.AudioSource`` + ``LocalAudioTrack``, publishing that track into the
room, pushing the mono PCM frames captured by the remote TTS service, and
flushing the source so LiveKit streams them to subscribed listeners.

This module keeps the LiveKit-specific publishing away from the transport-only
``voice.tts_client``, and exposes a pure ``pcm_to_frames`` helper so the
frame-splitting logic is unit-testable without a live RTC connection.
"""
from __future__ import annotations

import logging
from typing import Iterable

from livekit import rtc

from voice.tts_client import TTSAudio

logger = logging.getLogger(__name__)

# ~100 ms per frame is a comfortable, LiveKit-friendly push granularity.
SAMPLES_PER_FRAME_DIVISOR = 10


def pcm_to_frames(
    pcm_bytes: bytes,
    sample_rate: int,
    num_channels: int = 1,
    samples_per_frame: int | None = None,
) -> Iterable[rtc.AudioFrame]:
    """Split mono int16 PCM into equally sized rtc.AudioFrame chunks.

    ``samples_per_frame`` defaults to ``sample_rate // 10`` (~100 ms). The last
    frame is whatever remains. Total sample count is preserved.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    frame_size = samples_per_frame or max(1, sample_rate // SAMPLES_PER_FRAME_DIVISOR)
    frame_size = max(1, int(frame_size))

    total_samples = len(pcm_bytes) // 2
    offset = 0
    while offset < total_samples:
        n = min(frame_size, total_samples - offset)
        chunk = pcm_bytes[offset * 2 : (offset + n) * 2]
        yield rtc.AudioFrame(
            data=chunk,
            sample_rate=sample_rate,
            num_channels=num_channels,
            samples_per_channel=n,
        )
        offset += n


async def publish_tts_audio(
    participant: rtc.LocalParticipant,
    audio: TTSAudio,
    track_name: str = "tts-answer",
    options: rtc.TrackPublishOptions | None = None,
) -> rtc.TrackPublication:
    """Publish a synthesized TTSAudio utterance into the room as a spoken track.

    Creates an ``rtc.AudioSource`` at the audio's sample rate, wraps it in a
    LocalAudioTrack, publishes it, pushes all PCM frames, then flushes the
    source so LiveKit streams the whole clip to subscribing listeners.

    Returns the resulting ``TrackPublication`` so the caller can unpublish it
    later (e.g. after playback or on disconnect).
    """
    source = rtc.AudioSource(
        sample_rate=audio.sample_rate,
        num_channels=audio.num_channels,
    )
    track = rtc.LocalAudioTrack.create_audio_track(track_name, source)
    publication = await participant.publish_track(
        track,
        options=options or rtc.TrackPublishOptions(),
    )
    logger.info(
        "published TTS track %s (audio %.2fs, %d Hz)", track_name,
        audio.duration_seconds, audio.sample_rate,
    )

    for frame in pcm_to_frames(audio.pcm_bytes, audio.sample_rate):
        source.capture_frame(frame)
    source.flush()

    return publication
