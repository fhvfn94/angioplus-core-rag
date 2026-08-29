# -*- coding: utf-8 -*-
"""Unit tests for voice/tts_client.py (remote Qwen3-TTS HTTP client, D2.3).

No GPU / qwen_tts / torch / LiveKit is used. A fake httpx.AsyncClient returns
canned WAV blobs (built with stdlib ``wave``). Covers:
  - WAV -> mono int16 PCM decoding,
  - downmix (stereo -> mono),
  - resampling to the configured output rate,
  - timeout / HTTP-error / invalid-audio handling.
"""
from __future__ import annotations

import asyncio
import io
import json
import wave

import numpy as np
import pytest

import httpx

from voice.tts_client import (
    TTSAudio,
    TTSClient,
    TTSClientError,
    TTSInvalidAudioError,
    TTSResponseError,
    TTSUnavailableError,
)


def wav_bytes(pcm: np.ndarray, sr: int = 24000, channels: int = 1) -> bytes:
    """Build a 16-bit PCM WAV blob."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.astype("<i2").tobytes())
    return buf.getvalue()


class FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200, json_body=None):
        self.content = content
        self.status_code = status_code
        self._json = json_body

    def json(self):
        if self._json is not None:
            return self._json
        return json.loads(self.content.decode("utf-8") if self.content else "{}")


class FakeTransportClient:
    def __init__(self, handler):
        self.handler = handler

    async def post(self, url, json):
        return await self.handler(url, json)

    async def aclose(self):
        pass


def _client(handler=None, sample_rate=48000):
    if handler is None:
        async def ok(url, payload):
            return FakeResponse(wav_bytes(np.zeros(4800, dtype=np.int16)))
        handler = ok
    return TTSClient(
        base_url="http://127.0.0.1:8001",
        sample_rate=sample_rate,
        http_client=FakeTransportClient(handler),
    )


def _synth(client, text):
    return asyncio.run(client.synthesize(text))


# --- pure PCM helpers ------------------------------------------------------

def test_resample_same_rate_is_identity():
    from voice.tts_client import _resample_pcm
    pcm = np.array([-100, 0, 100, 200], dtype=np.int16)
    assert np.array_equal(_resample_pcm(pcm, 24000, 24000), pcm)


def test_resample_up_has_correct_length():
    from voice.tts_client import _resample_pcm
    out = _resample_pcm(np.arange(1000, dtype=np.int16), 24000, 48000)
    assert out.size == 2000


def test_resample_invalid_rate_raises():
    from voice.tts_client import _resample_pcm
    with pytest.raises(ValueError):
        _resample_pcm(np.zeros(10, dtype=np.int16), 0, 48000)


def test_downmix_to_mono_averages_channels():
    from voice.tts_client import _downmix_to_mono
    inter = np.array([100, 200, 300, 400], dtype=np.int16)
    assert list(_downmix_to_mono(inter, 2)) == [150, 350]


def test_raw_to_int16_u8_centered():
    from voice.tts_client import _raw_to_int16
    # (u - 128) * 256
    out = _raw_to_int16(bytes([0, 128, 255]), 1, 1)
    assert list(out) == [-32768, 0, 32512]


# --- client WAV decode -----------------------------------------------------

def test_synthesize_decodes_wav_to_mono_pcm():
    src = np.array([0, 1000, -1000, 2000, -2000, 3000], dtype=np.int16)

    async def handler(url, payload):
        assert url == "http://127.0.0.1:8001/tts"
        assert payload["speaker"] == "Vivian"
        assert payload["language"] == "Russian"
        assert payload["text"] == "привет"
        return FakeResponse(wav_bytes(src, sr=24000, channels=1))

    audio = _synth(_client(handler), "привет")
    assert isinstance(audio, TTSAudio)
    assert audio.num_channels == 1
    assert audio.sample_rate == 48000          # resampled 24k -> configured 48k
    assert audio.duration_seconds == pytest.approx(12 / 48000)


def test_synthesize_stereo_wav_downmixed_to_mono():
    wav = wav_bytes(np.array([1000, 3000], dtype=np.int16), sr=24000, channels=2)

    async def handler(url, payload):
        return FakeResponse(wav)

    audio = asyncio.run(TTSClient(
        base_url="http://x",
        sample_rate=24000,
        http_client=FakeTransportClient(handler),
    ).synthesize("x"))
    assert audio.sample_rate == 24000
    assert len(audio.pcm_bytes) // 2 == 1
    assert np.frombuffer(audio.pcm_bytes, dtype=np.int16)[0] == 2000


def test_synthesize_http_error_raises_response_error():
    async def handler(url, payload):
        return FakeResponse(b"", status_code=503, json_body={"detail": "busy"})
    with pytest.raises(TTSResponseError) as exc:
        _synth(_client(handler), "x")
    assert exc.value.status_code == 503


def test_synthesize_timeout_raises_timeout_error():
    async def handler(url, payload):
        raise httpx.ReadTimeout("t")
    client = _client(handler)
    with pytest.raises(TTSClientError):
        _synth(client, "x")


def test_synthesize_connection_error_raises_unavailable():
    async def handler(url, payload):
        raise httpx.ConnectError("refused")
    client = _client(handler)
    with pytest.raises(TTSUnavailableError):
        _synth(client, "x")


def test_synthesize_invalid_wav_raises_invalid_audio():
    async def handler(url, payload):
        return FakeResponse(b"this is not a wav")
    with pytest.raises(TTSInvalidAudioError):
        _synth(_client(handler), "x")


def test_tts_audio_duration_metadata():
    audio = TTSAudio(pcm_bytes=b"\x00\x00" * 48000, sample_rate=48000)
    assert audio.duration_seconds == pytest.approx(1.0)
