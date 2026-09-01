# -*- coding: utf-8 -*-
"""Async HTTP client for the remote Qwen3-TTS service (Phase D2.3).

The Qwen3-TTS 0.6B model does NOT run inside the local LiveKit voice worker
(transformers pin conflict + it needs a CUDA GPU). Instead it runs on a
remote GPU (e.g. a Vast.ai RTX 4090) behind a tiny FastAPI service
(scripts/tts/tts_server.py). This module is the ONLY boundary between the local
LiveKit worker and that GPU TTS service.

Responsibilities:
  - POST {text, speaker, language} to the remote /tts endpoint,
  - receive a ready-made WAV audio blob,
  - decode WAV -> mono int16 PCM and resample to a LiveKit-friendly rate
    (default 48 kHz),
  - surface a typed ``TTSAudio`` result with raw PCM + sample-rate metadata,
  - raise typed errors (``TTSUnavailableError`` / ``TTSTimeoutError`` /
    ``TTSResponseError``) so the caller can fail soft and keep the voice
    worker alive.

Design constraints:
  - no heavy imports (no torch / qwen_tts / transformers): only stdlib (wave,
    io) + numpy + httpx, all already in the project / livekit venv.
  - the caller decides whether to publish the returned PCM into LiveKit; this
    module does NOT touch rtc (kept transport-only, easy to unit test).
"""
from __future__ import annotations

import io
import logging
import wave
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np

logger = logging.getLogger(__name__)


class TTSClientError(Exception):
    """Base error for all remote-TTS client failures."""


class TTSUnavailableError(TTSClientError):
    """The HTTP service could not be reached (connection refused / DNS)."""


class TTSTimeoutError(TTSClientError):
    """The remote TTS service did not respond within the timeout window."""


class TTSResponseError(TTSClientError):
    """The remote service replied with a non-2xx status or a valid error."""

    def __init__(
        self,
        status_code: int,
        url: str,
        detail: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.url = url
        self.detail = detail
        super().__init__(
            f"TTS service returned HTTP {status_code} for POST {url}"
            + (f": {detail}" if detail else "")
        )


class TTSInvalidAudioError(TTSClientError):
    """The service returned bytes that are not a valid mono PCM WAV we can use."""


@dataclass(frozen=True)
class TTSAudio:
    """A complete synthesized utterance as mono int16 PCM.

    Attributes
    ----------
    pcm_bytes:
        interleaved mono int16 little-endian PCM samples.
    sample_rate:
        sample rate of ``pcm_bytes`` (already resampled to the requested
        LiveKit rate by the client).
    num_channels:
        always 1 (mono) - LiveKit audio-source convention.
    duration_seconds:
        ``len(pcm)//2 / sample_rate``.
    """
    pcm_bytes: bytes
    sample_rate: int
    num_channels: int = 1

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return (len(self.pcm_bytes) // 2) / float(self.sample_rate)


class TTSClient:
    """Thin async HTTP client around the remote Qwen3-TTS /tts endpoint.

    One instance owns its transport (an httpx.AsyncClient). Close with
    ``aclose()`` when done.
    """

    def __init__(
        self,
        base_url: str,
        speaker: str = "Vivian",
        language: str = "Russian",
        timeout_seconds: int = 120,
        sample_rate: int = 48000,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.speaker = speaker
        self.language = language
        self.timeout_seconds = timeout_seconds
        self.sample_rate = sample_rate
        self._http_client = http_client

    def _client(self) -> httpx.AsyncClient:
        if self._http_client is not None:
            return self._http_client
        timeout = httpx.Timeout(
            connect=10.0,
            read=float(self.timeout_seconds),
            write=10.0,
            pool=10.0,
        )
        return httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        """Close the internally-owned httpx client (no-op if injected)."""
        if self._http_client is not None:
            return  # caller owns the injected client
        client = getattr(self, "_owns_client", None)
        if client is not None:
            await client.aclose()

    async def __aenter__(self) -> "TTSClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # -- request ------------------------------------------------------------

    async def synthesize(self, text: str) -> TTSAudio:
        """Send ``text`` to the remote service and return synthesized audio.

        Raises TTSClientError subclasses on transport/timeout/HTTP failure so
        the voice worker can fail soft without crashing.
        """
        if not text or not str(text).strip():
            raise ValueError("text must be a non-empty string")

        url = f"{self.base_url}/tts"
        payload = {
            "text": text,
            "speaker": self.speaker,
            "language": self.language,
        }

        client = self._client()
        self._owns_client = client

        try:
            response = await client.post(url, json=payload)
        except httpx.TimeoutException as exc:
            raise TTSTimeoutError(
                f"TTS request timed out after {self.timeout_seconds}s "
                f"(POST {url})"
            ) from exc
        except httpx.HTTPError as exc:
            raise TTSUnavailableError(
                f"TTS service unreachable at {url}: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise TTSResponseError(
                status_code=response.status_code,
                url=url,
                detail=_safe_error_detail(response),
            )

        wav_bytes = response.content
        if not wav_bytes:
            raise TTSInvalidAudioError(
                "TTS service returned an empty body (no audio)"
            )

        return self._decode_wav(wav_bytes)

    # -- WAV -> PCM logic (pure, unit-testable) ------------------------------

    def _decode_wav(self, wav_bytes: bytes) -> TTSAudio:
        """Decode a mono PCM WAV blob into TTSAudio (mono int16 PCM).

        The remote service emits mono 16-bit PCM WAV. We read framerate,
        channels, width and raw frames, then downmix to mono if needed and
        resample to the configured ``sample_rate``.
        """
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
                src_rate = wav.getframerate()
                n_channels = wav.getnchannels()
                sampwidth = wav.getsampwidth()
                frames = wav.readframes(wav.getnframes())
        except (wave.Error, EOFError, OSError) as exc:
            raise TTSInvalidAudioError(
                f"TTS service returned invalid WAV audio: {exc}"
            ) from exc

        if src_rate <= 0:
            raise TTSInvalidAudioError(
                "TTS service returned WAV with invalid sample rate"
            )
        if sampwidth not in (1, 2):
            # The service always emits int16; guard against unsupported widths.
            raise TTSInvalidAudioError(
                f"TTS service returned unsupported sample width {sampwidth}"
            )

        pcm = _raw_to_int16(frames, sampwidth, n_channels)

        if n_channels != 1:
            pcm = _downmix_to_mono(pcm, n_channels)

        if src_rate != self.sample_rate:
            pcm = _resample_pcm(pcm, src_rate, self.sample_rate)

        return TTSAudio(
            pcm_bytes=pcm.tobytes(),
            sample_rate=self.sample_rate,
            num_channels=1,
        )


# ---------------------------------------------------------------------------
# Pure PCM helpers (unit-testable without LiveKit / torch)
# ---------------------------------------------------------------------------

def _raw_to_int16(frames: bytes, sampwidth: int, n_channels: int) -> np.ndarray:
    """Convert raw WAV frames to an int16 numpy array of interleaved samples."""
    if sampwidth == 2:
        # native little-endian int16 already
        arr = np.frombuffer(frames, dtype="<i2").astype(np.int16)
        # trim to a whole number of channel-groups
        n = arr.size - (arr.size % n_channels)
        return arr[:n]
    # sampwidth == 1: unsigned 8-bit -> shift to signed int16
    arr = np.frombuffer(frames, dtype=np.uint8)
    n = arr.size - (arr.size % n_channels)
    arr = arr[:n].astype(np.int16)
    return (arr - 128) * 256


def _downmix_to_mono(samples: np.ndarray, n_channels: int) -> np.ndarray:
    """Average interleaved channel groups into a single mono track (int16)."""
    total = samples.size // n_channels
    stereo = samples[: total * n_channels].reshape(total, n_channels)
    return stereo.mean(axis=1).round().astype(np.int16)


def _resample_pcm(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linearly resample an int16 PCM array from src_rate to dst_rate.

    Pure/numpy-only so TTS (and tests) do not depend on LiveKit resamplers.
    Linear interpolation is adequate for a voice reply; better quality is a
    later optimisation (not a Phase D2.3 goal).
    """
    if dst_rate <= 0 or src_rate <= 0:
        raise ValueError("sample rates must be positive")
    if src_rate == dst_rate:
        return pcm
    n_in = pcm.size
    if n_in == 0:
        return pcm
    n_out = int(round(n_in * dst_rate / src_rate))
    if n_out <= 0:
        return np.zeros(0, dtype=np.int16)
    x_float = pcm.astype(np.float32)
    xp = np.linspace(0.0, 1.0, n_in)
    fp = np.linspace(0.0, 1.0, n_out)
    resampled = np.interp(fp, xp, x_float)
    return np.rint(resampled).astype(np.int16)


def _safe_error_detail(response: httpx.Response) -> str | None:
    """Extract a short, safe error detail from a JSON error response."""
    try:
        data = response.json()
    except (ValueError, Exception):  # noqa: BLE001
        return None
    if isinstance(data, dict):
        detail = data.get("detail")
        if isinstance(detail, str):
            return detail[:300]
    return None
