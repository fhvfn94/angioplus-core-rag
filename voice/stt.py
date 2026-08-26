# -*- coding: utf-8 -*-
"""LiveKit faster-whisper STT (non-streaming) for the voice component.

Phase D1.2: prove
    LiveKit AudioBuffer
      -> audio normalization / resampling (mono 16 kHz WAV)
      -> faster-whisper (small / int8 / CPU)
      -> LiveKit FINAL_TRANSCRIPT SpeechEvent

Architecture decision (D1.1):
    - a custom non-streaming FasterWhisperSTT(agents.stt.STT) subclass that
      implements the single abstract hook `_recognize_impl`.
    - In a later phase a LiveKit StreamAdapter will wrap this STT together
      with a Silero VAD to provide end-of-turn/streaming behaviour.

This module intentionally:
    - does NOT touch RAG (/ask) or Qdrant,
    - does NOT implement TTS / SIP / fine-tuning,
    - loads the faster-whisper model lazily and caches it,
    - keeps faster-whisper's blocking inference off the asyncio loop via
      asyncio.to_thread.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import uuid
from typing import List

import numpy as np

from livekit import agents, rtc
from livekit.agents.stt import STT, STTCapabilities, SpeechData, SpeechEvent
from livekit.agents.stt.stt import SpeechEventType
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given, merge_frames

logger = logging.getLogger(__name__)

# Whisper is a 16 kHz mono model; normalize every input buffer to this.
WHISPER_SAMPLE_RATE = 16000

# faster-whisper runtime configuration (env-driven), same defaults/meaning as
# the existing Telegram STT in bot/services/stt.py.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small").strip()
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu").strip()
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip()
# Empty -> auto-detect audio language.
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "").strip()
# Optional conditioning prompt for faster-whisper. Empty -> not passed.
WHISPER_INITIAL_PROMPT = os.getenv("WHISPER_INITIAL_PROMPT", "").strip()


def _downmix_to_mono(frame: rtc.AudioFrame) -> rtc.AudioFrame:
    """Average interleaved channels into a single mono int16 frame."""
    channels = frame.num_channels
    data = np.frombuffer(frame.data, dtype=np.int16)
    total_samples = data.size // channels
    mono = (
        data.reshape(total_samples, channels)
        .mean(axis=1)
        .round()
        .astype(np.int16)
    )
    return rtc.AudioFrame(
        data=mono.tobytes(),
        sample_rate=frame.sample_rate,
        num_channels=1,
        samples_per_channel=len(mono),
    )


def _resample_to(mono_frame: rtc.AudioFrame, target_rate: int) -> rtc.AudioFrame:
    """Resample a mono frame to target_rate using the LiveKit resampler."""
    if mono_frame.sample_rate == target_rate:
        return mono_frame
    resampler = rtc.AudioResampler(
        mono_frame.sample_rate,
        target_rate,
        num_channels=1,
        quality=rtc.AudioResamplerQuality.MEDIUM,
    )
    out_frames: List[rtc.AudioFrame] = resampler.push(mono_frame)
    out_frames.extend(resampler.flush())
    return merge_frames(out_frames)


def _buffer_to_mono16k_wav(buffer: "agents.utils.AudioBuffer") -> bytes:
    """Normalize a LiveKit AudioBuffer to mono 16 kHz WAV bytes.

    Handles a single rtc.AudioFrame or a list of frames, downmixes to mono
    when needed, and resamples any input sample rate to 16 kHz (Whisper's
    native rate). The returned bytes are a WAV file faster-whisper can decode.
    """
    frame = merge_frames(buffer)  # single rate + channel count

    if frame.num_channels != 1:
        frame = _downmix_to_mono(frame)

    if frame.sample_rate != WHISPER_SAMPLE_RATE:
        frame = _resample_to(frame, WHISPER_SAMPLE_RATE)

    return frame.to_wav_bytes()


class FasterWhisperSTT(STT):
    """Non-streaming faster-whisper STT implementing the LiveKit STT hook.

    Subclasses agents.stt.STT; the only abstract method is `_recognize_impl`.
    """

    def __init__(
        self,
        *,
        model: str = WHISPER_MODEL,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
        language: str | None = STT_LANGUAGE or None,
        initial_prompt: str | None = WHISPER_INITIAL_PROMPT or None,
        _model_factory=None,
    ) -> None:
        super().__init__(
            capabilities=STTCapabilities(
                streaming=False,
                interim_results=False,
                offline_recognize=True,
            )
        )
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._initial_prompt = initial_prompt
        # Injectable model factory (mainly for tests; defaults to the real
        # faster_whisper.WhisperModel).
        self._model_factory = _model_factory
        self._model = None  # lazily loaded and cached

    # --- LiveKit STT metadata ---------------------------------------------

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "faster-whisper"

    # --- model lifecycle ---------------------------------------------------

    def _load_model(self):
        """Lazily load and cache the faster-whisper model.

        Started from the event loop only via to_thread; the model is reused
        across all utterances.
        """
        if self._model is None:
            if self._model_factory is not None:
                logger.info(
                    "using injected faster-whisper model factory "
                    "(model=%s device=%s compute_type=%s)",
                    self._model_name, self._device, self._compute_type,
                )
                model = self._model_factory(
                    self._model_name,
                    self._device,
                    self._compute_type,
                )
            else:
                # Imported lazily so importing voice.stt does not require the
                # library when the provider is unused.
                from faster_whisper import WhisperModel

                logger.info(
                    "loading faster-whisper model: model=%s device=%s "
                    "compute_type=%s",
                    self._model_name, self._device, self._compute_type,
                )
                model = WhisperModel(
                    self._model_name,
                    device=self._device,
                    compute_type=self._compute_type,
                )
            self._model = model
        return self._model

    @property
    def is_model_loaded(self) -> bool:
        """Whether the model has been loaded yet (for tests)."""
        return self._model is not None

    # --- audio normalization / blocking transcribe --------------------------

    def _transcribe_blocking(
        self,
        audio: bytes,
        *,
        language: str | None,
        initial_prompt: str | None,
    ) -> str:
        """Blocking faster-whisper call. Run via asyncio.to_thread."""
        model = self._load_model()
        wav = io.BytesIO(audio)

        segments, _info = model.transcribe(
            wav,
            language=language,
            initial_prompt=initial_prompt,
            beam_size=1,
            best_of=1,
            temperature=0,
        )

        transcription = "".join(segment.text for segment in segments).strip()
        if not transcription:
            raise ValueError(
                "faster-whisper returned an empty transcription"
            )
        return transcription

    # --- LiveKit STT hook ---------------------------------------------------

    async def _recognize_impl(
        self,
        buffer: "agents.utils.AudioBuffer",
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: "agents.APIConnectOptions",
    ) -> SpeechEvent:
        request_id = str(uuid.uuid4())

        wav_bytes = await asyncio.to_thread(_buffer_to_mono16k_wav, buffer)

        # Explicit language argument wins over the configured default.
        if is_given(language) and language is not None:
            use_language = language
        else:
            use_language = self._language

        async def _run() -> str:
            return await asyncio.to_thread(
                self._transcribe_blocking,
                wav_bytes,
                language=use_language,
                initial_prompt=self._initial_prompt,
            )

        alternative = SpeechData(
            language=agents.LanguageCode(use_language or "auto"),
            text=await _run(),
        )

        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            request_id=request_id,
            alternatives=[alternative],
        )
