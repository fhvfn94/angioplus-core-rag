# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import io
import logging
import os
from time import perf_counter

from aiogram import Bot, types
from google import genai
from google.genai import types as genai_types


logger = logging.getLogger(__name__)

# STT provider selection.
# Supported: "gemini" and "faster_whisper". faster-whisper is loaded lazily and
# only used when STT_PROVIDER=faster_whisper. Keeping the selection isolated here
# so bot/main.py and tests share one resolver.
STT_DEFAULT_PROVIDER = "gemini"
SUPPORTED_STT_PROVIDERS: tuple[str, ...] = ("gemini", "faster_whisper")

# faster-whisper runtime configuration (env-driven). Empty STT_LANGUAGE means
# auto-detect the audio language.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small").strip()
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu").strip()
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip()
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "").strip()
# Optional conditioning prompt for faster-whisper. Empty -> not passed.
WHISPER_INITIAL_PROMPT = os.getenv("WHISPER_INITIAL_PROMPT", "").strip()


def resolve_stt_provider() -> str:
    """Read and validate the STT_PROVIDER env value.

    Defaults to "gemini" (current production behaviour). Fails fast on any
    unknown value so misconfiguration is caught at startup.
    """
    raw = (os.getenv("STT_PROVIDER") or "").strip()
    provider = (raw or STT_DEFAULT_PROVIDER).lower()
    if provider not in SUPPORTED_STT_PROVIDERS:
        supported = ", ".join(SUPPORTED_STT_PROVIDERS)
        raise ValueError(
            f"Unknown STT_PROVIDER={raw!r}. Supported values: {supported}."
        )
    return provider


class STTService:
    def __init__(
        self,
        bot: Bot,
        gemini_api_key: str,
        gemini_audio_model: str,
        download_timeout: int = 60,
        provider: str | None = None,
    ) -> None:
        # Provider selection: explicit arg wins, otherwise resolve from env.
        # Only "gemini" and "faster_whisper" are supported.
        self.provider = (
            resolve_stt_provider() if provider is None else provider
        )
        if self.provider not in SUPPORTED_STT_PROVIDERS:
            raise ValueError(
                f"Unknown STT provider={self.provider!r}. "
                f"Supported values: {', '.join(SUPPORTED_STT_PROVIDERS)}."
            )

        self.bot = bot
        self.gemini_audio_model = gemini_audio_model
        self.download_timeout = download_timeout

        # faster-whisper config (used only by the faster_whisper provider).
        self.whisper_model = WHISPER_MODEL
        self.whisper_device = WHISPER_DEVICE
        self.whisper_compute_type = WHISPER_COMPUTE_TYPE
        # Empty string -> auto-detect audio language.
        self.whisper_language = STT_LANGUAGE or None
        # Optional conditioning prompt for faster-whisper (empty -> None).
        self.whisper_initial_prompt = WHISPER_INITIAL_PROMPT or None
        # Lazily loaded on first faster_whisper request, then cached/reused.
        self._whisper_model = None

        if self.provider == "gemini":
            self.gemini_client = genai.Client(
                api_key=gemini_api_key,
            )
        else:
            # No Gemini key/client is needed for other providers.
            self.gemini_client = None

    def _load_whisper_model(self):
        """Lazily load and cache the faster-whisper model.

        The model is created only on the first voice request and then reused
        across subsequent requests. `faster_whisper` is imported lazily so the
        module does not require the library unless the provider is used.
        """
        if self._whisper_model is None:
            # Imported here to keep module import-time dependency minimal and
            # to let tests substitute a fake without installing the library.
            from faster_whisper import WhisperModel

            logger.info(
                "Loading faster-whisper model: model=%s device=%s "
                "compute_type=%s",
                self.whisper_model,
                self.whisper_device,
                self.whisper_compute_type,
            )
            self._whisper_model = WhisperModel(
                self.whisper_model,
                device=self.whisper_device,
                compute_type=self.whisper_compute_type,
            )
        return self._whisper_model

    def _transcribe_audio_faster_whisper(
        self,
        audio_bytes: bytes,
    ) -> str:
        """Transcribe using the local faster-whisper model (blocking).

        The Telegram OGG/Opus bytes are decoded in-memory via io.BytesIO and
        PyAV inside faster-whisper; no external ffmpeg or temp files are used
        on this step.
        """
        model = self._load_whisper_model()
        audio = io.BytesIO(audio_bytes)

        started = perf_counter()

        segments, _info = model.transcribe(
            audio,
            language=self.whisper_language,
            initial_prompt=self.whisper_initial_prompt or None,
        )

        # Join the text of the returned segments into a single transcription.
        transcription = "".join(segment.text for segment in segments).strip()

        logger.info(
            "faster-whisper transcription took %.2f sec",
            perf_counter() - started,
        )

        if not transcription:
            raise RuntimeError(
                "faster-whisper returned an empty transcription"
            )

        return transcription

    def _transcribe_audio_sync(
        self,
        audio_bytes: bytes,
        mime_type: str,
    ) -> str:
        if self.provider == "faster_whisper":
            # Local provider; mime_type is unused (faster-whisper auto-detects
            # the audio format from the in-memory buffer via PyAV).
            return self._transcribe_audio_faster_whisper(audio_bytes)

        prompt = (
            "Точно расшифруй голосовое сообщение в текст. "
            "В аудио может использоваться русский или английский язык. "
            "Не переводи текст и не отвечай на вопрос. "
            "Верни только расшифровку без комментариев, кавычек, "
            "заголовков и пояснений. "
            "Названия AngioPlus Core, μFR, FFR, CFR и другие "
            "медицинские термины записывай максимально точно."
        )

        started = perf_counter()

        response = self.gemini_client.models.generate_content(
            model=self.gemini_audio_model,
            contents=[
                prompt,
                genai_types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=mime_type,
                ),
            ],
        )

        logger.info(
            "Gemini transcription request took %.2f sec",
            perf_counter() - started,
        )

        transcription = (response.text or "").strip()

        if not transcription:
            raise RuntimeError(
                "Gemini returned an empty transcription"
            )

        return transcription

    async def transcribe_voice(
        self,
        voice: types.Voice,
    ) -> str:
        total_started = perf_counter()
        audio_buffer = io.BytesIO()

        download_started = perf_counter()

        downloaded_file = await self.bot.download(
            voice,
            destination=audio_buffer,
            timeout=self.download_timeout,
        )

        logger.info(
            "Telegram voice download took %.2f sec",
            perf_counter() - download_started,
        )

        if downloaded_file is None:
            raise RuntimeError(
                "Telegram voice file could not be downloaded"
            )

        audio_bytes = audio_buffer.getvalue()

        if not audio_bytes:
            raise RuntimeError(
                "Downloaded Telegram voice file is empty"
            )

        logger.info(
            "Downloaded voice size: %.1f KB",
            len(audio_bytes) / 1024,
        )

        mime_type = voice.mime_type or "audio/ogg"

        transcription = await asyncio.to_thread(
            self._transcribe_audio_sync,
            audio_bytes,
            mime_type,
        )

        logger.info(
            "STT total took %.2f sec",
            perf_counter() - total_started,
        )

        return transcription
