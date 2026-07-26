# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import io
import logging
from time import perf_counter

from aiogram import Bot, types
from google import genai
from google.genai import types as genai_types


logger = logging.getLogger(__name__)


class STTService:
    def __init__(
        self,
        bot: Bot,
        gemini_api_key: str,
        gemini_audio_model: str,
        download_timeout: int = 60,
    ) -> None:
        self.bot = bot
        self.gemini_audio_model = gemini_audio_model
        self.download_timeout = download_timeout
        self.gemini_client = genai.Client(
            api_key=gemini_api_key,
        )

    def _transcribe_audio_sync(
        self,
        audio_bytes: bytes,
        mime_type: str,
    ) -> str:
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