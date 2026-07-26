# -*- coding: utf-8 -*-
from __future__ import annotations


import logging
import os

import httpx
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import CommandStart
from services.stt import STTService
from time import perf_counter
import asyncio


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
RAG_API_URL = os.getenv("RAG_API_URL", "http://rag:8000/ask").strip()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_AUDIO_MODEL = os.getenv(
    "GEMINI_AUDIO_MODEL",
    "gemini-2.5-flash",
).strip()

MAX_VOICE_DURATION_SECONDS = int(
    os.getenv("MAX_VOICE_DURATION_SECONDS", "120")
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")


bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()
stt_service = STTService(
    bot=bot,
    gemini_api_key=GEMINI_API_KEY,
    gemini_audio_model=GEMINI_AUDIO_MODEL,
)



@dp.message(CommandStart())
async def start(message: types.Message) -> None:
    await message.answer(
        "Привет. Я MVP ассистент поддержки AngioPlus Core.\n\n"
        "Задай вопрос текстом или отправь голосовое сообщение."
    )


def format_sources(sources: list[dict]) -> str:
    if not sources:
        return ""

    lines = ["\n\nИсточники:"]

    for source in sources:
        file_name = source.get("file_name", "")
        section = source.get("section", "")
        page_start = source.get("page_start", "")
        page_end = source.get("page_end", "")
        score = source.get("score", "")

        if file_name.endswith(".xlsx"):
            lines.append(
                f"{file_name}\n"
                f"Раздел: {section}\n"
                f"Строка FAQ: {page_start}, score {score}"
            )
        else:
            lines.append(
                f"{file_name}\n"
                f"{section}\n"
                f"pages {page_start} to {page_end}, score {score}"
            )

    return "\n\n".join(lines)


async def ask_rag(question: str) -> dict:
    async with httpx.AsyncClient(timeout=90) as client:
        response = await client.post(
            RAG_API_URL,
            json={
                "question": question,
                "top_k": 5,
            },
        )
        response.raise_for_status()
        return response.json()


async def send_rag_answer(
    message: types.Message,
    question: str,
) -> None:
    question = question.strip()

    if not question:
        await message.answer(
            "Не удалось определить текст вопроса. "
            "Попробуй записать голосовое сообщение ещё раз."
        )
        return

    await message.answer("Секунду, проверяю базу знаний...")

    try:
        data = await ask_rag(question)

    except httpx.HTTPStatusError as exc:
        logger.exception("RAG API returned an HTTP error")

        status_code = exc.response.status_code
        await message.answer(
            f"RAG API вернул ошибку HTTP {status_code}."
        )
        return

    except Exception as exc:
        logger.exception("RAG API request failed")
        await message.answer(
            f"Ошибка при обращении к RAG API: {exc}"
        )
        return

    answer = data.get(
        "answer",
        "Информация не найдена в документации.",
    )
    sources = data.get("sources", [])

    final_text = answer + format_sources(sources)

    if len(final_text) > 3900:
        final_text = (
            final_text[:3900]
            + "\n\nОтвет обрезан из-за лимита Telegram."
        )

    await message.answer(final_text)


@dp.message(F.voice)
async def handle_voice(message: types.Message) -> None:
    total_started = perf_counter()


    voice = message.voice

    if voice is None:
        await message.answer(
            "Не удалось получить голосовое сообщение."
        )
        return

    if voice.duration > MAX_VOICE_DURATION_SECONDS:
        await message.answer(
            "Голосовое сообщение слишком длинное.\n"
            f"Максимальная длительность: "
            f"{MAX_VOICE_DURATION_SECONDS} секунд."
        )
        return

    status_message = await message.answer(
        "🎧 Распознаю голосовое сообщение..."
    )

    transcription_started = perf_counter()

    try:
        transcription = await stt_service.transcribe_voice(voice)

        logger.info(
            "Voice transcription took %.2f sec",
            perf_counter() - transcription_started,
        )

    except Exception as exc:
        logger.exception("Voice transcription failed")

        error_text = str(exc).lower()

        if (
            "429" in error_text
            or "resource_exhausted" in error_text
            or "quota" in error_text
        ):
            await status_message.edit_text(
                "Не удалось распознать голосовое сообщение: "
                "превышен лимит Gemini API. "
                "Попробуй немного позже."
            )
        else:
            await status_message.edit_text(
                "Не удалось распознать голосовое сообщение. "
                "Попробуй записать его ещё раз."
            )

        return

    logger.info(
        "Voice message transcribed: user_id=%s, text=%r",
        message.from_user.id if message.from_user else None,
        transcription,
    )

    await status_message.edit_text(
        f"📝 Распознано:\n{transcription}"
    )

    rag_started = perf_counter()

    await send_rag_answer(
        message=message,
        question=transcription,
    )

    logger.info(
        "RAG answer took %.2f sec",
        perf_counter() - rag_started,
    )

    logger.info(
        "Full voice request took %.2f sec",
        perf_counter() - total_started,
    )


@dp.message(F.text)
async def handle_text_question(
    message: types.Message,
) -> None:
    question = (message.text or "").strip()

    if not question:
        await message.answer(
            "Пришли вопрос текстом или голосовым сообщением."
        )
        return

    await send_rag_answer(
        message=message,
        question=question,
    )


@dp.message()
async def handle_unsupported_message(
    message: types.Message,
) -> None:
    await message.answer(
        "Пришли вопрос текстом или голосовым сообщением."
    )


async def main() -> None:
    logger.info("Starting Telegram bot")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())