# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import logging

import httpx
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
RAG_API_URL = os.getenv("RAG_API_URL", "http://rag:8000/ask")

logging.basicConfig(level=logging.INFO)

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()


@dp.message(CommandStart())
async def start(message: types.Message) -> None:
    await message.answer(
        "Привет. Я MVP ассистент поддержки AngioPlus Core.\n\n"
        "Задай вопрос по документации."
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


@dp.message()
async def handle_question(message: types.Message) -> None:
    question = (message.text or "").strip()

    if not question:
        await message.answer("Пришли вопрос текстом.")
        return

    await message.answer("Секунду, проверяю базу знаний...")

    try:
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                RAG_API_URL,
                json={"question": question, "top_k": 5},
            )
            response.raise_for_status()
            data = response.json()

    except Exception as exc:
        logging.exception("RAG API request failed")
        await message.answer(f"Ошибка при обращении к RAG API: {exc}")
        return

    answer = data.get("answer", "Not found in documentation.")
    sources = data.get("sources", [])

    final_text = answer + format_sources(sources)

    if len(final_text) > 3900:
        final_text = final_text[:3900] + "\n\nОтвет обрезан из за лимита Telegram."

    await message.answer(final_text)


async def main() -> None:
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())