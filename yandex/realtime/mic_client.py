#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import base64
import json
import os
import time

import aiohttp
from dotenv import load_dotenv

from yandex_ai_studio_sdk._experimental.audio.microphone import AsyncMicrophone
from yandex_ai_studio_sdk._experimental.audio.out import AsyncAudioOut


# ============================================================
# ENV
# ============================================================

load_dotenv()

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")

if not YANDEX_API_KEY:
    raise RuntimeError("В .env отсутствует YANDEX_API_KEY")

if not YANDEX_FOLDER_ID:
    raise RuntimeError("В .env отсутствует YANDEX_FOLDER_ID")


# ============================================================
# REALTIME CONFIG
# ============================================================

IN_RATE = 44100
OUT_RATE = 44100

# Стандартный голос SpeechKit.
VOICE = "dasha"

MODEL = f"gpt://{YANDEX_FOLDER_ID}/speech-realtime-250923"
VECTOR_STORE_ID = "fvt8mt12qgtftnl2meh9"

WSS_URL = (
    "wss://ai.api.cloud.yandex.net/v1/realtime"
    f"?model={MODEL}"
)

HEADERS = {
    "Authorization": f"Api-Key {YANDEX_API_KEY}"
}

SYSTEM_PROMPT = """
Ты — голосовой ассистент технической поддержки медицинского
программного обеспечения AngioPlus Core.

Разговаривай на русском языке.
Отвечай естественно, кратко и по делу.

На любые вопросы, связанные с AngioPlus Core, его функциями,
настройкой, использованием, ошибками, требованиями,
интерфейсом и технической поддержкой, обязательно используй
функцию file_search перед формированием ответа.

Отвечай на технические вопросы об AngioPlus Core только на
основании информации, найденной через file_search.

Не придумывай факты и не дополняй документацию своими
предположениями.

Если после поиска информации недостаточно, скажи:
"Такой информации нет в имеющейся документации."

Если найденная информация противоречит сама себе,
отдавай приоритет официальной инструкции IFU.

Не принимай клинические решения за пользователя.
"""


# ============================================================
# HELPERS
# ============================================================

def b64_encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64_decode(data: str) -> bytes:
    return base64.b64decode(data)


# ============================================================
# SESSION
# ============================================================

async def setup_session(ws: aiohttp.ClientWebSocketResponse):
    event = {
        "type": "session.update",
        "session": {
            "instructions": SYSTEM_PROMPT,
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": IN_RATE,
                    },
                    "languages": ["ru-RU"],
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "silence_duration_ms": 900,
                    },
                },
                "output": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": OUT_RATE,
                    },
                    "voice": VOICE,
                },
            },
            "tools": [
                {
                    "type": "function",
                    "name": "file_search",
                    "description": VECTOR_STORE_ID,
                    "parameters": {},
                }
            ],
        },
    }

    await ws.send_json(event)


# ============================================================
# GREETING
# ============================================================

async def send_greeting(ws: aiohttp.ClientWebSocketResponse):
    await ws.send_json(
        {
            "type": "response.create",
            "response": {
                "instructions": (
                    'Скажи пользователю только эту фразу: '
                    '"Привет. Чем я могу вам помочь?"'
                ),
                "output_modalities": ["audio"],
            },
        }
    )



# ============================================================
# MICROPHONE -> YANDEX
# ============================================================

async def uplink(ws: aiohttp.ClientWebSocketResponse):
    print("🎤 Микрофон запущен.")
    print("   Говорите естественно — Yandex сам определяет конец фразы.\n")

    microphone = AsyncMicrophone(
        samplerate=IN_RATE,
        blocksize=882,
    )

    try:
        async for audio_chunk in microphone:
            await ws.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": b64_encode(audio_chunk),
                }
            )
    finally:
        await microphone.stop()


# ============================================================
# YANDEX -> SPEAKERS
# ============================================================

async def downlink(ws: aiohttp.ClientWebSocketResponse):
    first_audio_time = None
    speech_end_time = None

    async with AsyncAudioOut(
        samplerate=OUT_RATE,
        blocksize=882,
    ) as audio_out:

        async for msg in ws:

            if msg.type == aiohttp.WSMsgType.ERROR:
                print("❌ WebSocket error:", ws.exception())
                break

            if msg.type != aiohttp.WSMsgType.TEXT:
                continue

            try:
                event = json.loads(msg.data)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "")

            if event_type == "session.created":
                print("✅ Realtime session создана")

            elif event_type == "session.updated":
                print("✅ Realtime session настроена")
                print("🎙️ Можно говорить.\n")

            elif event_type == "input_audio_buffer.speech_started":
                print("\n🎤 Речь началась")
                first_audio_time = None

            elif event_type == "input_audio_buffer.speech_stopped":
                speech_end_time = time.perf_counter()
                print("🛑 Конец реплики определён")

            elif (
                "transcription" in event_type
                and event_type.endswith(".completed")
            ):
                transcript = (
                    event.get("transcript")
                    or event.get("text")
                    or ""
                )

                if transcript:
                    print(f"🗣️ Вы: {transcript}")

            elif event_type.endswith("audio_transcript.done"):
                transcript = (
                    event.get("transcript")
                    or event.get("text")
                    or ""
                )

                if transcript:
                    print(f"🤖 Ответ: {transcript}")

            elif event_type.endswith("audio.delta"):
                audio_b64 = event.get("delta")

                if not audio_b64:
                    continue

                if first_audio_time is None:
                    first_audio_time = time.perf_counter()

                    if speech_end_time is not None:
                        latency = first_audio_time - speech_end_time
                        print(
                            f"⚡ До первого аудио: "
                            f"{latency:.3f} сек"
                        )

                audio_bytes = b64_decode(audio_b64)
                await audio_out.write(audio_bytes)

            elif event_type == "error":
                print(
                    "\n❌ Yandex Realtime error:\n",
                    json.dumps(
                        event,
                        ensure_ascii=False,
                        indent=2,
                    ),
                )


# ============================================================
# MAIN
# ============================================================

async def main():
    print("=" * 60)
    print(" AngioPlus — Yandex AI Studio Realtime")
    print("=" * 60)
    print()
    print(f"Model: {MODEL}")
    print(f"Voice: {VOICE}")
    print(f"Input: {IN_RATE} Hz")
    print(f"Output: {OUT_RATE} Hz")
    print()
    print("Подключение к Yandex Realtime API...")

    timeout = aiohttp.ClientTimeout(total=None)

    async with aiohttp.ClientSession(timeout=timeout) as session:

        async with session.ws_connect(
            WSS_URL,
            headers=HEADERS,
            heartbeat=20.0,
        ) as ws:

            print("✅ WebSocket подключён")

            await setup_session(ws)

            await send_greeting(ws)

            await asyncio.gather(
                uplink(ws),
                downlink(ws),
            )


if __name__ == "__main__":
    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        print("\n🛑 Агент остановлен.")

    except Exception as exc:
        print(f"\n❌ Ошибка: {exc}")
