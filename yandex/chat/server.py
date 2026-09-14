#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from aiohttp import web
from dotenv import load_dotenv
from openai import OpenAI

from yandex.common.prompts import TEXT_SYSTEM_PROMPT
from yandex.retrieval.controller import RetrievalController

from support.db import get_session_factory
from support.domain import (
    ConversationMode,
    MessageRole,
    SupportEvent,
)
from support.idempotency import IdempotencyService
from support.message_service import MessageService
from support.session_service import SessionService
from support.state_service import SupportStateService


# ============================================================
# PATHS / ENV
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
YANDEX_VECTOR_STORE_ID = os.getenv("YANDEX_VECTOR_STORE_ID")

if not YANDEX_API_KEY:
    raise RuntimeError("В .env отсутствует YANDEX_API_KEY")

if not YANDEX_FOLDER_ID:
    raise RuntimeError("В .env отсутствует YANDEX_FOLDER_ID")

if not YANDEX_VECTOR_STORE_ID:
    raise RuntimeError("В .env отсутствует YANDEX_VECTOR_STORE_ID")


# ============================================================
# CONFIG
# ============================================================

HOST = "0.0.0.0"
PORT = 8001

BASE_URL = "https://ai.api.cloud.yandex.net/v1"
MODEL = f"gpt://{YANDEX_FOLDER_ID}/yandexgpt-lite/latest"

MAX_TOOL_ROUNDS = 3

DECISION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "angioplus_support_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": [
                        "ANSWER",
                        "CLARIFY",
                        "ESCALATE",
                        "OUT_OF_SCOPE",
                    ],
                },
                "reason": {
                    "type": "string",
                    "enum": [
                        "RESOLVED_BY_L1",
                        "NEEDS_CLARIFICATION",
                        "INSUFFICIENT_DOCUMENTATION",
                        "USER_REQUESTED_HUMAN",
                        "OUT_OF_SCOPE",
                        "TEMPORARY_SYSTEM_ERROR",
                    ],
                },
            },
            "required": [
                "decision",
                "reason",
            ],
            "additionalProperties": False,
        },
    },
}

# ============================================================
# CLIENTS
# ============================================================

client = OpenAI(
    api_key=YANDEX_API_KEY,
    base_url=BASE_URL,
)

retrieval_controller = RetrievalController(
    max_num_results=5,
)


# ============================================================
# TOOL SCHEMA
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_angioplus_knowledge",
            "description": (
                "Search the official AngioPlus Core knowledge base. "
                "Use this function for factual questions about "
                "AngioPlus Core, including functions, settings, "
                "DICOM, PACS, PStation, μFR, system requirements, "
                "usage instructions, limitations and technical support."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A standalone search query describing "
                            "the user's AngioPlus Core question."
                        ),
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
]


# ============================================================
# RETRIEVAL
# ============================================================

def build_retrieval_output(query: str) -> dict[str, Any]:
    response = retrieval_controller.search(query)

    evidence = []

    for item in response.evidence:
        evidence.append(
            {
                "rank": item.rank,
                "score": item.score,
                "source": item.source,
                "source_type": item.source_type,
                "authority": item.authority,
                "section": item.section,
                "pages": item.pages,
                "text": item.text,
            }
        )

    return {
        "ok": True,
        "query": response.query,
        "latency_ms": round(response.latency_ms, 1),
        "total_candidates": response.total_candidates,
        "evidence": evidence,
    }


def log_retrieval(payload: dict[str, Any]) -> None:
    print()
    print(
        "✅ Retrieval completed:",
        payload.get("total_candidates"),
        "chunks,",
        payload.get("latency_ms"),
        "ms",
    )
    print()

    for evidence in payload.get("evidence", []):
        rank = evidence.get("rank")
        source = evidence.get("source")
        score = evidence.get("score")
        text = evidence.get("text") or ""

        if isinstance(score, (int, float)):
            score_text = f"{score:.6f}"
        else:
            score_text = str(score)

        preview = text.replace("\n", " ").strip()[:220]

        print(
            f"#{rank} {source} "
            f"score={score_text}"
        )
        print(f"   {preview}")
        print()

def classify_support_decision_sync(
    user_message: str,
    answer: str,
    retrieval_used: bool,
    retrieval_failed: bool,
) -> dict[str, str]:

    if retrieval_failed:
        return {
            "decision": "ANSWER",
            "reason": "TEMPORARY_SYSTEM_ERROR",
        }

    decision_prompt = f"""
Ты классификатор технической поддержки AngioPlus Core.

Твоя задача — определить итоговый статус уже сформированного ответа.

Доступны четыре решения:


ANSWER

Используй ANSWER, если вопрос относится к AngioPlus Core
и пользователю дан подтверждённый ответ на его вопрос.

reason = RESOLVED_BY_L1


CLARIFY

Используй CLARIFY, если невозможно однозначно понять,
что именно спрашивает пользователь,
и для продолжения необходимо задать уточняющий вопрос.

Это относится в том числе к слишком коротким,
неполным или неоднозначным вопросам.

reason = NEEDS_CLARIFICATION


ESCALATE

Используй ESCALATE в двух случаях:

1. Пользователь явно просит подключить, вернуть,
перевести или соединить его со специалистом,
оператором или технической поддержкой.

В этом случае:
reason = USER_REQUESTED_HUMAN

2. Вопрос или проблема относится к AngioPlus Core,
но имеющейся документации недостаточно,
чтобы дать подтверждённый ответ или решение.

В этом случае:
reason = INSUFFICIENT_DOCUMENTATION
Запрос пользователя на оператора сам по себе относится
к процессу технической поддержки AngioPlus и НЕ является OUT_OF_SCOPE.

Примеры:
"подключи меня к оператору"
"хочу продолжить с оператором"
"верни меня к специалисту"
"я готов продолжить с техподдержкой"

→ decision = ESCALATE
→ reason = USER_REQUESTED_HUMAN

OUT_OF_SCOPE

Используй OUT_OF_SCOPE, если вопрос не относится
к AngioPlus Core или его технической поддержке.

Например:

"Какого цвета банан?"
"Какая сегодня погода?"
"Кто президент Франции?"
"Напиши стихотворение."
"Сколько будет 2 + 2?"

Такие вопросы нельзя передавать специалисту
технической поддержки AngioPlus.

reason = OUT_OF_SCOPE


ВАЖНО

Не определяй ESCALATE только потому,
что в ответе написано:

"Такой информации нет в имеющейся документации."

Сначала обязательно определи,
относится ли исходный вопрос пользователя
к AngioPlus Core.

Если вопрос не относится к AngioPlus Core,
выбирай OUT_OF_SCOPE,
даже если ассистент сообщил,
что информации нет в документации.

Если вопрос относится к AngioPlus Core,
но документация не позволяет дать подтверждённый ответ,
выбирай ESCALATE.

Если вопрос невозможно достаточно уверенно понять,
выбирай CLARIFY, а не ESCALATE.

Не придумывай новую информацию.
Не пытайся самостоятельно отвечать на вопрос.
Только классифицируй результат.


Вопрос пользователя:
{user_message}

Использовался поиск по документации:
{retrieval_used}

Ответ ассистента:
{answer}
""".strip()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": decision_prompt,
            }
        ],
        response_format=DECISION_RESPONSE_FORMAT,
    )

    raw_content = (
        response.choices[0].message.content
        or ""
    ).strip()

    parsed = json.loads(raw_content)

    return {
        "decision": str(
            parsed["decision"]
        ).upper(),
        "reason": str(
            parsed["reason"]
        ).upper(),
    }

# ============================================================
# YANDEX CHAT LOOP
# ============================================================

def run_chat_sync(user_message: str) -> dict[str, Any]:
    """
    Один текстовый turn.

    Пока без conversation history.
    Позже frontend будет передавать историю сообщений.
    """

    messages: list[Any] = [
        {
            "role": "system",
            "content": TEXT_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

    retrieval_used = False
    retrieval_failed = False
    retrieval_queries: list[str] = []

    for round_number in range(1, MAX_TOOL_ROUNDS + 1):
        print()
        print(
            f"🤖 Yandex round "
            f"{round_number}/{MAX_TOOL_ROUNDS}"
        )

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls or []

        # ----------------------------------------------------
        # Final answer
        # ----------------------------------------------------

        if not tool_calls:
            answer = (
                assistant_message.content
                or ""
            ).strip()

            decision_result = (
                classify_support_decision_sync(
                    user_message=user_message,
                    answer=answer,
                    retrieval_used=retrieval_used,
                    retrieval_failed=retrieval_failed,
                )
            )

            print()
            print(
                "🧭 Support decision:",
                decision_result["decision"],
                "/",
                decision_result["reason"],
            )

            return {
                "answer": answer,
                "decision": decision_result[
                    "decision"
                ],
                "reason": decision_result[
                    "reason"
                ],
                "retrieval_used": retrieval_used,
                "retrieval_queries": retrieval_queries,
            }

        # ----------------------------------------------------
        # Tool calls
        # ----------------------------------------------------

        messages.append(
            assistant_message.model_dump(
                exclude_none=True
            )
        )

        for tool_call in tool_calls:
            function_name = tool_call.function.name

            print()
            print("=" * 60)
            print("🛠️ FUNCTION CALL")
            print("=" * 60)
            print(f"name: {function_name}")
            print(
                "arguments:",
                tool_call.function.arguments,
            )

            if (
                function_name
                != "search_angioplus_knowledge"
            ):
                tool_result = {
                    "ok": False,
                    "error": "unknown_function",
                }

            else:
                try:
                    arguments = json.loads(
                        tool_call.function.arguments
                        or "{}"
                    )
                except json.JSONDecodeError:
                    arguments = {}

                model_query = str(
                    arguments.get("query")
                    or ""
                ).strip()

                # В текущей stateless-версии retrieval всегда выполняем
                # по исходному вопросу пользователя.
                #
                # LLM может использовать tool call, чтобы решить,
                # нужен ли поиск, но не должна самовольно менять
                # смысл поискового запроса.
                query = user_message.strip()

                if not query:
                    tool_result = {
                        "ok": False,
                        "error": "empty_query",
                    }

                else:
                    retrieval_used = True
                    retrieval_queries.append(query)

                    print()
                    print(
                        f"🧠 Model requested query: {model_query}"
                    )

                    print(
                        f"🔎 Actual retrieval query: {query}"
                    )

                    try:
                        tool_result = (
                            build_retrieval_output(
                                query
                            )
                        )

                        log_retrieval(tool_result)

                    except Exception as exc:
                        print()
                        print(
                            "❌ Retrieval error:",
                            repr(exc),
                        )
                        retrieval_failed = True
                        tool_result = {
                            "ok": False,
                            "error": "retrieval_failed",
                        }

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        tool_result,
                        ensure_ascii=False,
                    ),
                }
            )

    raise RuntimeError(
        "Превышено максимальное число tool rounds"
    )


async def run_chat(
    user_message: str,
) -> dict[str, Any]:
    """
    OpenAI-compatible SDK синхронный,
    поэтому не блокируем aiohttp event loop.
    """

    return await asyncio.to_thread(
        run_chat_sync,
        user_message,
    )


# ============================================================
# HTTP HANDLERS
# ============================================================

async def health_handler(
    request: web.Request,
) -> web.Response:
    return web.json_response(
        {
            "ok": True,
            "service": "angioplus-yandex-chat",
            "model": MODEL,
        }
    )


async def _handle_stateful_turn(
    body: dict[str, Any],
) -> web.Response:
    """
    Stateful path (only used when ``channel`` is present).

    Transaction discipline (approved design):
      TX1  -- claim inbound_event_id (short, committed)
      ---- -- LLM / retrieval happen OUTSIDE any DB transaction
      TX2  -- append messages + apply state transition + finalize (short)

    This function orchestrates; it does not re-implement the LLM core
    (``run_chat``) nor the FSM table (``support.domain``).
    """
    message = str(body.get("message") or "").strip()
    channel = str(body.get("channel") or "").strip()
    external_user_id = str(body.get("external_user_id") or "").strip()
    external_conversation_id = str(
        body.get("external_conversation_id") or ""
    ).strip()
    explicit_conversation_id = str(
        body.get("conversation_id") or ""
    ).strip()
    inbound_event_id = str(body.get("inbound_event_id") or "").strip()

    if not external_user_id:
        return web.json_response(
            {"ok": False, "error": "external_user_id_required"},
            status=400,
        )

    # Channels without a dedicated thread identity must fall back to the
    # user id (per approved rule); this keeps external_conversation_id
    # NOT NULL and the partial unique index airtight.
    if not external_conversation_id:
        return web.json_response(
            {"ok": False, "error": "external_conversation_id_required"},
            status=400,
        )

    session_factory = get_session_factory()

    # ---- TX1: claim inbound event ------------------------------------
    if inbound_event_id:
        async with session_factory() as session:
            idem = IdempotencyService(session)
            claim = await idem.claim(inbound_event_id, channel)
            await session.commit()

        if not claim.claimed:
            # Duplicate delivery: never call the LLM again.
            prior = claim.prior
            if prior is not None:
                return web.json_response(
                    {
                        "ok": True,
                        "answer": prior.answer or "",
                        "retrieval_used": False,
                        "decision": prior.decision or "",
                        "reason": prior.reason or "",
                        "conversation_id": prior.conversation_id,
                        "case_action": prior.case_action,
                        "deduplicated": True,
                    }
                )
            # In-flight duplicate: accept but produce no side effects.
            return web.json_response(
                {
                    "ok": True,
                    "answer": "",
                    "retrieval_used": False,
                    "decision": "",
                    "reason": "",
                    "deduplicated": True,
                }
            )

    # ---- LLM outside any DB transaction ------------------------------
    result = await run_chat(message)
    answer = result["answer"]
    decision = result["decision"]
    reason = result["reason"]

    # ---- TX2: persist turn + apply transition + finalize -------------
    async with session_factory() as session:
        session_service = SessionService(session)
        message_service = MessageService(session)
        state_service = SupportStateService(session)
        idem = IdempotencyService(session)

        if explicit_conversation_id:
            conv = await session_service.get_by_conversation_id(
                explicit_conversation_id
            )
        else:
            conv = None

        if conv is None:
            conv = await session_service.get_or_create_active(
                channel, external_user_id, external_conversation_id
            )

        conversation_id = conv.conversation_id
        mode = conv.conversation_mode

        await message_service.append(
            conversation_id, MessageRole.USER, message, external_user_id
        )

        assistant_message_id = await message_service.append(
            conversation_id, MessageRole.ASSISTANT, answer
        )

        # Map the turn to a support event and apply the FSM transition
        # (only when the conversation is currently human-owned do user
        # messages go to the operator; otherwise it is normal AI chat).
        case_action = "NONE"
        outcome = "answered"

        if decision == "ESCALATE":
            if mode == ConversationMode.HUMAN.value:
                event = SupportEvent.USER_MESSAGE_HUMAN
                case_action = "NONE"
            else:
                current_state = (
                    conv.conversation_mode,
                    conv.support_status,
                    conv.handoff_status,
                )
                if current_state[2] == "waiting_user":
                    event = SupportEvent.ESCALATE_RESUME
                    case_action = "RESUME"
                else:
                    event = SupportEvent.ESCALATE_NEW
                    case_action = "CREATE_NEW"
                outcome = "escalated"
        elif mode == ConversationMode.HUMAN.value:
            event = SupportEvent.USER_MESSAGE_HUMAN
        else:
            event = SupportEvent.USER_MESSAGE_AI

        outcome_obj = await state_service.handle(event, conversation_id)

        if inbound_event_id:
            await idem.finalize(
                inbound_event_id=inbound_event_id,
                outcome=outcome,
                conversation_id=conversation_id,
                assistant_message_id=assistant_message_id,
                decision=decision,
                reason=reason,
                case_action=case_action,
            )

        await session.commit()

        final_mode = (
            outcome_obj.state[0].value
            if outcome_obj.state is not None
            else conv.conversation_mode
        )
        final_status = (
            outcome_obj.state[1].value
            if outcome_obj.state is not None
            else conv.support_status
        )
        final_handoff = (
            outcome_obj.state[2].value
            if outcome_obj.state is not None and outcome_obj.state[2] is not None
            else None
        )

    print()
    print("=" * 60)
    print("🤖 FINAL ANSWER")
    print("=" * 60)
    print(answer)
    print()

    return web.json_response(
        {
            "ok": True,
            "answer": answer,
            "retrieval_used": result["retrieval_used"],
            "decision": decision,
            "reason": reason,
            "conversation_id": conversation_id,
            "conversation_mode": final_mode,
            "support_status": final_status,
            "handoff_status": final_handoff,
            "case_action": case_action,
            "deduplicated": False,
        }
    )


async def chat_handler(
    request: web.Request,
) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        return web.json_response(
            {
                "ok": False,
                "error": "invalid_json",
            },
            status=400,
        )

    message = str(
        body.get("message")
        or ""
    ).strip()

    if not message:
        return web.json_response(
            {
                "ok": False,
                "error": "message_required",
            },
            status=400,
        )

    print()
    print("=" * 60)
    print("💬 NEW CHAT REQUEST")
    print("=" * 60)
    print(f"👤 User: {message}")

    # ---- Stateful mode (channel present) -----------------------------
    # Legacy {"message": "..."} callers have no channel and keep the
    # exact old stateless behaviour, including no dedup.
    if str(body.get("channel") or "").strip():
        try:
            return await _handle_stateful_turn(body)
        except Exception as exc:
            print()
            print("❌ Stateful chat error:", repr(exc))
            return web.json_response(
                {
                    "ok": False,
                    "error": "chat_failed",
                },
                status=500,
            )

    # ---- Legacy stateless mode ---------------------------------------
    try:
        result = await run_chat(message)

    except Exception as exc:
        print()
        print(
            "❌ Chat error:",
            repr(exc),
        )

        return web.json_response(
            {
                "ok": False,
                "error": "chat_failed",
            },
            status=500,
        )

    answer = result["answer"]

    print()
    print("=" * 60)
    print("🤖 FINAL ANSWER")
    print("=" * 60)
    print(answer)
    print()

    return web.json_response(
        {
            "ok": True,
            "answer": answer,
            "retrieval_used": result[
                "retrieval_used"
            ],
            "decision": result[
                "decision"
            ],
            "reason": result[
                "reason"
            ],
        }
    )


# ============================================================
# CORS
# ============================================================

@web.middleware
async def cors_middleware(
    request: web.Request,
    handler,
):
    if request.method == "OPTIONS":
        response = web.Response()
    else:
        response = await handler(request)

    response.headers[
        "Access-Control-Allow-Origin"
    ] = "http://localhost:3000"

    response.headers[
        "Access-Control-Allow-Methods"
    ] = "GET,POST,OPTIONS"

    response.headers[
        "Access-Control-Allow-Headers"
    ] = "Content-Type"

    return response


# ============================================================
# APP
# ============================================================

def create_app() -> web.Application:
    app = web.Application(
        middlewares=[
            cors_middleware,
        ]
    )

    app.router.add_get(
        "/health",
        health_handler,
    )

    app.router.add_post(
        "/api/chat",
        chat_handler,
    )

    app.router.add_route(
        "OPTIONS",
        "/api/chat",
        chat_handler,
    )

    return app


def main() -> None:
    print("=" * 60)
    print(" AngioPlus — Yandex Text Backend")
    print("=" * 60)
    print()
    print(f"Model: {MODEL}")
    print(f"Vector Store: {YANDEX_VECTOR_STORE_ID}")
    print()
    print(
        f"HTTP: http://{HOST}:{PORT}"
    )
    print(
        f"Health: http://{HOST}:{PORT}/health"
    )
    print(
        f"Chat:   POST "
        f"http://{HOST}:{PORT}/api/chat"
    )
    print()

    web.run_app(
        create_app(),
        host=HOST,
        port=PORT,
    )


if __name__ == "__main__":
    main()