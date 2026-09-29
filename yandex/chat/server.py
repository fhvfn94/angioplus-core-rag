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
    HandoffStatus,
    MessageRole,
    SupportEvent,
    SupportStatus,
)
from support.idempotency import IdempotencyService
from support.message_service import MessageService
from support.conversation_context import build_conversation_history
from support.procedure_state_store import ProcedureStateConflict
from support.runtime_service import (
    FinalizedDelivery,
    RuntimeDelivery,
    RuntimeRecoveryUnavailable,
    StatefulTurnRuntime,
)
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
    raise RuntimeError("╨Т .env ╨╛╤В╤Б╤Г╤В╤Б╤В╨▓╤Г╨╡╤В YANDEX_API_KEY")

if not YANDEX_FOLDER_ID:
    raise RuntimeError("╨Т .env ╨╛╤В╤Б╤Г╤В╤Б╤В╨▓╤Г╨╡╤В YANDEX_FOLDER_ID")

if not YANDEX_VECTOR_STORE_ID:
    raise RuntimeError("╨Т .env ╨╛╤В╤Б╤Г╤В╤Б╤В╨▓╤Г╨╡╤В YANDEX_VECTOR_STORE_ID")


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
                "DICOM, PACS, PStation, ╬╝FR, system requirements, "
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
        "тЬЕ Retrieval completed:",
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
╨в╤Л ╨║╨╗╨░╤Б╤Б╨╕╤Д╨╕╨║╨░╤В╨╛╤А ╤В╨╡╤Е╨╜╨╕╤З╨╡╤Б╨║╨╛╨╣ ╨┐╨╛╨┤╨┤╨╡╤А╨╢╨║╨╕ AngioPlus Core.

╨в╨▓╨╛╤П ╨╖╨░╨┤╨░╤З╨░ тАФ ╨╛╨┐╤А╨╡╨┤╨╡╨╗╨╕╤В╤М ╨╕╤В╨╛╨│╨╛╨▓╤Л╨╣ ╤Б╤В╨░╤В╤Г╤Б ╤Г╨╢╨╡ ╤Б╤Д╨╛╤А╨╝╨╕╤А╨╛╨▓╨░╨╜╨╜╨╛╨│╨╛ ╨╛╤В╨▓╨╡╤В╨░.

╨Ф╨╛╤Б╤В╤Г╨┐╨╜╤Л ╤З╨╡╤В╤Л╤А╨╡ ╤А╨╡╤И╨╡╨╜╨╕╤П:


ANSWER

╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╣ ANSWER, ╨╡╤Б╨╗╨╕ ╨▓╨╛╨┐╤А╨╛╤Б ╨╛╤В╨╜╨╛╤Б╨╕╤В╤Б╤П ╨║ AngioPlus Core
╨╕ ╨┐╨╛╨╗╤М╨╖╨╛╨▓╨░╤В╨╡╨╗╤О ╨┤╨░╨╜ ╨┐╨╛╨┤╤В╨▓╨╡╤А╨╢╨┤╤С╨╜╨╜╤Л╨╣ ╨╛╤В╨▓╨╡╤В ╨╜╨░ ╨╡╨│╨╛ ╨▓╨╛╨┐╤А╨╛╤Б.

reason = RESOLVED_BY_L1


CLARIFY

╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╣ CLARIFY, ╨╡╤Б╨╗╨╕ ╨╜╨╡╨▓╨╛╨╖╨╝╨╛╨╢╨╜╨╛ ╨╛╨┤╨╜╨╛╨╖╨╜╨░╤З╨╜╨╛ ╨┐╨╛╨╜╤П╤В╤М,
╤З╤В╨╛ ╨╕╨╝╨╡╨╜╨╜╨╛ ╤Б╨┐╤А╨░╤И╨╕╨▓╨░╨╡╤В ╨┐╨╛╨╗╤М╨╖╨╛╨▓╨░╤В╨╡╨╗╤М,
╨╕ ╨┤╨╗╤П ╨┐╤А╨╛╨┤╨╛╨╗╨╢╨╡╨╜╨╕╤П ╨╜╨╡╨╛╨▒╤Е╨╛╨┤╨╕╨╝╨╛ ╨╖╨░╨┤╨░╤В╤М ╤Г╤В╨╛╤З╨╜╤П╤О╤Й╨╕╨╣ ╨▓╨╛╨┐╤А╨╛╤Б.

╨н╤В╨╛ ╨╛╤В╨╜╨╛╤Б╨╕╤В╤Б╤П ╨▓ ╤В╨╛╨╝ ╤З╨╕╤Б╨╗╨╡ ╨║ ╤Б╨╗╨╕╤И╨║╨╛╨╝ ╨║╨╛╤А╨╛╤В╨║╨╕╨╝,
╨╜╨╡╨┐╨╛╨╗╨╜╤Л╨╝ ╨╕╨╗╨╕ ╨╜╨╡╨╛╨┤╨╜╨╛╨╖╨╜╨░╤З╨╜╤Л╨╝ ╨▓╨╛╨┐╤А╨╛╤Б╨░╨╝.

reason = NEEDS_CLARIFICATION


ESCALATE

╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╣ ESCALATE ╨▓ ╨┤╨▓╤Г╤Е ╤Б╨╗╤Г╤З╨░╤П╤Е:

1. ╨Я╨╛╨╗╤М╨╖╨╛╨▓╨░╤В╨╡╨╗╤М ╤П╨▓╨╜╨╛ ╨┐╤А╨╛╤Б╨╕╤В ╨┐╨╛╨┤╨║╨╗╤О╤З╨╕╤В╤М, ╨▓╨╡╤А╨╜╤Г╤В╤М,
╨┐╨╡╤А╨╡╨▓╨╡╤Б╤В╨╕ ╨╕╨╗╨╕ ╤Б╨╛╨╡╨┤╨╕╨╜╨╕╤В╤М ╨╡╨│╨╛ ╤Б╨╛ ╤Б╨┐╨╡╤Ж╨╕╨░╨╗╨╕╤Б╤В╨╛╨╝,
╨╛╨┐╨╡╤А╨░╤В╨╛╤А╨╛╨╝ ╨╕╨╗╨╕ ╤В╨╡╤Е╨╜╨╕╤З╨╡╤Б╨║╨╛╨╣ ╨┐╨╛╨┤╨┤╨╡╤А╨╢╨║╨╛╨╣.

╨Т ╤Н╤В╨╛╨╝ ╤Б╨╗╤Г╤З╨░╨╡:
reason = USER_REQUESTED_HUMAN

2. ╨Т╨╛╨┐╤А╨╛╤Б ╨╕╨╗╨╕ ╨┐╤А╨╛╨▒╨╗╨╡╨╝╨░ ╨╛╤В╨╜╨╛╤Б╨╕╤В╤Б╤П ╨║ AngioPlus Core,
╨╜╨╛ ╨╕╨╝╨╡╤О╤Й╨╡╨╣╤Б╤П ╨┤╨╛╨║╤Г╨╝╨╡╨╜╤В╨░╤Ж╨╕╨╕ ╨╜╨╡╨┤╨╛╤Б╤В╨░╤В╨╛╤З╨╜╨╛,
╤З╤В╨╛╨▒╤Л ╨┤╨░╤В╤М ╨┐╨╛╨┤╤В╨▓╨╡╤А╨╢╨┤╤С╨╜╨╜╤Л╨╣ ╨╛╤В╨▓╨╡╤В ╨╕╨╗╨╕ ╤А╨╡╤И╨╡╨╜╨╕╨╡.

╨Т ╤Н╤В╨╛╨╝ ╤Б╨╗╤Г╤З╨░╨╡:
reason = INSUFFICIENT_DOCUMENTATION
╨Ч╨░╨┐╤А╨╛╤Б ╨┐╨╛╨╗╤М╨╖╨╛╨▓╨░╤В╨╡╨╗╤П ╨╜╨░ ╨╛╨┐╨╡╤А╨░╤В╨╛╤А╨░ ╤Б╨░╨╝ ╨┐╨╛ ╤Б╨╡╨▒╨╡ ╨╛╤В╨╜╨╛╤Б╨╕╤В╤Б╤П
╨║ ╨┐╤А╨╛╤Ж╨╡╤Б╤Б╤Г ╤В╨╡╤Е╨╜╨╕╤З╨╡╤Б╨║╨╛╨╣ ╨┐╨╛╨┤╨┤╨╡╤А╨╢╨║╨╕ AngioPlus ╨╕ ╨Э╨Х ╤П╨▓╨╗╤П╨╡╤В╤Б╤П OUT_OF_SCOPE.

╨Я╤А╨╕╨╝╨╡╤А╤Л:
"╨┐╨╛╨┤╨║╨╗╤О╤З╨╕ ╨╝╨╡╨╜╤П ╨║ ╨╛╨┐╨╡╤А╨░╤В╨╛╤А╤Г"
"╤Е╨╛╤З╤Г ╨┐╤А╨╛╨┤╨╛╨╗╨╢╨╕╤В╤М ╤Б ╨╛╨┐╨╡╤А╨░╤В╨╛╤А╨╛╨╝"
"╨▓╨╡╤А╨╜╨╕ ╨╝╨╡╨╜╤П ╨║ ╤Б╨┐╨╡╤Ж╨╕╨░╨╗╨╕╤Б╤В╤Г"
"╤П ╨│╨╛╤В╨╛╨▓ ╨┐╤А╨╛╨┤╨╛╨╗╨╢╨╕╤В╤М ╤Б ╤В╨╡╤Е╨┐╨╛╨┤╨┤╨╡╤А╨╢╨║╨╛╨╣"

тЖТ decision = ESCALATE
тЖТ reason = USER_REQUESTED_HUMAN

OUT_OF_SCOPE

╨Ш╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╣ OUT_OF_SCOPE, ╨╡╤Б╨╗╨╕ ╨▓╨╛╨┐╤А╨╛╤Б ╨╜╨╡ ╨╛╤В╨╜╨╛╤Б╨╕╤В╤Б╤П
╨║ AngioPlus Core ╨╕╨╗╨╕ ╨╡╨│╨╛ ╤В╨╡╤Е╨╜╨╕╤З╨╡╤Б╨║╨╛╨╣ ╨┐╨╛╨┤╨┤╨╡╤А╨╢╨║╨╡.

╨Э╨░╨┐╤А╨╕╨╝╨╡╤А:

"╨Ъ╨░╨║╨╛╨│╨╛ ╤Ж╨▓╨╡╤В╨░ ╨▒╨░╨╜╨░╨╜?"
"╨Ъ╨░╨║╨░╤П ╤Б╨╡╨│╨╛╨┤╨╜╤П ╨┐╨╛╨│╨╛╨┤╨░?"
"╨Ъ╤В╨╛ ╨┐╤А╨╡╨╖╨╕╨┤╨╡╨╜╤В ╨д╤А╨░╨╜╤Ж╨╕╨╕?"
"╨Э╨░╨┐╨╕╤И╨╕ ╤Б╤В╨╕╤Е╨╛╤В╨▓╨╛╤А╨╡╨╜╨╕╨╡."
"╨б╨║╨╛╨╗╤М╨║╨╛ ╨▒╤Г╨┤╨╡╤В 2 + 2?"

╨в╨░╨║╨╕╨╡ ╨▓╨╛╨┐╤А╨╛╤Б╤Л ╨╜╨╡╨╗╤М╨╖╤П ╨┐╨╡╤А╨╡╨┤╨░╨▓╨░╤В╤М ╤Б╨┐╨╡╤Ж╨╕╨░╨╗╨╕╤Б╤В╤Г
╤В╨╡╤Е╨╜╨╕╤З╨╡╤Б╨║╨╛╨╣ ╨┐╨╛╨┤╨┤╨╡╤А╨╢╨║╨╕ AngioPlus.

reason = OUT_OF_SCOPE


╨Т╨Р╨Ц╨Э╨Ю

╨Э╨╡ ╨╛╨┐╤А╨╡╨┤╨╡╨╗╤П╨╣ ESCALATE ╤В╨╛╨╗╤М╨║╨╛ ╨┐╨╛╤В╨╛╨╝╤Г,
╤З╤В╨╛ ╨▓ ╨╛╤В╨▓╨╡╤В╨╡ ╨╜╨░╨┐╨╕╤Б╨░╨╜╨╛:

"╨в╨░╨║╨╛╨╣ ╨╕╨╜╤Д╨╛╤А╨╝╨░╤Ж╨╕╨╕ ╨╜╨╡╤В ╨▓ ╨╕╨╝╨╡╤О╤Й╨╡╨╣╤Б╤П ╨┤╨╛╨║╤Г╨╝╨╡╨╜╤В╨░╤Ж╨╕╨╕."

╨б╨╜╨░╤З╨░╨╗╨░ ╨╛╨▒╤П╨╖╨░╤В╨╡╨╗╤М╨╜╨╛ ╨╛╨┐╤А╨╡╨┤╨╡╨╗╨╕,
╨╛╤В╨╜╨╛╤Б╨╕╤В╤Б╤П ╨╗╨╕ ╨╕╤Б╤Е╨╛╨┤╨╜╤Л╨╣ ╨▓╨╛╨┐╤А╨╛╤Б ╨┐╨╛╨╗╤М╨╖╨╛╨▓╨░╤В╨╡╨╗╤П
╨║ AngioPlus Core.

╨Х╤Б╨╗╨╕ ╨▓╨╛╨┐╤А╨╛╤Б ╨╜╨╡ ╨╛╤В╨╜╨╛╤Б╨╕╤В╤Б╤П ╨║ AngioPlus Core,
╨▓╤Л╨▒╨╕╤А╨░╨╣ OUT_OF_SCOPE,
╨┤╨░╨╢╨╡ ╨╡╤Б╨╗╨╕ ╨░╤Б╤Б╨╕╤Б╤В╨╡╨╜╤В ╤Б╨╛╨╛╨▒╤Й╨╕╨╗,
╤З╤В╨╛ ╨╕╨╜╤Д╨╛╤А╨╝╨░╤Ж╨╕╨╕ ╨╜╨╡╤В ╨▓ ╨┤╨╛╨║╤Г╨╝╨╡╨╜╤В╨░╤Ж╨╕╨╕.

╨Х╤Б╨╗╨╕ ╨▓╨╛╨┐╤А╨╛╤Б ╨╛╤В╨╜╨╛╤Б╨╕╤В╤Б╤П ╨║ AngioPlus Core,
╨╜╨╛ ╨┤╨╛╨║╤Г╨╝╨╡╨╜╤В╨░╤Ж╨╕╤П ╨╜╨╡ ╨┐╨╛╨╖╨▓╨╛╨╗╤П╨╡╤В ╨┤╨░╤В╤М ╨┐╨╛╨┤╤В╨▓╨╡╤А╨╢╨┤╤С╨╜╨╜╤Л╨╣ ╨╛╤В╨▓╨╡╤В,
╨▓╤Л╨▒╨╕╤А╨░╨╣ ESCALATE.

╨Х╤Б╨╗╨╕ ╨▓╨╛╨┐╤А╨╛╤Б ╨╜╨╡╨▓╨╛╨╖╨╝╨╛╨╢╨╜╨╛ ╨┤╨╛╤Б╤В╨░╤В╨╛╤З╨╜╨╛ ╤Г╨▓╨╡╤А╨╡╨╜╨╜╨╛ ╨┐╨╛╨╜╤П╤В╤М,
╨▓╤Л╨▒╨╕╤А╨░╨╣ CLARIFY, ╨░ ╨╜╨╡ ESCALATE.

╨Э╨╡ ╨┐╤А╨╕╨┤╤Г╨╝╤Л╨▓╨░╨╣ ╨╜╨╛╨▓╤Г╤О ╨╕╨╜╤Д╨╛╤А╨╝╨░╤Ж╨╕╤О.
╨Э╨╡ ╨┐╤Л╤В╨░╨╣╤Б╤П ╤Б╨░╨╝╨╛╤Б╤В╨╛╤П╤В╨╡╨╗╤М╨╜╨╛ ╨╛╤В╨▓╨╡╤З╨░╤В╤М ╨╜╨░ ╨▓╨╛╨┐╤А╨╛╤Б.
╨в╨╛╨╗╤М╨║╨╛ ╨║╨╗╨░╤Б╤Б╨╕╤Д╨╕╤Ж╨╕╤А╤Г╨╣ ╤А╨╡╨╖╤Г╨╗╤М╤В╨░╤В.


╨Т╨╛╨┐╤А╨╛╤Б ╨┐╨╛╨╗╤М╨╖╨╛╨▓╨░╤В╨╡╨╗╤П:
{user_message}

╨Ш╤Б╨┐╨╛╨╗╤М╨╖╨╛╨▓╨░╨╗╤Б╤П ╨┐╨╛╨╕╤Б╨║ ╨┐╨╛ ╨┤╨╛╨║╤Г╨╝╨╡╨╜╤В╨░╤Ж╨╕╨╕:
{retrieval_used}

╨Ю╤В╨▓╨╡╤В ╨░╤Б╤Б╨╕╤Б╤В╨╡╨╜╤В╨░:
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
    ╨Ю╨┤╨╕╨╜ ╤В╨╡╨║╤Б╤В╨╛╨▓╤Л╨╣ turn.

    ╨Я╨╛╨║╨░ ╨▒╨╡╨╖ conversation history.
    ╨Я╨╛╨╖╨╢╨╡ frontend ╨▒╤Г╨┤╨╡╤В ╨┐╨╡╤А╨╡╨┤╨░╨▓╨░╤В╤М ╨╕╤Б╤В╨╛╤А╨╕╤О ╤Б╨╛╨╛╨▒╤Й╨╡╨╜╨╕╨╣.
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
            f"ЁЯдЦ Yandex round "
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
                "ЁЯзн Support decision:",
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
            print("ЁЯЫая╕П FUNCTION CALL")
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

                # ╨Т ╤В╨╡╨║╤Г╤Й╨╡╨╣ stateless-╨▓╨╡╤А╤Б╨╕╨╕ retrieval ╨▓╤Б╨╡╨│╨┤╨░ ╨▓╤Л╨┐╨╛╨╗╨╜╤П╨╡╨╝
                # ╨┐╨╛ ╨╕╤Б╤Е╨╛╨┤╨╜╨╛╨╝╤Г ╨▓╨╛╨┐╤А╨╛╤Б╤Г ╨┐╨╛╨╗╤М╨╖╨╛╨▓╨░╤В╨╡╨╗╤П.
                #
                # LLM ╨╝╨╛╨╢╨╡╤В ╨╕╤Б╨┐╨╛╨╗╤М╨╖╨╛╨▓╨░╤В╤М tool call, ╤З╤В╨╛╨▒╤Л ╤А╨╡╤И╨╕╤В╤М,
                # ╨╜╤Г╨╢╨╡╨╜ ╨╗╨╕ ╨┐╨╛╨╕╤Б╨║, ╨╜╨╛ ╨╜╨╡ ╨┤╨╛╨╗╨╢╨╜╨░ ╤Б╨░╨╝╨╛╨▓╨╛╨╗╤М╨╜╨╛ ╨╝╨╡╨╜╤П╤В╤М
                # ╤Б╨╝╤Л╤Б╨╗ ╨┐╨╛╨╕╤Б╨║╨╛╨▓╨╛╨│╨╛ ╨╖╨░╨┐╤А╨╛╤Б╨░.
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
                        f"ЁЯза Model requested query: {model_query}"
                    )

                    print(
                        f"ЁЯФО Actual retrieval query: {query}"
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
                            "тЭМ Retrieval error:",
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
        "╨Я╤А╨╡╨▓╤Л╤И╨╡╨╜╨╛ ╨╝╨░╨║╤Б╨╕╨╝╨░╨╗╤М╨╜╨╛╨╡ ╤З╨╕╤Б╨╗╨╛ tool rounds"
    )


async def run_chat(
    user_message: str,
) -> dict[str, Any]:
    """
    OpenAI-compatible SDK ╤Б╨╕╨╜╤Е╤А╨╛╨╜╨╜╤Л╨╣,
    ╨┐╨╛╤Н╤В╨╛╨╝╤Г ╨╜╨╡ ╨▒╨╗╨╛╨║╨╕╤А╤Г╨╡╨╝ aiohttp event loop.
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


# ============================================================
# THE TURN RUNTIME (composition root)
# ============================================================

# HTTP-only compatibility: the runtime's ``RuntimeTurn`` deliberately does not
# carry the legacy ``retrieval_used`` flag (it is an HTTP compatibility
# concern, not a runtime fact). The turn FAMILY tells the transport whether
# approved documentation was consulted: a procedure turn (``TURN_EXECUTED``)
# and a knowledge answer (``KNOWLEDGE_QA``) both use retrieval; every other
# family answers without it.
_RETRIEVAL_TURN_STATUSES = ("TURN_EXECUTED", "KNOWLEDGE_QA")


def _turn_runtime(
    session_factory,
) -> StatefulTurnRuntime:
    """Compose the ONE runtime boundary for an inbound support turn.

    A tiny module-level factory: the transport stays transport-only (it never
    constructs a provider), and the whole boundary is injectable in tests by
    replacing this single function.
    """
    return StatefulTurnRuntime(session_factory=session_factory)


def _turn_retrieval_used(turn) -> bool:
    """The legacy HTTP ``retrieval_used`` flag for ONE runtime turn."""
    return turn.outcome_status in _RETRIEVAL_TURN_STATUSES


def _turn_response_body(
    turn,
    *,
    conversation_id: str,
    final_mode: str,
    final_status: str,
    final_handoff,
    external_case_id,
    deduplicated: bool,
) -> dict[str, Any]:
    """Map ONE ``RuntimeTurn`` to the EXISTING ``/api/chat`` body.

    The turn contributes the wording and the projection's
    ``decision`` / ``reason`` / ``case_action``; the transport contributes the
    lifecycle triple it just applied, the HTTP-only ``retrieval_used`` and the
    Tracker case key.
    """
    return {
        "ok": True,
        "answer": turn.text,
        "retrieval_used": _turn_retrieval_used(turn),
        "decision": turn.decision or "",
        "reason": turn.reason,
        "conversation_id": conversation_id,
        "conversation_mode": final_mode,
        "support_status": final_status,
        "handoff_status": final_handoff,
        "case_action": turn.case_action.value,
        "external_case_id": external_case_id,
        "deduplicated": deduplicated,
    }


async def _resolve_duplicate_turn(
    runtime: StatefulTurnRuntime,
    *,
    inbound_event_id: str,
    conversation_id: str,
) -> web.Response:
    """Serve an ALREADY-APPLIED turn through the runtime (READ ONLY).

    The runtime reproduces the EXACT finalized body (LEVEL A) or the ORIGINAL
    turn from the durable recovery record (LEVEL B); it never calls the model
    and never mutates conversation state. A duplicate whose first attempt is
    still in flight has nothing durable to reproduce and is accepted with an
    EMPTY, side-effect-free body.
    """
    try:
        delivery: RuntimeDelivery = await runtime.resolve_duplicate(
            inbound_event_id=inbound_event_id,
            conversation_id=conversation_id,
        )
    except RuntimeRecoveryUnavailable:
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

    if isinstance(delivery, FinalizedDelivery):
        # The EXACT original body, reproduced verbatim from the snapshot.
        return web.json_response(delivery.body.model_dump())

    # RecoveredTurnDelivery: the engine applied the turn but the runtime never
    # finalized it, so there is no persisted lifecycle triple to echo.
    turn = delivery.turn
    return web.json_response(
        {
            "ok": True,
            "answer": turn.text,
            "retrieval_used": _turn_retrieval_used(turn),
            "decision": turn.decision or "",
            "reason": turn.reason,
            "conversation_id": turn.conversation_id,
            "case_action": turn.case_action.value,
            "deduplicated": True,
        }
    )


async def _handle_stateful_turn(
    body: dict[str, Any],
) -> web.Response:
    """
    Stateful path (only used when ``channel`` is present).

    Transaction discipline (approved design):
      TX1  -- claim inbound_event_id (short, committed)
      ---- -- the engine turn happens OUTSIDE any DB transaction
      TX2  -- append messages + apply the runtime's FSM event + finalize (short)

    The turn ITSELF is decided and executed by the ONE runtime boundary
    (:class:`~support.runtime_service.StatefulTurnRuntime`) and its projection;
    this function only drives the inbound-event lifecycle, applies the FSM
    event the runtime selected and maps the typed turn to HTTP. It
    re-implements no routing, no FSM table (``support.domain``) and no turn
    execution.
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
    runtime = _turn_runtime(session_factory)

    # ---- TX1: claim inbound event ------------------------------------
    if inbound_event_id:
        claim = await runtime.claim(
            inbound_event_id=inbound_event_id,
            channel=channel,
        )

        if not claim.claimed:
            # Duplicate delivery: never run the engine again. When the first
            # delivery is known, the runtime reproduces the ALREADY-APPLIED
            # turn from durable state -- the finalized snapshot (LEVEL A) or
            # the engine's recovery record (LEVEL B) -- and the response is
            # ``deduplicated: True`` so the channel does NOT forward twice.
            prior = claim.prior
            if prior is not None and prior.conversation_id:
                return await _resolve_duplicate_turn(
                    runtime,
                    inbound_event_id=inbound_event_id,
                    conversation_id=prior.conversation_id,
                )
            # In-flight duplicate (or a prior row with no conversation): accept
            # but produce no side effects.
            return web.json_response(
                {
                    "ok": True,
                    "answer": (prior.answer or "") if prior else "",
                    "retrieval_used": False,
                    "decision": (prior.decision or "") if prior else "",
                    "reason": (prior.reason or "") if prior else "",
                    "conversation_id": prior.conversation_id if prior else None,
                    "case_action": prior.case_action if prior else None,
                    "deduplicated": True,
                }
            )

    # ---- Resolve the conversation BEFORE any LLM/retrieval work -------
    # PostgreSQL is the source of truth for conversation ownership. If an
    # existing conversation for this thread is HUMAN-owned, the AI must not
    # run: no retrieval, no LLM call, no assistant message, no AI answer.
    # The turn is still persisted (USER message, exactly once), the existing
    # FSM transition for a human turn is applied, and the routing signal
    # ``case_action = "HUMAN_FORWARD"`` is returned so the channel forwards
    # the ORIGINAL user text to the operator instead of an AI reply.
    async with session_factory() as session:
        session_service = SessionService(session)

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

        # The immutable FSM triple the runtime projects the turn against,
        # read from the conversation row BEFORE any engine work. The runtime
        # never touches PostgreSQL for it.
        projection_state = (
            ConversationMode(conv.conversation_mode),
            SupportStatus(conv.support_status),
            (
                None
                if conv.handoff_status is None
                else HandoffStatus(conv.handoff_status)
            ),
        )

        # A freshly-created active conversation MUST be persisted before
        # the LLM/human work runs: the message rows written in TX2 carry a
        # FK to ``conversations``, so an uncommitted INSERT would be rolled
        # back when this block exits and the later message append would
        # fail with a ForeignKeyViolationError. Committing here is a no-op
        # when the conversation already existed (no dirty writes).
        await session.commit()

    if mode == ConversationMode.HUMAN.value:
        # ---- HUMAN_FORWARD: no retrieval, no LLM, no assistant message --
        async with session_factory() as session:
            message_service = MessageService(session)
            state_service = SupportStateService(session)
            idem = IdempotencyService(session)

            # Persist the incoming USER message EXACTLY once.
            await message_service.append(
                conversation_id, MessageRole.USER, message, external_user_id
            )

            # Apply the SAME FSM transition as a human-owned turn; the
            # backend never invents a new state here.
            outcome_obj = await state_service.handle(
                SupportEvent.USER_MESSAGE_HUMAN, conversation_id
            )

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
                if outcome_obj.state is not None
                and outcome_obj.state[2] is not None
                else None
            )

            # Idempotency: record the turn WITH conversation_id but with
            # assistant_message_id=None (no assistant message exists), so a
            # replay is deduplicated without re-forwarding. The lifecycle
            # snapshot is written ATOMICALLY here so a replay is served
            # verbatim (LEVEL A) without re-deriving mutable current state.
            if inbound_event_id:
                await idem.finalize(
                    inbound_event_id=inbound_event_id,
                    outcome="forwarded",
                    conversation_id=conversation_id,
                    assistant_message_id=None,
                    decision="",
                    reason="",
                    case_action="HUMAN_FORWARD",
                    retrieval_used=False,
                    conversation_mode=final_mode,
                    support_status=final_status,
                    handoff_status=final_handoff,
                    external_case_id=conv.external_case_id,
                )

            await session.commit()

        print()
        print("=" * 60)
        print("➡️ HUMAN_FORWARD (no AI reply)")
        print("=" * 60)
        print(message)
        print()

        return web.json_response(
            {
                "ok": True,
                "answer": "",
                "retrieval_used": False,
                "decision": "",
                "reason": "",
                "conversation_id": conversation_id,
                "conversation_mode": final_mode,
                "support_status": final_status,
                "handoff_status": final_handoff,
                "case_action": "HUMAN_FORWARD",
                # Contract: the channel must forward the ORIGINAL user text
                # to the EXISTING Tracker case attached to THIS conversation;
                # it must NEVER create a new issue. The key is the backend's
                # PostgreSQL source of truth, not a channel-side guess.
                "external_case_id": conv.external_case_id,
                "deduplicated": False,
            }
        )

    # ---- Load bounded conversation history ---------------------------
    async with session_factory() as session:
        message_service = MessageService(session)

        recent_messages = await message_service.list_recent_for_conversation(
            conversation_id,
            limit=10,
        )

    history = build_conversation_history(recent_messages)

    # ---- THE TURN, outside any DB transaction ------------------------
    # The ONE runtime boundary decides and executes the turn (analysis ->
    # global policy -> exactly one family -> projection). Its ``support_core``
    # writes procedure state / the recovery record in its own bounded
    # transactions; no transaction is held across the engine call.
    try:
        delivery = await runtime.execute_fresh_turn(
            conversation_id=conversation_id,
            message=message,
            projection_state=projection_state,
            history=history,
            execution_id=inbound_event_id or None,
        )
    except ProcedureStateConflict:
        # The engine's optimistic CAS lost a race: another writer already owns
        # a newer procedure state for this conversation. Nothing is retried and
        # nothing is persisted -- NO second engine call, NO USER/ASSISTANT
        # message, NO FSM transition, and the inbound event stays ``received``
        # (it is never falsely finalized). Surface a STABLE conflict signal so
        # the channel can reconcile instead of reporting a generic failure.
        print()
        print("Procedure state conflict; not retrying or re-running the engine.")
        return web.json_response(
            {"ok": False, "error": "procedure_state_conflict"},
            status=409,
        )

    if isinstance(delivery, FinalizedDelivery):
        # The fresh path never serves a finalized replay; fail closed rather
        # than guess.
        raise RuntimeError("unexpected finalized delivery on the fresh path")

    turn = delivery.turn

    if turn.decision is None:
        # PROCEDURE_NOT_FOUND: an internal-consistency failure, NOT a turn.
        # Nothing is persisted, the FSM does not move and no answer is
        # fabricated for the user.
        print()
        print("❌ Support turn internal error:", turn.reason)
        return web.json_response(
            {"ok": False, "error": "internal_error"},
            status=500,
        )

    # ---- TX2: persist turn + apply the selected event + finalize -----
    async with session_factory() as session:
        message_service = MessageService(session)
        state_service = SupportStateService(session)
        idem = IdempotencyService(session)

        await message_service.append(
            conversation_id, MessageRole.USER, message, external_user_id
        )

        assistant_message_id = await message_service.append(
            conversation_id, MessageRole.ASSISTANT, turn.text
        )

        # The runtime owns WHICH FSM event; the transport only applies it
        # (or, for a NO_EFFECT turn, applies nothing and keeps the current
        # state). The transition table (``support.domain``) stays the single
        # validator.
        state_after = None
        if turn.fsm_event is not None:
            outcome_obj = await state_service.handle(
                turn.fsm_event, conversation_id
            )
            state_after = outcome_obj.state

        if state_after is not None:
            final_mode = state_after[0].value
            final_status = state_after[1].value
            final_handoff = (
                state_after[2].value if state_after[2] is not None else None
            )
        else:
            final_mode = conv.conversation_mode
            final_status = conv.support_status
            final_handoff = conv.handoff_status

        case_action = turn.case_action.value
        outcome = (
            "escalated"
            if case_action in ("CREATE_NEW", "RESUME")
            else "answered"
        )
        retrieval_used = _turn_retrieval_used(turn)

        if inbound_event_id:
            await idem.finalize(
                inbound_event_id=inbound_event_id,
                outcome=outcome,
                conversation_id=conversation_id,
                assistant_message_id=assistant_message_id,
                decision=turn.decision,
                reason=turn.reason,
                case_action=case_action,
                retrieval_used=retrieval_used,
                conversation_mode=final_mode,
                support_status=final_status,
                handoff_status=final_handoff,
                external_case_id=conv.external_case_id,
            )

        await session.commit()

    print()
    print("=" * 60)
    print("🤖 FINAL ANSWER")
    print("=" * 60)
    print(turn.text)
    print()

    return web.json_response(
        _turn_response_body(
            turn,
            conversation_id=conversation_id,
            final_mode=final_mode,
            final_status=final_status,
            final_handoff=final_handoff,
            external_case_id=conv.external_case_id,
            deduplicated=False,
        )
    )


async def session_start_handler(request: web.Request) -> web.Response:
    """Atomically reset one external thread and return its new conversation."""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "invalid_json"}, status=400)

    channel = str(body.get("channel") or "").strip()
    external_user_id = str(body.get("external_user_id") or "").strip()
    external_conversation_id = str(
        body.get("external_conversation_id") or ""
    ).strip()
    inbound_event_id = str(body.get("inbound_event_id") or "").strip()

    missing = [
        name
        for name, value in (
            ("channel", channel),
            ("external_user_id", external_user_id),
            ("external_conversation_id", external_conversation_id),
            ("inbound_event_id", inbound_event_id),
        )
        if not value
    ]
    if missing:
        return web.json_response(
            {"ok": False, "error": "required_fields", "fields": missing},
            status=400,
        )

    session_factory = get_session_factory()
    async with session_factory() as session:
        idem = IdempotencyService(session)
        claim = await idem.claim(inbound_event_id, channel)
        if not claim.claimed:
            prior = claim.prior
            if (
                prior is None
                or prior.conversation_id is None
                or prior.case_action != "START_NEW_SESSION"
            ):
                return web.json_response(
                    {"ok": False, "error": "start_event_incomplete"},
                    status=409,
                )
            return web.json_response(
                {
                    "ok": True,
                    "conversation_id": prior.conversation_id,
                    "conversation_mode": ConversationMode.AI.value,
                    "support_status": SupportStatus.ACTIVE.value,
                    "handoff_status": None,
                    "deduplicated": True,
                }
            )

        try:
            conversation = await SessionService(session).close_and_open_new(
                channel,
                external_user_id,
                external_conversation_id,
            )
            await idem.finalize(
                inbound_event_id=inbound_event_id,
                outcome="answered",
                conversation_id=conversation.conversation_id,
                assistant_message_id=None,
                decision="START_NEW_SESSION",
                reason="START_NEW_SESSION",
                case_action="START_NEW_SESSION",
            )
            await session.commit()
        except Exception:
            await session.rollback()
            return web.json_response(
                {"ok": False, "error": "session_start_failed"},
                status=500,
            )

    return web.json_response(
        {
            "ok": True,
            "conversation_id": conversation.conversation_id,
            "conversation_mode": conversation.conversation_mode,
            "support_status": conversation.support_status,
            "handoff_status": conversation.handoff_status,
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
    print("ЁЯТм NEW CHAT REQUEST")
    print("=" * 60)
    print(f"ЁЯСд User: {message}")

    # ---- Stateful mode (channel present) -----------------------------
    # Legacy {"message": "..."} callers have no channel and keep the
    # exact old stateless behaviour, including no dedup.
    if str(body.get("channel") or "").strip():
        try:
            return await _handle_stateful_turn(body)
        except Exception as exc:
            print()
            print("тЭМ Stateful chat error:", repr(exc))
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
            "тЭМ Chat error:",
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
    print("ЁЯдЦ FINAL ANSWER")
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

    app.router.add_post(
        "/api/session/start",
        session_start_handler,
    )

    app.router.add_route(
        "OPTIONS",
        "/api/chat",
        chat_handler,
    )

    app.router.add_route(
        "OPTIONS",
        "/api/session/start",
        session_start_handler,
    )

    return app


def main() -> None:
    print("=" * 60)
    print(" AngioPlus тАФ Yandex Text Backend")
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
