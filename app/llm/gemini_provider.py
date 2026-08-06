# -*- coding: utf-8 -*-
from __future__ import annotations

import json as _json
from typing import Any

from google import genai
from google.genai import types

from app.llm.base import LLMProvider
from app.llm.errors import LLMError, LLMErrorType
from app.llm.models import GateResult

DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"


class GeminiProvider(LLMProvider):
    """Gemini-backed LLM provider.

    This is a 1:1 port of the LLM logic that previously lived directly in
    app/main.py (direct-answer gate + main answer generation).
    """

    def __init__(self, api_key: str, model: str = DEFAULT_GEMINI_MODEL) -> None:
        # Constructing the client does not perform any network call.
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def check_direct_answer(
        self,
        question: str,
        context: str,
    ) -> GateResult:
        prompt = (
            "You decide whether the RETRIEVED CONTEXT directly answers the user question.\n"
            "Answer only with valid JSON: {\"direct_answer\": true|false, \"reason\": \"short\"}\n"
            "Rules:\n"
            "- direct_answer=true ONLY if the context directly contains the answer.\n"
            "- Topical similarity is NOT enough.\n"
            "- For 'How to do X?' / 'Как выполнить X?' questions, "
            "direct_answer=true if the context directly contains AT LEAST ONE "
            "of the following:\n"
            "  (1) a step-by-step procedure;\n"
            "  (2) a statement that the action is performed only by an "
            "authorized specialist;\n"
            "  (3) a direct prohibition on doing it yourself;\n"
            "  (4) the exact ordering/contact procedure;\n"
            "  (5) a required precondition;\n"
            "  (6) a description of who performs the action, where, or under "
            "what conditions.\n"
            "- For 'How to do X?' / 'Как выполнить X?' questions, "
            "direct_answer=false only if the topic is merely mentioned and "
            "there is no practical answer, no procedure, no restriction, no "
            "responsible party, and no preconditions.\n"
            "- direct_answer=true ONLY if the practical answer, restriction, "
            "responsible party, or condition explicitly applies to the REQUESTED "
            "action X.\n"
            "- PROHIBITED:\n"
            "  * do NOT transfer a rule about a neighbouring action to the "
            "requested action X;\n"
            "  * do NOT treat 'after X' information as the procedure for "
            "performing X;\n"
            "  * do NOT treat a mention of repair/maintenance as an answer to "
            "a question about updating;\n"
            "  * do NOT merge separate sentences into a new cause-and-effect "
            "relationship that the text does not state.\n"
            "- For identification questions ('Кто такой/такие X?', 'Что такое "
            "X?', 'Какая это компания?', 'Кому принадлежит продукт/торговый "
            "знак?'), direct_answer=true if the context directly indicates at "
            "least ONE of the following:\n"
            "  (1) X is a company/manufacturer/developer;\n"
            "  (2) X provides support or maintenance for the product;\n"
            "  (3) the product or trademark belongs to X;\n"
            "  (4) the official full company name of X is stated;\n"
            "  (5) the role of X with respect to AngioPlus Core is described.\n"
            "- For identification questions, do NOT require a full corporate "
            "profile, company history, or detailed business description; a "
            "directly supported identifier is enough.\n"
            "- Do NOT merge separate sentences into claims the context does "
            "not state.\n"
            "- Do NOT reveal credentials in the reason; a fact about who "
            "issues them is enough.\n\n"
            "EXAMPLES:\n"
            "Question: 'Как установить AngioPlus Core?' Context: 'установка "
            "выполняется исключительно сертифицированным инженером.' -> "
            "{\"direct_answer\": true}\n"
            "Question: 'Как обновить AngioPlus Core?' Context: only "
            "installation steps and system requirements -> "
            "{\"direct_answer\": false}\n"
            "Question: 'Как войти в систему?' Context: 'учётные данные выдаёт "
            "администратор.' -> {\"direct_answer\": true}\n"
            "Question: 'Как обновить AngioPlus Core?' Context: 'Ремонт и "
            "техническое обслуживание выполняются авторизованным специалистом. "
            "После обновления проверьте вход, электронный ключ и анализ "
            "данных.' -> {\"direct_answer\": false} (context describes "
            "post-update checks, not the update procedure, and does not say "
            "who performs the update)\n"
            "Question: 'Что проверить после обновления AngioPlus Core?' "
            "Context: 'Ремонт и техническое обслуживание выполняются "
            "авторизованным специалистом. После обновления проверьте вход, "
            "электронный ключ и анализ данных.' -> {\"direct_answer\": true}\n"
            "Question: 'Кто выполняет техническое обслуживание?' Context: "
            "'Техническое обслуживание выполняется авторизованным "
            "специалистом.' -> {\"direct_answer\": true}\n"
            "Question: 'Кто такие Pulse Medical?' Context: 'Shanghai Pulse "
            "Medical Technology, Inc. Компания Pulse Medical предоставляет "
            "техническую поддержку. AngioPlus и µFR являются "
            "зарегистрированными торговыми знаками Shanghai Pulse Medical "
            "Technology, Inc.' -> {\"direct_answer\": true}\n"
            "Question: 'Кто такие Pulse Medical?' Context: 'Для установки "
            "обратитесь к представителю.' -> {\"direct_answer\": false}\n"
            "Question: 'Что такое DICOM?' Context: 'DICOM — это стандарт "
            "цифровой передачи медицинской информации.' -> "
            "{\"direct_answer\": true}\n\n"
            "RETRIEVED CONTEXT:\n"
            "------------------------\n"
            f"{context}\n"
            "------------------------\n"
            "USER QUESTION:\n"
            f"{question}\n"
        )

        try:
            try:
                schema = {
                    "type": "OBJECT",
                    "properties": {
                        "direct_answer": {"type": "BOOLEAN"},
                        "reason": {"type": "STRING"},
                    },
                    "required": ["direct_answer", "reason"],
                }
                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                )
            except (TypeError, ValueError):
                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                )

            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=config,
            )
            raw = (response.text or "").strip()
            if raw.startswith("```"):
                lines = raw.splitlines()
                if lines and lines[0].strip().startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                raw = "\n".join(lines).strip()

            parsed: Any = _json.loads(raw)
            return GateResult(**parsed)
        except LLMError:
            raise
        except Exception as exc:
            raise _map_error(exc) from exc

    def generate_answer(
        self,
        question: str,
        context: str,
        system_prompt: str,
    ) -> str:
        prompt = f"""
--------------------
RETRIEVED CONTEXT / НАЙДЕННЫЙ КОНТЕКСТ
--------------------

{context}

--------------------
USER QUESTION / ВОПРОС ПОЛЬЗОВАТЕЛЯ
--------------------

{question}
""".strip()

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                ),
            )
            answer = response.text
            if not answer:
                return "Такой информации нет в имеющейся документации."
            return answer.strip()
        except LLMError:
            raise
        except Exception as exc:
            raise _map_error(exc) from exc


def _map_error(exc: Exception) -> LLMError:
    """Translate a google.genai / network exception into a single LLMError."""
    from google.genai import errors as genai_errors

    status = str(getattr(exc, "status", "") or "").upper()
    code = getattr(exc, "code", None)

    # Authentication (401/403 / UNAUTHENTICATED / PERMISSION_DENIED).
    if (
        isinstance(exc, genai_errors.ClientError) and code in (401, 403)
    ) or status in ("UNAUTHENTICATED", "PERMISSION_DENIED"):
        return LLMError(
            LLMErrorType.AUTHENTICATION,
            "LLM authentication failed",
            code,
        )

    # Quota / rate limit.
    if code == 429 or status == "RESOURCE_EXHAUSTED":
        return LLMError(LLMErrorType.QUOTA_EXCEEDED, "LLM quota exceeded", code)

    # Timeout / network errors.
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return LLMError(
            LLMErrorType.TIMEOUT_OR_NETWORK, "LLM timeout/network error", code
        )
    try:
        import httpx

        if isinstance(
            exc,
            (
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.ReadError,
                httpx.RemoteProtocolError,
            ),
        ):
            return LLMError(
                LLMErrorType.TIMEOUT_OR_NETWORK,
                "LLM network error",
                code,
            )
    except Exception:
        pass

    # Temporary availability (5xx / UNAVAILABLE / INTERNAL / DEADLINE_EXCEEDED).
    if (
        isinstance(exc, genai_errors.ServerError)
        or status in {"DEADLINE_EXCEEDED", "UNAVAILABLE", "INTERNAL"}
        or code in (500, 502, 503, 504)
    ):
        return LLMError(
            LLMErrorType.TEMPORARY_UNAVAILABLE,
            "LLM temporarily unavailable",
            code,
        )

    # Invalid JSON / schema (mostly the gate path) surfaces as INVALID_RESPONSE.
    if isinstance(exc, _json.JSONDecodeError) or "ValidationError" in type(
        exc
    ).__name__:
        return LLMError(
            LLMErrorType.INVALID_RESPONSE,
            "LLM returned an invalid response",
            code,
        )

    # Default: unexpected/internal error.
    return LLMError(
        LLMErrorType.UNEXPECTED_INTERNAL,
        "Unexpected LLM provider error",
        code,
    )
