# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from functools import cached_property
from typing import Optional

from pydantic import ValidationError

from app.llm.base import LLMProvider
from app.llm.errors import LLMError, LLMErrorType
from app.llm.models import GateResult

# ---------------------------------------------------------------------------
# Env configuration (defaults)
# ---------------------------------------------------------------------------

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_TIMEOUT = 60
DEFAULT_DEEPSEEK_MAX_RETRIES = 3

# Official DeepSeek models must be set explicitly via DEEPSEEK_MODEL.
# Legacy deepseek-chat / deepseek-reasoner are intentionally NOT hard-coded.


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


class DeepSeekProvider(LLMProvider):
    """DeepSeek-backed LLM provider (OpenAI-compatible SDK, openai==2.52.0).

    Uses DeepSeek's JSON Output via `response_format={"type": "json_object"}`
    with strict re-validation through `GateResult`. The OpenAI client is
    created lazily (cached_property), so constructing this provider performs
    no network call.
    """

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = (
            os.getenv("DEEPSEEK_BASE_URL") or DEFAULT_DEEPSEEK_BASE_URL
        ).strip()
        self._timeout = _env_float(
            "DEEPSEEK_TIMEOUT", DEFAULT_DEEPSEEK_TIMEOUT
        )
        self._max_retries = _env_int(
            "DEEPSEEK_MAX_RETRIES", DEFAULT_DEEPSEEK_MAX_RETRIES
        )

    @cached_property
    def _client(self):
        """Lazily create the OpenAI-compatible client (no network here)."""
        from openai import OpenAI

        return OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )

    # -- direct-answer gate -------------------------------------------------

    def check_direct_answer(self, question: str, context: str) -> GateResult:
        prompt = (
            "You decide whether the RETRIEVED CONTEXT directly answers the "
            "user question. Answer ONLY with a single JSON object.\n"
            "Your reply must be valid JSON and must match this exact schema:\n"
            '{"direct_answer": true, "reason": "short"}\n'
            "Rules:\n"
            "- direct_answer=true ONLY if the context directly contains the "
            "answer.\n"
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
            "issues them is enough.\n"
            "- reason must be a short string (1-200 chars).\n\n"
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
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
            )
        except LLMError:
            raise
        except Exception as exc:
            raise _map_error(exc) from exc

        choice = response.choices[0] if response.choices else None
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason in ("length", "content_filter"):
            raise LLMError(
                LLMErrorType.INVALID_RESPONSE,
                "LLM response was truncated or filtered",
                None,
            )

        content = getattr(choice.message, "content", None) if choice else None
        if not content or not content.strip():
            raise LLMError(
                LLMErrorType.INVALID_RESPONSE,
                "LLM returned an empty response",
                None,
            )

        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError) as exc:
            raise _map_error(exc) from exc

        try:
            return GateResult.model_validate(data, strict=True)
        except ValidationError as exc:
            raise _map_error(exc) from exc

    # -- answer generation --------------------------------------------------

    def generate_answer(
        self,
        question: str,
        context: str,
        system_prompt: str,
    ) -> str:
        user_content = (
            "RETRIEVED CONTEXT / НАЙДЕННЫЙ КОНТЕКСТ\n"
            "--------------------\n"
            f"{context}\n\n"
            "USER QUESTION / ВОПРОС ПОЛЬЗОВАТЕЛЯ\n"
            "--------------------\n"
            f"{question}"
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                extra_body={"thinking": {"type": "disabled"}},
            )
        except LLMError:
            raise
        except Exception as exc:
            raise _map_error(exc) from exc

        message = response.choices[0].message if response.choices else None
        content = getattr(message, "content", None) if message else None
        if not content or not content.strip():
            return "Такой информации нет в имеющейся документации."

        return content.strip()


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------

_TRANSIENT_5XX = (500, 502, 503, 504)


def _map_error(exc: Exception) -> LLMError:
    """Translate an openai SDK / network exception into a single LLMError."""
    from openai import APIStatusError

    # Network / timeout errors (openai SDK + httpx).
    if _is_timeout_or_network(exc):
        return LLMError(
            LLMErrorType.TIMEOUT_OR_NETWORK,
            "LLM timeout/network error",
            _status(exc),
        )

    if isinstance(exc, APIStatusError):
        code = exc.status_code
        if code in (401, 403):
            return LLMError(
                LLMErrorType.AUTHENTICATION,
                "LLM authentication failed",
                code,
            )
        if code == 402 or code == 429:
            # 402 = insufficient balance, 429 = rate limit.
            return LLMError(
                LLMErrorType.QUOTA_EXCEEDED,
                "LLM quota exceeded",
                code,
            )
        if code in _TRANSIENT_5XX:
            return LLMError(
                LLMErrorType.TEMPORARY_UNAVAILABLE,
                "LLM temporarily unavailable",
                code,
            )
        if code in (400, 422):
            return LLMError(
                LLMErrorType.UNEXPECTED_INTERNAL,
                "LLM request configuration error",
                code,
            )
        return LLMError(
            LLMErrorType.UNEXPECTED_INTERNAL,
            "Unexpected LLM provider error",
            code,
        )

    # Invalid JSON / Pydantic validation / empty content.
    if isinstance(exc, (json.JSONDecodeError, ValidationError, TypeError)):
        return LLMError(
            LLMErrorType.INVALID_RESPONSE,
            "LLM returned an invalid response",
            None,
        )

    # Default: unexpected/internal error.
    return LLMError(
        LLMErrorType.UNEXPECTED_INTERNAL,
        "Unexpected LLM provider error",
        _status(exc),
    )


def _is_timeout_or_network(exc: Exception) -> bool:
    from openai import APIConnectionError, APITimeoutError

    if isinstance(exc, (APITimeoutError, APIConnectionError, TimeoutError)):
        return True
    if isinstance(exc, ConnectionError):
        return True
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
            return True
    except Exception:
        pass
    return False


def _status(exc: Exception) -> Optional[int]:
    """Return the HTTP/API status code if the exception exposes one."""
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    return None
