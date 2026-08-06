# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from functools import cached_property
from typing import Union

from pydantic import ValidationError

from app.llm.base import LLMProvider
from app.llm.errors import LLMError, LLMErrorType
from app.llm.models import GateResult

# ---------------------------------------------------------------------------
# Env configuration (defaults)
# ---------------------------------------------------------------------------

DEFAULT_GIGACHAT_SCOPE = "GIGACHAT_API_PERS"
DEFAULT_GIGACHAT_BASE_URL = "https://api.giga.chat/v1"
DEFAULT_GIGACHAT_AUTH_URL = (
    "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
)
DEFAULT_GIGACHAT_VERIFY_SSL_CERTS = True
DEFAULT_GIGACHAT_TIMEOUT = 60
DEFAULT_GIGACHAT_MAX_RETRIES = 3
DEFAULT_GIGACHAT_RETRY_BACKOFF_FACTOR = 0.5


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


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


class GigaChatProvider(LLMProvider):
    """GigaChat-backed LLM provider (official SDK gigachat==0.2.3).

    The SDK client is created lazily (cached_property) so that constructing
    this provider performs no network / OAuth call.
    """

    def __init__(self, credentials: str, model: str) -> None:
        self._credentials = credentials
        self._model = model
        self._scope = (
            os.getenv("GIGACHAT_SCOPE") or DEFAULT_GIGACHAT_SCOPE
        ).strip()
        self._base_url = (
            os.getenv("GIGACHAT_BASE_URL") or DEFAULT_GIGACHAT_BASE_URL
        ).strip()
        self._auth_url = (
            os.getenv("GIGACHAT_AUTH_URL") or DEFAULT_GIGACHAT_AUTH_URL
        ).strip()
        self._verify_ssl_certs = _env_bool(
            "GIGACHAT_VERIFY_SSL_CERTS", DEFAULT_GIGACHAT_VERIFY_SSL_CERTS
        )
        self._ca_bundle_file = (os.getenv("GIGACHAT_CA_BUNDLE_FILE") or "").strip() or None
        self._timeout = _env_float("GIGACHAT_TIMEOUT", DEFAULT_GIGACHAT_TIMEOUT)
        self._max_retries = _env_int(
            "GIGACHAT_MAX_RETRIES", DEFAULT_GIGACHAT_MAX_RETRIES
        )
        self._retry_backoff_factor = _env_float(
            "GIGACHAT_RETRY_BACKOFF_FACTOR",
            DEFAULT_GIGACHAT_RETRY_BACKOFF_FACTOR,
        )

    @cached_property
    def _client(self):
        """Lazily create the GigaChat SDK client (no network at construction)."""
        from gigachat import GigaChat

        return GigaChat(
            credentials=self._credentials,
            scope=self._scope,
            model=self._model,
            base_url=self._base_url,
            auth_url=self._auth_url,
            verify_ssl_certs=self._verify_ssl_certs,
            ca_bundle_file=self._ca_bundle_file,
            timeout=self._timeout,
            max_retries=self._max_retries,
            retry_backoff_factor=self._retry_backoff_factor,
        )

    # -- direct-answer gate -------------------------------------------------

    def check_direct_answer(self, question: str, context: str) -> GateResult:
        from gigachat.models.chat_completions import (
            ChatCompletionRequest,
            ChatMessage,
        )

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

        request = ChatCompletionRequest(
            model=self._model,
            messages=[
                ChatMessage(role="user", content=prompt),
            ],
        )

        try:
            response, parsed = self._client.chat.parse(
                request,
                response_format=GateResult,
                strict=True,
            )
            # Additional strict validation on top of the SDK result.
            del response
            validated = GateResult.model_validate(
                parsed.model_dump(),
                strict=True,
            )
            return validated
        except LLMError:
            raise
        except ValidationError as exc:
            raise _map_error(exc) from exc
        except json.JSONDecodeError as exc:
            raise _map_error(exc) from exc
        except Exception as exc:
            raise _map_error(exc) from exc

    # -- answer generation --------------------------------------------------

    def generate_answer(
        self,
        question: str,
        context: str,
        system_prompt: str,
    ) -> str:
        from gigachat.models.chat_completions import (
            ChatCompletionRequest,
            ChatMessage,
        )

        user_content = (
            "RETRIEVED CONTEXT / НАЙДЕННЫЙ КОНТЕКСТ\n"
            "--------------------\n"
            f"{context}\n\n"
            "USER QUESTION / ВОПРОС ПОЛЬЗОВАТЕЛЯ\n"
            "--------------------\n"
            f"{question}"
        )

        request = ChatCompletionRequest(
            model=self._model,
            messages=[
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_content),
            ],
        )

        try:
            response = self._client.chat.create(request)
        except LLMError:
            raise
        except Exception as exc:
            raise _map_error(exc) from exc

        return self._extract_text(response)

    def _extract_text(self, response) -> str:
        """Extract assistant text from a ChatCompletionResponse (SDK 0.2.3)."""
        messages = getattr(response, "messages", None)
        if not messages:
            return "Такой информации нет в имеющейся документации."

        content = getattr(messages[0], "content", None)
        if not content:
            return "Такой информации нет в имеющейся документации."

        text = "".join(getattr(part, "text", "") or "" for part in content).strip()
        if not text:
            return "Такой информации нет в имеющейся документации."

        return text


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------

_TRANSIENT_5XX = (500, 502, 503, 504)


def _map_error(exc: Exception) -> LLMError:
    """Translate a gigachat SDK / network exception into a single LLMError."""
    from gigachat.exceptions import (
        AuthenticationError,
        ForbiddenError,
        LengthFinishReasonError,
        ModelNotSpecifiedError,
        RateLimitError,
        ServerError,
    )

    # Authentication (401/403).
    if isinstance(exc, (AuthenticationError, ForbiddenError)):
        return LLMError(
            LLMErrorType.AUTHENTICATION,
            "LLM authentication failed",
            _status(exc),
        )

    # Exceptions without a status_code mapping by type.
    if isinstance(exc, LengthFinishReasonError):
        return LLMError(
            LLMErrorType.INVALID_RESPONSE,
            "LLM response was truncated",
            None,
        )
    if isinstance(exc, ModelNotSpecifiedError):
        return LLMError(
            LLMErrorType.UNEXPECTED_INTERNAL,
            "LLM configuration error",
            None,
        )

    # Status-code based mapping (ResponseError and subclasses).
    code = _status(exc)
    if code == 401 or code == 403:
        return LLMError(
            LLMErrorType.AUTHENTICATION,
            "LLM authentication failed",
            code,
        )
    if code == 429:
        return LLMError(LLMErrorType.QUOTA_EXCEEDED, "LLM quota exceeded", code)
    if code in _TRANSIENT_5XX:
        return LLMError(
            LLMErrorType.TEMPORARY_UNAVAILABLE,
            "LLM temporarily unavailable",
            code,
        )

    if isinstance(exc, RateLimitError):
        return LLMError(LLMErrorType.QUOTA_EXCEEDED, "LLM quota exceeded", None)
    if isinstance(exc, ServerError):
        return LLMError(
            LLMErrorType.TEMPORARY_UNAVAILABLE,
            "LLM temporarily unavailable",
            _status(exc),
        )

    # Timeout / network errors from the HTTP layer.
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return LLMError(
            LLMErrorType.TIMEOUT_OR_NETWORK,
            "LLM timeout/network error",
            None,
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
                None,
            )
    except Exception:
        pass

    # Invalid JSON / Pydantic validation surfaces as INVALID_RESPONSE.
    if isinstance(exc, (json.JSONDecodeError,)) or "ValidationError" in type(
        exc
    ).__name__:
        return LLMError(
            LLMErrorType.INVALID_RESPONSE,
            "LLM returned an invalid response",
            _status(exc),
        )

    # Default: unexpected/internal error.
    return LLMError(
        LLMErrorType.UNEXPECTED_INTERNAL,
        "Unexpected LLM provider error",
        _status(exc),
    )


def _status(exc: Exception):
    """Return the HTTP status code if the exception exposes one."""
    code = getattr(exc, "status_code", None)
    if code is not None:
        return code
    return getattr(exc, "code", None)
