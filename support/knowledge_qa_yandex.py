"""
Yandex Knowledge QA provider -- Slice 11B.1b.3 adapter over the EXISTING
Yandex Alice Responses API for the Knowledge QA provider port.

Infrastructure only. NOT imported by any production runtime path
(server.py, alice_support_agent.py, yandex/realtime, n8n, Tracker, DB,
retrieval). This module is the provider-specific implementation of the frozen
:class:`~support.knowledge_qa.KnowledgeQAProvider` port.

It is a THIN, SINGLE-CALL TRANSPORT and nothing else:

  * it receives the COMPLETE QA prompt and sends it verbatim as the sole user
    message -- it never re-wraps, prefixes, re-formats, translates or appends
    instructions of its own;
  * it returns the RAW ``output_text`` unchanged -- it performs NO parsing,
    NO JSON validation, NO grounding and NO decision;
  * it never retrieves evidence, never selects a procedure/step, never routes,
    never escalates and never reads conversation history -- all of those gates
    belong to the Knowledge QA engine (a separate, offline module);
  * it attaches no search / no auxiliary capability and no state: the request
    carries model, temperature, input, a bounded output budget, ``store=False``
    and the strict single-field JSON-format envelope ONLY;
  * any SDK / auth / transport / response failure PROPAGATES unchanged, so the
    QA engine maps it to its deterministic fail-safe result (never a crash,
    never user-facing internals).

Model, endpoint and transport defaults are PINNED here: there is no model
environment switch and no per-request override. The canonical Alice boundary
is ``https://ai.api.cloud.yandex.net/v1`` and the model id is
``gpt://<YANDEX_FOLDER_ID>/aliceai-llm-flash/latest``. Only ``YANDEX_FOLDER_ID``
and ``YANDEX_API_KEY`` are read from the environment. The SDK is imported
LAZILY so this module -- and its unit tests -- import without the SDK and
without credentials.

The output is constrained with the strict, single-field JSON envelope
``{"answer": "..."}`` so the QA parser only ever sees the shape it accepts. An
empty / blank answer is a legal provider outcome that the QA engine
deterministically treats as ``NOT_ANSWERED``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from support.knowledge_qa import KnowledgeQAProvider


# Canonical Alice boundary and model. Both are PINNED: no env switch, no
# per-request override.
DEFAULT_BASE_URL = "https://ai.api.cloud.yandex.net/v1"
MODEL_ID_TEMPLATE = "gpt://{folder_id}/aliceai-llm-flash/latest"

# Transport defaults. These only bound a single SDK call so an unresponsive
# transport cannot hang a request forever. Exactly ONE HTTP attempt is made.
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_OUTPUT_TOKENS = 500

# The QA parser accepts exactly one field; the envelope mirrors it.
ANSWER_FIELD = "answer"
ANSWER_FORMAT_NAME = "angioplus_knowledge_qa_answer"

_ENV_API_KEY = "YANDEX_API_KEY"
_ENV_FOLDER_ID = "YANDEX_FOLDER_ID"


def _require_env(name: str) -> str:
    """Read a required, non-blank environment value (fail closed)."""
    value = os.environ.get(name, "")
    if not value.strip():
        raise RuntimeError(
            f"{name} is not configured; refusing to call the Yandex "
            "Alice API."
        )
    return value.strip()


def build_qa_model_id(folder_id: str) -> str:
    """The canonical Alice model id for ``folder_id`` (pinned template)."""
    if not isinstance(folder_id, str) or not folder_id.strip():
        raise ValueError("folder_id must be a non-blank string.")
    return MODEL_ID_TEMPLATE.format(folder_id=folder_id.strip())


def _answer_format() -> Dict[str, Any]:
    """A fresh strict single-field JSON envelope for one request.

    This is the INNER ``format`` object of the Responses API ``text`` config
    (``text={"format": {...}}``). Built per call so no shared mutable structure
    can leak between calls.
    """
    return {
        "type": "json_schema",
        "name": ANSWER_FORMAT_NAME,
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {ANSWER_FIELD: {"type": "string"}},
            "required": [ANSWER_FIELD],
            "additionalProperties": False,
        },
    }


def _raw_output_text(response: object) -> str:
    """The response's raw ``output_text`` -- never parsed, never screened.

    A response without text is a transport-level failure and is raised so the
    QA engine maps it to its deterministic fail-safe result.
    """
    text = getattr(response, "output_text", None)
    if not isinstance(text, str):
        raise RuntimeError("Yandex Responses API returned no output_text.")
    return text


class YandexKnowledgeQAProvider(KnowledgeQAProvider):
    """The Knowledge QA ``KnowledgeQAProvider`` over the Yandex Alice API.

    ``client`` / ``api_key`` / ``folder_id`` are injectable so the adapter can
    be unit-tested with a fake client, without credentials and without the
    network. When no client is injected, the real SDK client is built LAZILY on
    the first call (a missing environment therefore surfaces inside the call and
    is mapped by the QA engine to a fail-safe result instead of crashing at
    construction time).
    """

    def __init__(
        self,
        *,
        client: Optional[object] = None,
        api_key: Optional[str] = None,
        folder_id: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0.")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be greater than 0.")

        self._client = client
        self._api_key = api_key
        self._folder_id = folder_id
        self._timeout = timeout
        self._max_output_tokens = max_output_tokens

    def __call__(self, prompt: str) -> str:
        """Send the complete QA prompt and return the raw ``output_text``.

        The prompt is forwarded VERBATIM as the sole user message; nothing is
        added, removed or reinterpreted. Exactly one SDK call is made. Any
        failure propagates unchanged.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-blank string.")

        model = build_qa_model_id(self._resolve_folder_id())
        client = self._resolve_client()

        response = client.responses.create(
            model=model,
            temperature=0,
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=self._max_output_tokens,
            store=False,
            text={"format": _answer_format()},
        )

        return _raw_output_text(response)

    # ------------------------------------------------------------------
    # lazy wiring (imports / credentials happen only inside a real call)
    # ------------------------------------------------------------------

    def _resolve_folder_id(self) -> str:
        folder_id = self._folder_id
        if folder_id is None:
            folder_id = _require_env(_ENV_FOLDER_ID)
            self._folder_id = folder_id
        return folder_id

    def _resolve_client(self) -> object:
        """Return the injected client, building the SDK client at most once.

        The SDK import is local so the module imports (and unit-tests) without
        the Yandex SDK installed.
        """
        if self._client is None:
            api_key = self._api_key
            if api_key is None or not str(api_key).strip():
                api_key = _require_env(_ENV_API_KEY)

            from openai import OpenAI  # local: keep the module import SDK-free

            self._client = OpenAI(
                api_key=api_key,
                base_url=DEFAULT_BASE_URL,
                project=self._resolve_folder_id(),
                timeout=self._timeout,
                max_retries=0,
            )

        return self._client


__all__ = [
    "YandexKnowledgeQAProvider",
    "build_qa_model_id",
    "DEFAULT_BASE_URL",
    "MODEL_ID_TEMPLATE",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "ANSWER_FIELD",
    "ANSWER_FORMAT_NAME",
]
