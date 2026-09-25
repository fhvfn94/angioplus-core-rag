"""
Yandex Alice analyzer provider -- Slice 11A.2 adapter over the EXISTING
Yandex Alice Responses API for the frozen Slice 4 ``AnalyzerProvider`` port.

Infrastructure only. NOT imported by any production runtime path
(server.py, alice_support_agent.py, yandex/realtime, n8n, Tracker, DB,
retrieval, File Search). This module is the provider-specific
implementation of the frozen analyzer boundary

    AnalyzerProvider = Callable[[str, Sequence[Mapping[str, str]]], str]

already declared in :mod:`support.alice_analyzer`.

It is a THIN, SINGLE-CALL TRANSPORT and nothing else. It performs EXACTLY
one mechanical transformation:

    prompt + history  ->  Yandex model  ->  RAW JSON text

and then STOPS. The raw JSON text is handed straight back to the frozen
Slice 4 parser (:func:`support.alice_analyzer.parse_semantic_output`),
which owns all validation. This adapter:

  * receives the COMPLETE Slice 4 prompt and sends it VERBATIM as the sole
    user message -- the history is ALREADY rendered into that prompt by
    :func:`support.alice_analyzer.build_analyzer_prompt`, so it is never
    re-sent or re-interpreted here;
  * returns the RAW ``output_text`` unchanged -- it performs NO parsing,
    NO JSON validation, NO enum checking and NO decision;
  * selects NO procedure, touches NO ProcedureCatalog, and never routes;
  * makes NO support-policy decision, retrieves NO knowledge, attaches NO
    File Search / tools and mutates NO state;
  * generates NO final user-facing answer: the renderer (Slice 6) alone
    owns user-facing wording;
  * any SDK / auth / transport / response failure PROPAGATES unchanged, so
    the frozen :func:`~support.alice_analyzer.analyze` maps it to its
    deterministic fail-safe observation (never a crash).

Model, endpoint and transport defaults are PINNED here: there is no model
environment switch and no per-request model override. The canonical Alice
boundary is ``https://ai.api.cloud.yandex.net/v1`` and the model id is
``gpt://<YANDEX_FOLDER_ID>/aliceai-llm-flash/latest``. Only ``YANDEX_FOLDER_ID``
and ``YANDEX_API_KEY`` are read from the environment. The SDK is imported
LAZILY so this module -- and its unit tests -- import without the SDK and
without credentials.

The output is constrained with the strict JSON schema the Slice 4 parser
accepts (exactly seven top-level fields, no extras), so a well-behaved model
can only ever produce the shape the parser validates.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Sequence


# Canonical Alice boundary and model. Both are PINNED: no env switch, no
# per-request override (see the module docstring).
DEFAULT_BASE_URL = "https://ai.api.cloud.yandex.net/v1"
MODEL_ID_TEMPLATE = "gpt://{folder_id}/aliceai-llm-flash/latest"

# Transport defaults. These are NOT a production SLA / retry policy: they
# only bound a single SDK call so an unresponsive transport cannot hang a
# request forever. Exactly ONE HTTP attempt is ever made (max_retries=0).
DEFAULT_TIMEOUT_SECONDS = 60.0
# Mirrors the production semantic stage's output budget.
DEFAULT_MAX_OUTPUT_TOKENS = 1000

# The strict Slice 4 schema name (used for the Responses API JSON format).
SEMANTIC_FORMAT_NAME = "angioplus_semantic_observation"

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


def build_analyzer_model_id(folder_id: str) -> str:
    """The canonical Alice model id for ``folder_id`` (pinned template)."""
    if not isinstance(folder_id, str) or not folder_id.strip():
        raise ValueError("folder_id must be a non-blank string.")
    return MODEL_ID_TEMPLATE.format(folder_id=folder_id.strip())


def _semantic_format() -> Dict[str, Any]:
    """A fresh strict JSON envelope for one request.

    This is the INNER ``format`` object of the Responses API ``text``
    config (``text={"format": {...}}``), i.e. the exact shape the other
    canonical Alice call sites in this repo use. The schema mirrors the
    FROZEN Slice 4 output contract (the seven top-level fields and the
    ``{step_id, outcome}`` action shape) so the model cannot drift from
    what :func:`support.alice_analyzer.parse_semantic_output` validates.
    Built per call so no shared mutable structure can leak between calls.
    """
    return {
        "type": "json_schema",
        "name": SEMANTIC_FORMAT_NAME,
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "scope": {
                    "type": "string",
                    "enum": ["in_scope", "out_of_scope", "unclear"],
                },
                "reported_actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "step_id": {"type": ["string", "null"]},
                            "outcome": {
                                "type": ["string", "null"],
                                "enum": [
                                    "failed",
                                    "succeeded",
                                    "skipped",
                                    "done_unknown",
                                    None,
                                ],
                            },
                        },
                        "required": ["step_id", "outcome"],
                    },
                },
                "problem_status": {
                    "type": ["string", "null"],
                    "enum": ["unknown", "ongoing", "resolved", None],
                },
                "human_requested": {"type": "boolean"},
                "clarification_facts": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "new_issue": {"type": ["string", "null"]},
                "request_kind": {
                    "type": "string",
                    "enum": ["problem", "information", "unclear"],
                },
            },
            "required": [
                "scope",
                "reported_actions",
                "problem_status",
                "human_requested",
                "clarification_facts",
                "new_issue",
                "request_kind",
            ],
        },
    }


def _raw_output_text(response: object) -> str:
    """The response's raw ``output_text`` -- never parsed, never screened.

    A response without text is a transport-level failure and is raised so
    the frozen :func:`~support.alice_analyzer.analyze` maps it to its
    deterministic fail-safe observation.
    """
    text = getattr(response, "output_text", None)
    if not isinstance(text, str):
        raise RuntimeError("Yandex Responses API returned no output_text.")
    return text


class YandexAnalyzerProvider:
    """The frozen Slice 4 ``AnalyzerProvider`` over the Yandex Alice API.

    Callable as ``provider(prompt, history) -> raw_json_text`` to match the
    frozen port exactly. ``client`` / ``api_key`` / ``folder_id`` are
    injectable so the adapter can be unit-tested with a fake client,
    without credentials and without the network. When no client is
    injected, the real SDK client is built LAZILY on the first call (a
    missing environment therefore surfaces inside the call and is mapped by
    :func:`support.alice_analyzer.analyze` to a fail-safe observation
    instead of crashing at construction time).
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

    def __call__(
        self,
        prompt: str,
        history: Sequence[Mapping[str, str]] = (),
    ) -> str:
        """Send the complete Slice 4 prompt and return the raw output text.

        The prompt is forwarded VERBATIM as the sole user message; nothing
        is added, removed or reinterpreted. ``history`` is accepted to match
        the frozen port but is NOT re-sent: it is already rendered into
        ``prompt`` by :func:`support.alice_analyzer.build_analyzer_prompt`.
        Exactly one SDK call is made. Any failure propagates unchanged.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-blank string.")

        model = build_analyzer_model_id(self._resolve_folder_id())
        client = self._resolve_client()

        response = client.responses.create(
            model=model,
            temperature=0,
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=self._max_output_tokens,
            store=False,
            text={"format": _semantic_format()},
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

        The SDK import is local so the module imports (and unit-tests)
        without the Yandex SDK installed.
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
    "YandexAnalyzerProvider",
    "build_analyzer_model_id",
    "DEFAULT_BASE_URL",
    "MODEL_ID_TEMPLATE",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "SEMANTIC_FORMAT_NAME",
]
