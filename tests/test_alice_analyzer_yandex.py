# -*- coding: utf-8 -*-
"""PURE unit tests for the Slice 11A.2 Yandex analyzer-provider adapter.

These tests run with NO network and NO credentials:

  * the adapter's SDK client is INJECTED as a fake, so every assertion about
    the request is made against a recorded call, never a real endpoint;
  * the *real* client construction path (credentials, pinned base URL,
    ``max_retries=0``, timeout) is exercised by monkeypatching
    ``openai.OpenAI`` with a recording factory -- no HTTP is performed;
  * the end-to-end tests drive the FROZEN Slice 4 analyzer
    (``support.alice_analyzer.analyze``) with the adapter as the injected
    ``AnalyzerProvider``, proving the port boundary: the adapter forwards
    the complete Slice 4 prompt verbatim, returns the raw provider text,
    and the FROZEN parser -- never the adapter -- owns all validation.

The real, opt-in Alice call lives in
``tests/test_alice_analyzer_yandex_cloud.py``.

The final section contains STATIC, anti-regression guards on the adapter
module: pinned endpoint/model, no production/network imports, no search or
routing vocabulary, a lazily-imported SDK, and no wiring into any
production runtime path.
"""

from __future__ import annotations

import ast
import inspect
import json
import re
from pathlib import Path

import pytest

from support import alice_analyzer_yandex as ay
from support.alice_analyzer import (
    AnalyzerProvider,
    analyze,
    build_allowed_steps,
    build_analyzer_prompt,
)
from support.observation import (
    ObservationScope,
    ProblemStatus,
    ReportedAction,
    RequestKind,
    SemanticObservation,
    StepOutcome,
)
from support.procedure import load_procedure_from_dict

# ============================================================
# FIXTURES / CONSTANTS
# ============================================================

_MESSAGE = "Перезапустил программу, но не помогло"

# The allowlisted step id the fake model may legitimately map to.
_STEP_ID = "RESTART_SOFTWARE"


def _procedure():
    return load_procedure_from_dict(
        {
            "procedure_id": "BLACK_SCREEN",
            "version": 1,
            "issue": "black screen",
            "entry_step": _STEP_ID,
            "terminators": ["ESCALATE", "RESOLVED"],
            "steps": [
                {
                    "id": _STEP_ID,
                    "action": "Restart the software.",
                    "evidence_query": "angioplus black screen restart",
                    "transitions": {
                        "attempted_failed": "CHECK_POWER",
                        "resolved": "RESOLVED",
                    },
                },
                {
                    "id": "CHECK_POWER",
                    "action": "Check the power LED.",
                    "evidence_query": "angioplus black screen power led",
                    "transitions": {
                        "attempted_failed": "ESCALATE",
                        "resolved": "RESOLVED",
                    },
                },
            ],
        }
    )


def _json_observation(**overrides) -> str:
    """A well-formed Slice 4 analyzer JSON payload (the happy shape)."""
    payload = {
        "scope": "in_scope",
        "reported_actions": [
            {"step_id": _STEP_ID, "outcome": "failed"},
        ],
        "problem_status": "ongoing",
        "human_requested": False,
        "clarification_facts": [],
        "new_issue": None,
        "request_kind": "problem",
    }
    payload.update(overrides)
    return json.dumps(payload)


# ============================================================
# FAKES (no SDK, no network)
# ============================================================

class _FakeResponse:
    def __init__(self, output_text):
        self.output_text = output_text


class _FakeResponses:
    """Records every request and returns / raises deterministically."""

    def __init__(self, output_text=None, raises=None):
        self.requests: list[dict] = []
        self._output_text = output_text
        self._raises = raises

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self._raises is not None:
            raise self._raises
        return _FakeResponse(self._output_text)


class _FakeClient:
    def __init__(self, output_text=None, raises=None):
        self.responses = _FakeResponses(output_text=output_text, raises=raises)


class _TransportBoom(Exception):
    """Stand-in for an SDK / auth / network failure."""


def _provider(client) -> ay.YandexAnalyzerProvider:
    return ay.YandexAnalyzerProvider(client=client, folder_id="test-folder")


# ============================================================
# PINNED ENDPOINT / MODEL (pure)
# ============================================================

def test_endpoint_and_model_are_pinned_to_the_canonical_alice_boundary():
    assert ay.DEFAULT_BASE_URL == "https://ai.api.cloud.yandex.net/v1"
    assert ay.MODEL_ID_TEMPLATE == "gpt://{folder_id}/aliceai-llm-flash/latest"
    assert ay.build_analyzer_model_id("my-folder") == (
        "gpt://my-folder/aliceai-llm-flash/latest"
    )
    assert ay.build_analyzer_model_id("  my-folder  ") == (
        "gpt://my-folder/aliceai-llm-flash/latest"
    )
    assert ay.SEMANTIC_FORMAT_NAME == "angioplus_semantic_observation"


@pytest.mark.parametrize("bad_folder", ["", "   ", None, 42])
def test_blank_or_non_string_folder_id_is_rejected(bad_folder):
    with pytest.raises(ValueError):
        ay.build_analyzer_model_id(bad_folder)


def test_constructor_exposes_no_model_override():
    parameters = set(
        inspect.signature(ay.YandexAnalyzerProvider.__init__).parameters
    )
    # There is deliberately NO `model` / `base_url` knob: the model id is
    # derived from the folder id only.
    assert parameters == {
        "self",
        "client",
        "api_key",
        "folder_id",
        "timeout",
        "max_output_tokens",
    }


@pytest.mark.parametrize(
    "kwargs",
    [{"timeout": 0}, {"timeout": -1.0}, {"max_output_tokens": 0}],
)
def test_invalid_transport_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ay.YandexAnalyzerProvider(**kwargs)


def test_provider_conforms_to_the_frozen_analyzer_port():
    # Typed pin: the class is assignable to the frozen ``AnalyzerProvider``
    # port (a bare ``Callable`` alias) and is callable as ``(prompt,
    # history)``.
    provider: AnalyzerProvider = _provider(
        _FakeClient(output_text=_json_observation())
    )
    assert callable(provider)

    parameters = list(
        inspect.signature(ay.YandexAnalyzerProvider.__call__).parameters
    )
    assert parameters == ["self", "prompt", "history"]


# ============================================================
# REQUEST SHAPE / RAW PASS-THROUGH (pure, injected fake)
# ============================================================

def test_request_is_pinned_and_the_prompt_is_sent_verbatim():
    prompt = "ПОЛНЫЙ промпт Slice 4\n<<РАЗРЕШЁННЫЕ ШАГИ\n- step_id: \"X\"\n"

    client = _FakeClient(output_text=_json_observation())
    provider = _provider(client)

    assert provider(prompt) == _json_observation()

    assert len(client.responses.requests) == 1
    request = client.responses.requests[0]

    assert request["model"] == "gpt://test-folder/aliceai-llm-flash/latest"
    assert request["temperature"] == 0
    assert request["input"] == [{"role": "user", "content": prompt}]
    assert request["max_output_tokens"] == ay.DEFAULT_MAX_OUTPUT_TOKENS
    assert request["store"] is False

    # Closed key set: no search/tools, no instructions, no stored state.
    assert set(request) == {
        "model",
        "temperature",
        "input",
        "max_output_tokens",
        "store",
        "text",
    }


def test_history_is_accepted_but_never_re_sent():
    prompt = "ПОЛНЫЙ промпт Slice 4 (история уже внутри)"
    client = _FakeClient(output_text=_json_observation())

    _provider(client)(prompt, [{"role": "user", "text": "привет"}])

    request = client.responses.requests[0]
    # Exactly ONE user message == the prompt; the history is NOT a second
    # message and does not appear as an extra input item.
    assert request["input"] == [{"role": "user", "content": prompt}]


def test_request_pins_the_strict_single_object_envelope():
    client = _FakeClient(output_text=_json_observation())
    _provider(client)("prompt")

    request = client.responses.requests[0]
    fmt = request["text"]["format"]

    assert fmt["type"] == "json_schema"
    assert fmt["name"] == ay.SEMANTIC_FORMAT_NAME
    assert fmt["strict"] is True

    schema = fmt["schema"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "scope",
        "reported_actions",
        "problem_status",
        "human_requested",
        "clarification_facts",
        "new_issue",
        "request_kind",
    }

    # request_kind is REQUIRED and closed to exactly the three enum values.
    request_kind = schema["properties"]["request_kind"]
    assert request_kind["type"] == "string"
    assert request_kind["enum"] == ["problem", "information", "unclear"]
    assert "enum" in request_kind and len(request_kind["enum"]) == 3
    assert "request_kind" in schema["required"]
    assert set(schema["properties"]) == set(schema["required"])


def test_raw_provider_text_is_returned_unchanged():
    raw = '  {"scope": "unclear"}  \n'
    client = _FakeClient(output_text=raw)

    result = _provider(client)("prompt")

    # Identity: no parsing, no stripping, no re-serialization.
    assert result is raw


def test_response_without_output_text_is_a_transport_failure():
    client = _FakeClient(output_text=None)

    with pytest.raises(RuntimeError):
        _provider(client)("prompt")

    assert len(client.responses.requests) == 1  # the call did happen


@pytest.mark.parametrize("blank", ["", "   ", "\n\t", None, 42])
def test_blank_prompt_is_refused_without_a_call(blank):
    client = _FakeClient(output_text=_json_observation())

    with pytest.raises(ValueError):
        _provider(client)(blank)

    assert client.responses.requests == []


@pytest.mark.parametrize(
    "failure",
    [
        _TransportBoom("429 rate limited"),
        TimeoutError("read timeout"),
        PermissionError("invalid api key"),
        RuntimeError("connection reset"),
    ],
)
def test_transport_failures_propagate_unchanged(failure):
    client = _FakeClient(raises=failure)

    # No swallowing, no retry: the frozen analyzer owns the fail-safe mapping.
    with pytest.raises(type(failure)):
        _provider(client)("prompt")

    assert len(client.responses.requests) == 1


# ============================================================
# LAZY WIRING / CREDENTIALS (pure, no HTTP)
# ============================================================

def test_real_client_is_built_lazily_once_with_pinned_transport_config(
    monkeypatch,
):
    monkeypatch.setenv("YANDEX_API_KEY", "test-key")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "env-folder")

    client = _FakeClient(output_text=_json_observation())
    built: list[dict] = []

    def _factory(**kwargs):
        built.append(kwargs)
        return client

    monkeypatch.setattr("openai.OpenAI", _factory)

    provider = ay.YandexAnalyzerProvider()
    assert built == []  # nothing is built at construction time

    provider("prompt")
    provider("prompt")

    assert built == [
        {
            "api_key": "test-key",
            "base_url": "https://ai.api.cloud.yandex.net/v1",
            "project": "env-folder",
            "timeout": ay.DEFAULT_TIMEOUT_SECONDS,
            "max_retries": 0,
        }
    ]
    assert len(client.responses.requests) == 2
    assert client.responses.requests[1]["model"] == (
        "gpt://env-folder/aliceai-llm-flash/latest"
    )


def test_missing_folder_id_fails_closed_before_any_call(monkeypatch):
    monkeypatch.delenv("YANDEX_FOLDER_ID", raising=False)
    monkeypatch.delenv("YANDEX_API_KEY", raising=False)

    client = _FakeClient(output_text=_json_observation())
    provider = ay.YandexAnalyzerProvider(client=client)

    with pytest.raises(RuntimeError):
        provider("prompt")

    assert client.responses.requests == []


def test_missing_api_key_fails_closed(monkeypatch):
    monkeypatch.setenv("YANDEX_FOLDER_ID", "env-folder")
    monkeypatch.delenv("YANDEX_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        ay.YandexAnalyzerProvider()("prompt")


def test_injected_credentials_are_used_when_the_client_is_built(monkeypatch):
    monkeypatch.setenv("YANDEX_FOLDER_ID", "env-folder")
    monkeypatch.delenv("YANDEX_API_KEY", raising=False)

    client = _FakeClient(output_text=_json_observation())
    built: list[dict] = []
    monkeypatch.setattr(
        "openai.OpenAI",
        lambda **kwargs: (built.append(kwargs), client)[1],
    )

    provider = ay.YandexAnalyzerProvider(api_key="injected-key")
    provider("prompt")

    assert built[0]["api_key"] == "injected-key"


# ============================================================
# END-TO-END THROUGH THE FROZEN SLICE 4 ANALYZER
# ============================================================

def test_adapter_feeds_the_frozen_analyzer():
    client = _FakeClient(output_text=_json_observation())
    provider = _provider(client)

    observation = analyze(
        message=_MESSAGE,
        history=[],
        procedure=_procedure(),
        last_recommendation=_STEP_ID,
        provider=provider,
    )

    assert observation.scope is ObservationScope.IN_SCOPE
    assert observation.problem_status is ProblemStatus.ONGOING
    assert observation.reported_actions == (
        ReportedAction(step_id=_STEP_ID, outcome=StepOutcome.FAILED),
    )
    assert observation.human_requested is False

    # Exactly one provider call, carrying the COMPLETE frozen Slice 4 prompt.
    assert len(client.responses.requests) == 1


def test_adapter_forwards_the_complete_frozen_prompt_verbatim():
    history = [{"role": "user", "text": "экран чёрный"}]
    client = _FakeClient(output_text=_json_observation())

    analyze(
        message=_MESSAGE,
        history=history,
        procedure=_procedure(),
        last_recommendation=_STEP_ID,
        provider=_provider(client),
    )

    sent = client.responses.requests[0]["input"][0]["content"]
    expected = build_analyzer_prompt(
        message=_MESSAGE,
        history=history,
        allowed_steps=build_allowed_steps(_procedure()),
        last_recommendation=_STEP_ID,
    )

    assert sent == expected
    assert _MESSAGE in sent  # the raw message is present, unmodified


def test_the_frozen_parser_owns_validation_not_the_adapter():
    # An INVENTED step id: the ADAPTER must pass it straight through, and
    # the FROZEN parser (not the adapter) downgrades it to ``None``.
    raw = _json_observation(
        reported_actions=[{"step_id": "INVENTED_STEP", "outcome": "failed"}]
    )
    client = _FakeClient(output_text=raw)

    observation = analyze(
        message=_MESSAGE,
        procedure=_procedure(),
        provider=_provider(client),
    )

    assert observation.reported_actions == (
        ReportedAction(step_id=None, outcome=StepOutcome.FAILED),
    )
    # The adapter returned the raw text untouched (one call, no re-formatting).
    assert len(client.responses.requests) == 1


def test_transport_error_is_mapped_to_the_fail_safe_observation():
    client = _FakeClient(raises=_TransportBoom("429"))

    observation = analyze(
        message=_MESSAGE,
        procedure=_procedure(),
        provider=_provider(client),
    )

    assert observation == SemanticObservation(
        scope=ObservationScope.UNCLEAR,
        reported_actions=(ReportedAction(step_id=None, outcome=None),),
        problem_status=None,
        human_requested=False,
        clarification_facts=(),
        new_issue=None,
        request_kind=RequestKind.UNCLEAR,
    )
    assert len(client.responses.requests) == 1


@pytest.mark.parametrize(
    "malformed",
    ["", "   ", "not json at all", "{broken", "[]", '{"scope": "in_scope"}'],
)
def test_malformed_provider_output_is_mapped_to_the_fail_safe_observation(
    malformed,
):
    client = _FakeClient(output_text=malformed)

    observation = analyze(
        message=_MESSAGE,
        procedure=_procedure(),
        provider=_provider(client),
    )

    assert observation.scope is ObservationScope.UNCLEAR
    assert observation.reported_actions == (
        ReportedAction(step_id=None, outcome=None),
    )
    assert len(client.responses.requests) == 1


# ============================================================
# STATIC ANTI-REGRESSION GUARDS ON THE ADAPTER MODULE
# ============================================================

def _adapter_source() -> str:
    return Path(ay.__file__).read_text(encoding="utf-8")


def _adapter_tree() -> ast.Module:
    return ast.parse(_adapter_source())


def _module_level_imports() -> set[str]:
    """Dotted module names imported at MODULE level (not inside functions)."""
    imported: set[str] = set()
    for node in _adapter_tree().body:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module)
    return imported


def _code_only_text() -> str:
    """Identifiers, attribute names, keywords, class/function names and
    string LITERALS -- everything EXCEPT comments and docstrings (the module
    docstring legitimately NAMES the anti-patterns it avoids)."""
    tree = _adapter_tree()

    docstring_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstring_ids.add(id(body[0].value))

    parts: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            parts.append(node.attr)
        elif isinstance(node, ast.Name):
            parts.append(node.id)
        elif isinstance(node, ast.keyword) and node.arg:
            parts.append(node.arg)
        elif isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            parts.append(node.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstring_ids:
                parts.append(node.value)

    return " ".join(parts).lower()


def test_no_production_or_network_imports():
    imports = _module_level_imports()
    roots = {name.split(".")[0] for name in imports}

    assert roots <= {"__future__", "os", "typing"}
    # The adapter is a PURE transport: it imports nothing from ``support``
    # (the frozen port is a bare ``Callable`` alias, satisfied structurally).
    assert not any(name.startswith("support") for name in imports)


def test_openai_sdk_is_imported_lazily_inside_the_call_path():
    source = _adapter_source()
    imports = _module_level_imports()

    assert "from openai import OpenAI" in source
    assert "openai" not in {name.split(".")[0] for name in imports}
    # ... and it happens INSIDE the class, i.e. only when a real call is made.
    assert source.index("from openai import OpenAI") > source.index(
        "class YandexAnalyzerProvider"
    )


def test_only_the_two_frozen_environment_variables_are_read():
    env_names = set(re.findall(r"yandex_[a-z0-9_]+", _code_only_text()))
    assert env_names == {"yandex_api_key", "yandex_folder_id"}
    # No model / endpoint switch of any kind.
    assert "YANDEX_ANALYZER_MODEL" not in _adapter_source()


def test_no_search_routing_or_legacy_vocabulary_in_the_adapter_code():
    code = _code_only_text()

    forbidden = (
        # retrieval / search / tools / history / state
        "file_search",
        "vector_store",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "previous_response_id",
        "conversation",
        "stream",
        # routing / procedure / legacy policy
        "routeaction",
        "routingdecision",
        "apply_support_policy",
        "candidate_l1_action",
        "candidate_action_already_attempted",
        "transition_type",
        "next_step",
        "escalate",
    )
    for token in forbidden:
        assert token not in code, token


def test_adapter_source_never_parses_or_screens_the_provider_text():
    code = _code_only_text()

    # All validation belongs to the frozen parser, never to the transport.
    for token in ("json.loads", "loads", "splitlines", "findall", "search"):
        assert token not in code, token


def test_public_api_surface():
    assert set(ay.__all__) == {
        "YandexAnalyzerProvider",
        "build_analyzer_model_id",
        "DEFAULT_BASE_URL",
        "MODEL_ID_TEMPLATE",
        "DEFAULT_TIMEOUT_SECONDS",
        "DEFAULT_MAX_OUTPUT_TOKENS",
        "SEMANTIC_FORMAT_NAME",
    }


# Slice 11A.3a: ``support/support_core.py`` is the ONE sanctioned production
# COMPOSITION root -- the single module allowed to reference this adapter and
# construct it. Every OTHER support / runtime module must still not reference
# it (it is still not wired into any runtime path).
_COMPOSITION_ROOT = Path("support/support_core.py")


def test_not_wired_into_any_production_runtime_path():
    adapter_path = Path(ay.__file__).resolve()
    root = adapter_path.parents[1]
    composition_root = (root / _COMPOSITION_ROOT).resolve()

    scanned = 0
    for pattern in ("support/**/*.py", "yandex/**/*.py", "*.py"):
        for path in root.glob(pattern):
            if path.resolve() in (adapter_path, composition_root):
                continue
            if "__pycache__" in path.parts:
                continue
            scanned += 1
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert "alice_analyzer_yandex" not in text, path

    assert scanned > 0
