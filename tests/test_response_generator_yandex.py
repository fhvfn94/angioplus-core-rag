"""
PURE unit tests for the Slice 6C Yandex explanation-provider adapter.

These tests run with NO network and NO credentials:

  * the adapter's SDK client is INJECTED as a fake, so every assertion about
    the request is made against a recorded call, never a real endpoint;
  * the *real* client construction path (credentials, pinned base URL,
    ``max_retries=0``, timeout) is exercised by monkeypatching
    ``openai.OpenAI`` with a recording factory -- no HTTP is performed;
  * the end-to-end tests drive the FROZEN 6B renderer
    (``support.response_generator.render_response``) with the adapter as the
    injected ``ResponseProvider``, proving the port boundary: the adapter
    forwards the complete 6B prompt verbatim, returns the raw provider text,
    and can never replace the procedure-owned instruction.

The real, opt-in Alice call lives in
``tests/test_response_generator_yandex_cloud.py``.

The final section contains STATIC, anti-regression guards on the adapter
module: pinned endpoint/model, no production/network imports, no search or
routing vocabulary, a lazily-imported SDK, and no wiring into any production
runtime path.
"""

from __future__ import annotations

import ast
import inspect
import json
import re
from pathlib import Path

import pytest

from support import response_generator as rg
from support import response_generator_yandex as ry
from support.knowledge import KnowledgeItem, KnowledgeResult, KnowledgeStatus
from support.procedure import load_procedure_from_dict
from support.response_generator import (
    ResponseStatus,
    build_renderer_prompt,
    render_response,
)
from support.router import (
    ClarifyReason,
    EscalateReason,
    RouteAction,
    RoutingDecision,
)

# ============================================================
# FIXTURES / CONSTANTS
# ============================================================

_ACTION_A = "Confirm the power LED is lit."
_ACTION_B = "Reseat the display cable on both ends."
_EVIDENCE_QUERY_A = "angioplus black screen power led"
_EVIDENCE_QUERY_B = "angioplus black screen display cable"

# Associated with STEP_A and minimized by 6B to exactly this fragment.
_EVIDENCE_A = "The power LED is lit."
_EVIDENCE_FRAGMENT_A = "The power LED is lit"

# Shares no meaningful token with STEP_A's authorized vocabulary.
_UNRELATED_EVIDENCE = (
    "The warranty period is twelve months from the invoice date."
)

# Accepted by 6B: supporting, non-imperative, no continuation, no control token.
_SAFE_EXPLANATION = (
    "Индикатор питания подтверждает, что модуль получает питание."
)

_UNSAFE_EXPLANATIONS = (
    "Перезапустите программу.",  # introduces a NEW imperative action
    "Contact the manufacturer for a replacement.",  # external-contact directive
    "Затем перейдите в меню настроек.",  # continuation + new instruction
    "Вместо этого нажмите кнопку сброса.",  # replaces the instruction
    "escalate",  # routing / control vocabulary
)

_MALFORMED_PROVIDER_OUTPUTS = (
    "",
    "   ",
    "not json at all",
    "{broken",
    "[]",
    '"just a string"',
    '{"other": "x"}',
    '{"explanation": ""}',
    '{"explanation": "ok", "extra": 1}',
    '{"explanation": 42}',
)


def _procedure():
    return load_procedure_from_dict(
        {
            "procedure_id": "BLACK_SCREEN",
            "version": 1,
            "issue": "black screen",
            "entry_step": "STEP_A",
            "terminators": ["ESCALATE", "RESOLVED"],
            "steps": [
                {
                    "id": "STEP_A",
                    "action": _ACTION_A,
                    "evidence_query": _EVIDENCE_QUERY_A,
                    "transitions": {
                        "attempted_failed": "STEP_B",
                        "resolved": "RESOLVED",
                    },
                },
                {
                    "id": "STEP_B",
                    "action": _ACTION_B,
                    "evidence_query": _EVIDENCE_QUERY_B,
                    "transitions": {
                        "attempted_failed": "ESCALATE",
                        "resolved": "RESOLVED",
                    },
                },
            ],
        }
    )


def _recommend(step_id: str = "STEP_A") -> RoutingDecision:
    return RoutingDecision(action=RouteAction.RECOMMEND_STEP, next_step=step_id)


def _found(*texts: str) -> KnowledgeResult:
    return KnowledgeResult(
        status=KnowledgeStatus.FOUND,
        query=_EVIDENCE_QUERY_A,
        items=tuple(KnowledgeItem(text=text) for text in texts),
    )


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


def _provider(client) -> ry.YandexExplanationProvider:
    return ry.YandexExplanationProvider(
        client=client, folder_id="test-folder"
    )


def _json_explanation(text: str) -> str:
    return json.dumps({"explanation": text})


# ============================================================
# PINNED ENDPOINT / MODEL (pure)
# ============================================================

def test_endpoint_and_model_are_pinned_to_the_canonical_alice_boundary():
    assert ry.DEFAULT_BASE_URL == "https://ai.api.cloud.yandex.net/v1"
    assert ry.MODEL_ID_TEMPLATE == "gpt://{folder_id}/aliceai-llm-flash/latest"
    assert ry.build_explanation_model_id("my-folder") == (
        "gpt://my-folder/aliceai-llm-flash/latest"
    )
    assert ry.build_explanation_model_id("  my-folder  ") == (
        "gpt://my-folder/aliceai-llm-flash/latest"
    )
    assert ry.EXPLANATION_FIELD == "explanation"
    assert ry.EXPLANATION_FORMAT_NAME == "angioplus_response_explanation"


@pytest.mark.parametrize("bad_folder", ["", "   ", None, 42])
def test_blank_or_non_string_folder_id_is_rejected(bad_folder):
    with pytest.raises(ValueError):
        ry.build_explanation_model_id(bad_folder)


def test_constructor_exposes_no_model_override():
    parameters = set(
        inspect.signature(ry.YandexExplanationProvider.__init__).parameters
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
        ry.YandexExplanationProvider(**kwargs)


def test_provider_is_the_frozen_6b_port():
    # The adapter explicitly implements the frozen 6B Protocol.
    assert rg.ResponseProvider in ry.YandexExplanationProvider.__mro__
    assert callable(_provider(_FakeClient(output_text=_json_explanation("x"))))


# ============================================================
# REQUEST SHAPE / RAW PASS-THROUGH (pure, injected fake)
# ============================================================

def test_request_is_pinned_and_the_prompt_is_sent_verbatim():
    prompt = "ПОЛНЫЙ промпт 6B\n<<EVIDENCE\nLED is lit\nEVIDENCE\n"

    client = _FakeClient(output_text=_json_explanation(""))
    provider = _provider(client)

    assert provider(prompt) == _json_explanation("")

    assert len(client.responses.requests) == 1
    request = client.responses.requests[0]

    assert request == {
        "model": "gpt://test-folder/aliceai-llm-flash/latest",
        "temperature": 0,
        "input": [{"role": "user", "content": prompt}],
        "max_output_tokens": ry.DEFAULT_MAX_OUTPUT_TOKENS,
        "store": False,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "angioplus_response_explanation",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"explanation": {"type": "string"}},
                    "required": ["explanation"],
                    "additionalProperties": False,
                },
            }
        },
    }

    # Closed key set: no search/tools, no history, no stored state.
    assert set(request) == {
        "model",
        "temperature",
        "input",
        "max_output_tokens",
        "store",
        "text",
    }


def test_raw_provider_text_is_returned_unchanged():
    raw = '  {"explanation": "  not touched  "}\n'
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
    client = _FakeClient(output_text=_json_explanation("x"))

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

    # No swallowing, no retry: 6B owns the fail-safe mapping.
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

    client = _FakeClient(output_text=_json_explanation("ok"))
    built: list[dict] = []

    def _factory(**kwargs):
        built.append(kwargs)
        return client

    monkeypatch.setattr("openai.OpenAI", _factory)

    provider = ry.YandexExplanationProvider()
    assert built == []  # nothing is built at construction time

    provider("prompt")
    provider("prompt")

    assert built == [
        {
            "api_key": "test-key",
            "base_url": "https://ai.api.cloud.yandex.net/v1",
            "project": "env-folder",
            "timeout": ry.DEFAULT_TIMEOUT_SECONDS,
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

    client = _FakeClient(output_text=_json_explanation("ok"))
    provider = ry.YandexExplanationProvider(client=client)

    with pytest.raises(RuntimeError):
        provider("prompt")

    assert client.responses.requests == []


def test_missing_api_key_fails_closed(monkeypatch):
    monkeypatch.setenv("YANDEX_FOLDER_ID", "env-folder")
    monkeypatch.delenv("YANDEX_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        ry.YandexExplanationProvider()("prompt")


def test_injected_credentials_are_used_when_the_client_is_built(
    monkeypatch,
):
    monkeypatch.setenv("YANDEX_FOLDER_ID", "env-folder")
    monkeypatch.delenv("YANDEX_API_KEY", raising=False)

    client = _FakeClient(output_text=_json_explanation("ok"))
    built: list[dict] = []
    monkeypatch.setattr(
        "openai.OpenAI",
        lambda **kwargs: (built.append(kwargs), client)[1],
    )

    provider = ry.YandexExplanationProvider(api_key="injected-key")
    provider("prompt")

    assert built[0]["api_key"] == "injected-key"


# ============================================================
# END-TO-END THROUGH THE FROZEN 6B RENDERER
# ============================================================

def test_6b_grounded_render_appends_a_validated_explanation():
    client = _FakeClient(output_text=_json_explanation(_SAFE_EXPLANATION))
    provider = _provider(client)

    result = render_response(
        _recommend(),
        procedure=_procedure(),
        knowledge=_found(_EVIDENCE_A),
        provider=provider,
    )

    assert result.status is ResponseStatus.RENDERED
    assert result.provider_called is True
    assert result.step_id == "STEP_A"
    assert result.text == f"{_ACTION_A} {_SAFE_EXPLANATION}"
    assert result.text.startswith(_ACTION_A)

    # Exactly one provider call, carrying the COMPLETE frozen 6B prompt.
    assert len(client.responses.requests) == 1
    assert client.responses.requests[0]["input"] == [
        {
            "role": "user",
            "content": build_renderer_prompt(
                step_action=_ACTION_A,
                evidence_fragment=_EVIDENCE_FRAGMENT_A,
            ),
        }
    ]


@pytest.mark.parametrize("unsafe", _UNSAFE_EXPLANATIONS)
def test_6b_rejects_unsafe_explanations_and_keeps_the_action_verbatim(unsafe):
    client = _FakeClient(output_text=_json_explanation(unsafe))
    provider = _provider(client)

    result = render_response(
        _recommend(),
        procedure=_procedure(),
        knowledge=_found(_EVIDENCE_A),
        provider=provider,
    )

    assert result.status is ResponseStatus.FALLBACK
    assert result.reason == "explanation_rejected"
    assert result.provider_called is True
    assert result.text == _ACTION_A  # unchanged, never weakened
    assert len(client.responses.requests) == 1


@pytest.mark.parametrize("malformed", _MALFORMED_PROVIDER_OUTPUTS)
def test_6b_malformed_provider_output_is_fail_safe(malformed):
    client = _FakeClient(output_text=malformed)
    provider = _provider(client)

    result = render_response(
        _recommend(),
        procedure=_procedure(),
        knowledge=_found(_EVIDENCE_A),
        provider=provider,
    )

    assert result.status is ResponseStatus.FALLBACK
    assert result.reason == "explanation_rejected"
    assert result.provider_called is True
    assert result.text == _ACTION_A
    assert len(client.responses.requests) == 1


def test_6b_transport_error_is_fail_safe():
    client = _FakeClient(raises=_TransportBoom("429"))
    provider = _provider(client)

    result = render_response(
        _recommend(),
        procedure=_procedure(),
        knowledge=_found(_EVIDENCE_A),
        provider=provider,
    )

    assert result.status is ResponseStatus.FALLBACK
    assert result.reason == "provider_error"
    assert result.provider_called is True
    assert result.text == _ACTION_A
    assert len(client.responses.requests) == 1


def test_6b_provider_is_not_called_without_associated_evidence():
    client = _FakeClient(output_text=_json_explanation(_SAFE_EXPLANATION))
    provider = _provider(client)

    result = render_response(
        _recommend(),
        procedure=_procedure(),
        knowledge=_found(_UNRELATED_EVIDENCE),
        provider=provider,
    )

    assert result.status is ResponseStatus.RENDERED
    assert result.reason == "no_associated_evidence"
    assert result.provider_called is False
    assert result.text == _ACTION_A
    assert client.responses.requests == []  # zero paid calls


@pytest.mark.parametrize(
    "decision",
    [
        RoutingDecision(
            action=RouteAction.CLARIFY,
            clarify_reason=ClarifyReason.PENDING_CLARIFICATION,
        ),
        RoutingDecision(
            action=RouteAction.ESCALATE,
            escalate_reason=EscalateReason.PROCEDURE_TERMINAL,
        ),
        RoutingDecision(action=RouteAction.RESOLVE),
        RoutingDecision(action=RouteAction.OUT_OF_SCOPE),
    ],
    ids=["clarify", "escalate", "resolve", "out_of_scope"],
)
def test_6b_provider_is_never_called_for_non_recommend_actions(decision):
    client = _FakeClient(output_text=_json_explanation(_SAFE_EXPLANATION))
    provider = _provider(client)

    result = render_response(
        decision,
        procedure=_procedure(),
        knowledge=_found(_EVIDENCE_A),
        provider=provider,
    )

    assert result.provider_called is False
    assert client.responses.requests == []
    assert _ACTION_A not in result.text


def test_6b_action_is_always_rendered_verbatim_across_provider_outcomes():
    cases = [
        _FakeClient(output_text=_json_explanation(_SAFE_EXPLANATION)),
        _FakeClient(output_text=_json_explanation(_UNSAFE_EXPLANATIONS[0])),
        _FakeClient(output_text="not json"),
        _FakeClient(raises=_TransportBoom("boom")),
        _FakeClient(output_text=_json_explanation("")),
        _FakeClient(output_text=None),
    ]

    for client in cases:
        result = render_response(
            _recommend(),
            procedure=_procedure(),
            knowledge=_found(_EVIDENCE_A),
            provider=_provider(client),
        )
        assert result.text.startswith(_ACTION_A), result


# ============================================================
# STATIC ANTI-REGRESSION GUARDS ON THE ADAPTER MODULE
# ============================================================

def _adapter_source() -> str:
    return Path(ry.__file__).read_text(encoding="utf-8")


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

    assert roots <= {"__future__", "os", "typing", "support"}
    # The ONLY support import is the frozen 6B port itself.
    assert {n for n in imports if n.startswith("support")} == {
        "support.response_generator"
    }


def test_openai_sdk_is_imported_lazily_inside_the_call_path():
    source = _adapter_source()
    imports = _module_level_imports()

    assert "from openai import OpenAI" in source
    assert "openai" not in {name.split(".")[0] for name in imports}
    # ... and it happens INSIDE the class, i.e. only when a real call is made.
    assert source.index("from openai import OpenAI") > source.index(
        "class YandexExplanationProvider"
    )


def test_only_the_two_frozen_environment_variables_are_read():
    env_names = set(re.findall(r"yandex_[a-z0-9_]+", _code_only_text()))
    assert env_names == {"yandex_api_key", "yandex_folder_id"}
    # No model / endpoint switch of any kind.
    assert "YANDEX_EXPLANATION_MODEL" not in _adapter_source()


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

    for token in ("json.loads", "loads", "splitlines", "findall", "search"):
        assert token not in code, token


def test_public_api_surface():
    assert set(ry.__all__) == {
        "YandexExplanationProvider",
        "build_explanation_model_id",
        "DEFAULT_BASE_URL",
        "MODEL_ID_TEMPLATE",
        "DEFAULT_TIMEOUT_SECONDS",
        "DEFAULT_MAX_OUTPUT_TOKENS",
        "EXPLANATION_FIELD",
        "EXPLANATION_FORMAT_NAME",
    }


def test_not_wired_into_any_production_runtime_path():
    adapter_path = Path(ry.__file__).resolve()
    root = adapter_path.parents[1]

    scanned = 0
    for pattern in ("support/**/*.py", "yandex/**/*.py", "*.py"):
        for path in root.glob(pattern):
            if path.resolve() == adapter_path or "__pycache__" in path.parts:
                continue
            scanned += 1
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert "response_generator_yandex" not in text, path

    assert scanned > 0
