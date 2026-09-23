# -*- coding: utf-8 -*-
"""Opt-in REAL-Alice tests for support.response_generator_yandex (Slice 6C).

These tests hit the REAL Yandex Alice Responses API over the network and are
gated behind ``@pytest.mark.cloud``; they are SKIPPED unless you explicitly
pass ``--run-cloud``:

    python -m pytest tests/test_response_generator_yandex_cloud.py --run-cloud -q

They exercise the adapter through its DEFAULT wiring -- NO injected client --
so the pinned endpoint, the pinned model id, the credentials, the strict
single-field JSON envelope and the raw-output pass-through are all proven at
the real boundary. The adapter is then driven through the FROZEN 6B renderer
to prove the end-to-end grounded path (instruction + validated explanation).

BUDGET: at most TWO real model calls in total -- one per network test. The
calls are counted in ``_CALLS`` and asserted after every call, so adding a
third network call fails loudly instead of silently spending money. Both
network tests are SKIPPED when credentials are absent, so the module stays
safe to run in CI without secrets.

It deliberately imports NO legacy support module (neither
``yandex/chat/alice_support_agent.py`` nor ``yandex/chat/server.py``): only
the frozen 6B port, the adapter under test and in-memory 6B value objects.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import re
from pathlib import Path

import pytest

from support import response_generator as rg
from support import response_generator_yandex as ry
from support.knowledge import KnowledgeItem, KnowledgeResult, KnowledgeStatus
from support.procedure import load_procedure_from_dict
from support.response_generator import ResponseStatus, render_response
from support.router import RouteAction, RoutingDecision


# NOTE: ``cloud`` is applied per network test, NOT at module level. The
# static anti-regression guards below are pure, contain no real call and
# must ALWAYS run -- they must not be skippable by omitting ``--run-cloud``.
cloud = pytest.mark.cloud

# Credentials (guarded dotenv load; fail-closed but not fail-hard). Loaded
# HERE so the credential check below does not depend on import order.
try:  # dotenv is a repo dependency; load the project .env if present.
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:  # pragma: no cover - import/env is best-effort
    pass

_HAVE_CREDS = bool(
    os.getenv("YANDEX_API_KEY", "").strip()
    and os.getenv("YANDEX_FOLDER_ID", "").strip()
)

needs_creds = pytest.mark.skipif(
    not _HAVE_CREDS,
    reason="YANDEX_API_KEY / YANDEX_FOLDER_ID not set; skipping cloud test.",
)

# Real model calls spent by this module. Hard budget: 2.
_CALLS = 0
_MAX_CALLS = 2

# Canonical Alice boundary / model tier, composed HERE (independently of the
# adapter) so the cloud test asserts against its OWN expectation.
_CLOUD_BASE_URL = "https://ai.api.cloud.yandex.net/v1"
_CLOUD_MODEL_TEMPLATE = "gpt://{folder_id}/aliceai-llm-flash/latest"


def _cloud_model_id(folder_id: str) -> str:
    return _CLOUD_MODEL_TEMPLATE.format(folder_id=folder_id)


# ----------------------------------------------------------------
# In-memory procedure (NOT a frozen fixture)
# ----------------------------------------------------------------

_ACTION_A = "Confirm the power LED is lit."
_ACTION_B = "Reseat the display cable on both ends."
_EVIDENCE_QUERY_A = "angioplus black screen power led"
_EVIDENCE_QUERY_B = "angioplus black screen display cable"

# Needs >=2 meaningful tokens shared with STEP_A's authorized vocabulary so
# 6B's evidence association accepts it (and minimizes it to the fragment).
_EVIDENCE_A = "The power LED is lit."
_EVIDENCE_FRAGMENT_A = "The power LED is lit"


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


# ================================================================
# 1. REAL grounded render through the frozen 6B renderer
# ================================================================

@cloud
@needs_creds
def test_cloud_grounded_explanation_through_6b():
    global _CALLS

    # DEFAULT wiring: the adapter builds the real client itself.
    provider = ry.YandexExplanationProvider()

    _CALLS += 1
    result = render_response(
        _recommend(),
        procedure=_procedure(),
        knowledge=_found(_EVIDENCE_A),
        provider=provider,
    )

    print("\n================ CLOUD GROUNDED RENDER ================")
    print("status         :", result.status)
    print("reason         :", result.reason)
    print("provider_called:", result.provider_called)
    print("text           :", result.text)
    print("======================================================\n")

    assert _CALLS == 1

    # The adapter really was exercised at the real boundary, and it never
    # turns a working call into an internal error.
    assert result.provider_called is True
    assert result.status is not ResponseStatus.INTERNAL_ERROR
    assert result.step_id == "STEP_A"

    # INVARIANT: the procedure-owned instruction is NEVER altered -- no
    # matter what the model returns.
    assert result.text.startswith(_ACTION_A)

    if result.status is ResponseStatus.RENDERED:
        # A non-empty explanation was appended AFTER the verbatim instruction.
        assert result.text.startswith(f"{_ACTION_A} ")
        assert len(result.text) > len(_ACTION_A) + 1
    else:
        # Only the documented deterministic fail-safe is acceptable: the
        # instruction alone. A transport error is NOT tolerated here -- with
        # credentials and --run-cloud present, the real call must succeed.
        assert result.status is ResponseStatus.FALLBACK
        assert result.reason == "explanation_rejected", (
            f"real Alice call failed / produced an unusable result: "
            f"{result.reason!r}"
        )
        assert result.text == _ACTION_A
        print(
            "NOTE: the model's explanation was rejected by the 6B safety "
            "checks (deterministic fail-safe path)."
        )


# ================================================================
# 2. REAL raw output: the strict single-field envelope
# ================================================================

@cloud
@needs_creds
def test_cloud_raw_output_is_the_strict_single_field_envelope():
    global _CALLS

    prompt = rg.build_renderer_prompt(
        step_action=_ACTION_A,
        evidence_fragment=_EVIDENCE_FRAGMENT_A,
    )
    provider = ry.YandexExplanationProvider()

    _CALLS += 1
    raw = provider(prompt)
    assert _CALLS == 2  # hard budget reached; nothing else may call out

    print("\n================ CLOUD RAW PROVIDER OUTPUT ============")
    print("model:", _cloud_model_id(os.environ["YANDEX_FOLDER_ID"]))
    print("raw  :", raw)
    print("======================================================\n")

    # The strict ``json_schema`` envelope forces parseable, single-field
    # JSON. A network / SDK failure would have raised INSIDE ``provider``
    # instead of reaching this assertion.
    assert isinstance(raw, str)
    payload = json.loads(raw)
    assert isinstance(payload, dict)
    assert set(payload) == {"explanation"}
    assert isinstance(payload["explanation"], str)

    # And the frozen 6B parser agrees -- no salvage, no partial repair.
    parsed = rg.parse_renderer_output(raw)
    assert parsed is None or isinstance(parsed, str)
    if parsed is not None:
        assert parsed.strip() != ""


# ================================================================
# STATIC GUARDS (pure, always run)
# ================================================================

def test_cloud_module_imports_no_legacy_pipeline():
    """This module must never import the legacy Yandex pipeline.

    Enforced by AST so the guard runs even WITHOUT --run-cloud.
    """
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

        roots = {name.split(".")[0] for name in imported}
    assert roots <= {
        "__future__",
        "ast",
        "dotenv",
        "inspect",
        "json",
        "os",
        "re",
        "pathlib",
        "pytest",
        "support",
    }, sorted(roots)
    assert not any(
        name == "yandex" or name.startswith("yandex.") for name in imported
    ), sorted(imported)


def test_cloud_provider_model_matches_the_cloud_template():
    """The adapter's pinned model id equals this test's own composition."""
    assert ry.MODEL_ID_TEMPLATE == _CLOUD_MODEL_TEMPLATE
    assert ry.DEFAULT_BASE_URL == _CLOUD_BASE_URL
    assert ry.build_explanation_model_id("my-folder") == (
        _cloud_model_id("my-folder")
    )


def test_cloud_adapter_uses_the_canonical_nested_text_format_envelope():
    """The Responses API ``text`` config must be ``{"format": {...}}``.

    This is the shape every other canonical Alice call site in this repo
    uses; a bare ``text={"type": "json_schema", ...}`` is NOT accepted by
    the API, so this pin protects the real boundary from regressing.
    """
    code = re.sub(r"\s+", "", inspect.getsource(ry))
    assert 'text={"format":_explanation_format()}' in code
    assert "text=_explanation_format()" not in code


def test_cloud_call_budget_is_bounded():
    """Hard budget: this module never spends more than TWO real calls."""
    assert _CALLS <= _MAX_CALLS