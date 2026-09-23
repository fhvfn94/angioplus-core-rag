# -*- coding: utf-8 -*-
"""Opt-in REAL-Alice SEMANTIC tests for support.alice_analyzer.

These tests call the real Yandex Alice model through a provider built here,
so they require the network and are gated behind ``@pytest.mark.cloud``.
They are SKIPPED unless you explicitly pass ``--run-cloud``:

    python -m pytest tests/test_alice_analyzer_cloud.py --run-cloud -q

The provider talks DIRECTLY to the canonical Alice boundary over the
Responses API -- it builds its own client and constructs the model id
inline. Validation now runs on the cheaper, currently supported
Alice AI LLM **Flash**:
    model = gpt://<YANDEX_FOLDER_ID>/aliceai-llm-flash/latest
    client.responses.create(..., text={"format": {json_schema}})

It deliberately imports NO legacy support module: neither
``yandex/chat/alice_support_agent.py`` nor ``yandex/chat/server.py``. The
model identifier is composed here from ``YANDEX_FOLDER_ID`` and the public
``base_url`` so the cloud test can never drag the legacy pipeline (File
Search, ``candidate_l1_action``, the string-priority cascade) into the
analyzer contract.

They perform NO File Search: only the analyzer boundary is exercised.

These tests assert SEMANTICS ONLY -- never ANSWER/ESCALATE/CLARIFY decisions
and never any legacy-string cascade vocabulary. The procedure used is an
in-memory BLACK_SCREEN-like graph containing RESTART_SOFTWARE / CHECK_CABLE,
built via load_procedure_from_dict so NO frozen fixture is touched.

If credentials (YANDEX_API_KEY / YANDEX_FOLDER_ID) are missing, every test
in this module is skipped rather than failed.
"""

from __future__ import annotations

import json
import os

import pytest

from support.alice_analyzer import analyze, build_allowed_steps
from support.observation import (
    ObservationScope,
    ProblemStatus,
    StepOutcome,
)
from support.procedure import load_procedure_from_dict


# NOTE: ``cloud`` is applied per network test, NOT at module level. The
# two static anti-regression guards (no legacy import; canonical model id)
# are pure and must ALWAYS run -- they must not be skippable by omitting
# ``--run-cloud``.


# ----------------------------------------------------------------
# Skip-if-no-credentials (fail-closed but not fail-hard)
# ----------------------------------------------------------------

_HAVE_CREDS = bool(
    os.getenv("YANDEX_API_KEY", "").strip()
    and os.getenv("YANDEX_FOLDER_ID", "").strip()
)

needs_creds = pytest.mark.skipif(
    not _HAVE_CREDS,
    reason="YANDEX_API_KEY / YANDEX_FOLDER_ID not set; skipping cloud test.",
)

# Every REAL network test carries this (opt-in gate in conftest.py).
cloud = pytest.mark.cloud


# ----------------------------------------------------------------
# In-memory procedure (NOT a frozen fixture)
# ----------------------------------------------------------------

def _procedure():
    return load_procedure_from_dict(
        {
            "procedure_id": "BLACK_SCREEN",
            "version": 1,
            "issue": "black screen",
            "entry_step": "RESTART_SOFTWARE",
            "steps": [
                {
                    "id": "RESTART_SOFTWARE",
                    "action": (
                        "Перезапустите AngioPlus Core и дождитесь "
                        "полной загрузки."
                    ),
                    "evidence_query": "angioplus black screen restart",
                    "transitions": {
                        "attempted_failed": "CHECK_CABLE",
                        "resolved": "RESOLVED",
                    },
                },
                {
                    "id": "CHECK_CABLE",
                    "action": (
                        "Переподключите кабель дисплея с обоих концов."
                    ),
                    "evidence_query": "angioplus black screen display cable",
                    "transitions": {
                        "attempted_failed": "ESCALATE",
                        "resolved": "RESOLVED",
                    },
                },
            ],
            "terminators": ["ESCALATE", "RESOLVED"],
        }
    )


# ----------------------------------------------------------------
# REAL provider: canonical Alice boundary, built inline (NO legacy import)
# ----------------------------------------------------------------

# The canonical Alice boundary constants. Composed HERE from the public
# environment + the documented Yandex Cloud endpoint, so this test does NOT
# import yandex.chat.alice_support_agent or yandex.chat.server.
_ALICE_BASE_URL = "https://ai.api.cloud.yandex.net/v1"
# Validation runs on Alice AI LLM Flash (cheaper tier). The identifier was
# confirmed against the live AI Studio models endpoint
# (GET https://ai.api.cloud.yandex.net/v1/models) for this folder on
# 2026-06: gpt://<folder>/aliceai-llm-flash/latest.
_ALICE_MODEL_TEMPLATE = "gpt://{folder_id}/aliceai-llm-flash/latest"
# Mirrors the production semantic stage's output budget.
_ALICE_MAX_OUTPUT_TOKENS = 1000


def _alice_model_id(folder_id: str) -> str:
    """The Alice Flash support model id: gpt://<folder>/aliceai-llm-flash/latest."""
    return _ALICE_MODEL_TEMPLATE.format(folder_id=folder_id)


def _build_semantic_schema() -> dict:
    """The strict JSON schema the Slice 4 analyzer expects back."""
    return {
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
        },
        "required": [
            "scope",
            "reported_actions",
            "problem_status",
            "human_requested",
            "clarification_facts",
            "new_issue",
        ],
    }


def _real_provider(prompt: str, history):
    """Call the real Alice model directly at the canonical boundary.

    The client and model id are built INLINE here from the public Yandex
    Cloud endpoint and ``YANDEX_FOLDER_ID`` (composed as
    ``gpt://<folder>/aliceai-llm-flash/latest``). No legacy support module is
    imported -- this test must not depend on
    ``yandex/chat/alice_support_agent.py`` or ``yandex/chat/server.py``.

        client = OpenAI(api_key=..., base_url=<yandex v1>, project=folder)
        client.responses.create(
            model="gpt://<folder>/aliceai-llm-flash/latest",
            instructions=...,
            input=[{"role": "user", "content": prompt}],
            text={"format": {"type": "json_schema", "name": ..., "strict": True,
                             "schema": {...}}},
            store=False,
        )

    Returns the raw JSON TEXT (a str). Any SDK/network error propagates and
    is turned into the analyzer fail-safe by ``analyze``.

    The raw model output is printed so ``pytest -s`` shows exactly what the
    model returned (needed to diagnose a semantic-mismatch failure).
    """
    # Local import: the SDK and env are only needed when --run-cloud is set.
    from openai import OpenAI

    api_key = os.environ["YANDEX_API_KEY"]
    folder_id = os.environ["YANDEX_FOLDER_ID"]
    client = OpenAI(
        api_key=api_key,
        base_url=_ALICE_BASE_URL,
        project=folder_id,
    )
    model = _alice_model_id(folder_id)

    fmt = {
        "type": "json_schema",
        "name": "angioplus_semantic_observation",
        "strict": True,
        "schema": _build_semantic_schema(),
    }

    print("\n================ ALICE CLOUD CALL ================")
    print("model:", model)
    print("prompt (first 400 chars):", prompt[:400])

    response = client.responses.create(
        model=model,
        temperature=0,
        instructions=prompt,
        input=[{"role": "user", "content": prompt}],
        max_output_tokens=_ALICE_MAX_OUTPUT_TOKENS,
        store=False,
        text={"format": fmt},
    )

    raw = (getattr(response, "output_text", "") or "").strip()

    print("--- RAW MODEL OUTPUT ---")
    print(raw)
    print("--- STATUS ---")
    print(getattr(response, "status", None))
    print("--- INCOMPLETE ---")
    print(getattr(response, "incomplete_details", None))
    print("===============================================\n")
    return raw


def _analyze(message, *, history=None, last_recommendation=None, procedure=None):
    proc = procedure if procedure is not None else _procedure()
    return analyze(
        message=message,
        history=history or [],
        procedure=proc,
        last_recommendation=last_recommendation,
        provider=_real_provider,
    )


# ================================================================
# 1. FAILED attribution: "already restarted twice, nothing changed"
# ================================================================

@cloud
@needs_creds
def test_cloud_failed_restart_attributed():
    obs = _analyze(
        "Я уже два раза перезапускал, ничего не изменилось",
        history=[
            {"role": "user", "text": "После запуска чёрный экран."},
            {
                "role": "assistant",
                "text": (
                    "Перезапустите AngioPlus Core и дождитесь полной "
                    "загрузки."
                ),
            },
        ],
        last_recommendation="RESTART_SOFTWARE",
        procedure=_procedure(),
    )
    assert obs.scope == ObservationScope.IN_SCOPE
    assert obs.problem_status == ProblemStatus.ONGOING
    # The restart must be reported as FAILED and attributed to its step.
    matched = [
        a
        for a in obs.reported_actions
        if a.step_id == "RESTART_SOFTWARE" and a.outcome == StepOutcome.FAILED
    ]
    assert matched, obs.reported_actions


# ================================================================
# 2. EXACT SKIPPED: "I already did that" with last rec = CHECK_CABLE
# ================================================================

@cloud
@needs_creds
def test_cloud_exact_skipped():
    obs = _analyze(
        "Я это уже делал раньше",
        history=[
            {"role": "user", "text": "Чёрный экран на AngioPlus."},
            {
                "role": "assistant",
                "text": "Переподключите кабель дисплея с обоих концов.",
            },
        ],
        last_recommendation="CHECK_CABLE",
        procedure=_procedure(),
    )
    matched = [
        a
        for a in obs.reported_actions
        if a.step_id == "CHECK_CABLE" and a.outcome == StepOutcome.SKIPPED
    ]
    assert matched, obs.reported_actions


# ================================================================
# 3. EXACT DONE_UNKNOWN: "Yes, I did it" (result not stated)
# ================================================================

@cloud
@needs_creds
def test_cloud_exact_done_unknown():
    obs = _analyze(
        "Да, я это сделал только что",
        history=[
            {"role": "user", "text": "Чёрный экран на AngioPlus."},
            {
                "role": "assistant",
                "text": "Переподключите кабель дисплея с обоих концов.",
            },
        ],
        last_recommendation="CHECK_CABLE",
        procedure=_procedure(),
    )
    matched = [
        a
        for a in obs.reported_actions
        if a.step_id == "CHECK_CABLE"
        and a.outcome == StepOutcome.DONE_UNKNOWN
    ]
    assert matched, obs.reported_actions


# ================================================================
# 4. MULTI-ACTION: restart + cable, error remains
# ================================================================

@cloud
@needs_creds
def test_cloud_multi_action():
    obs = _analyze(
        (
            "Я перезапускал систему, потом переподключал кабель дисплея, "
            "но чёрный экран всё равно остаётся."
        ),
        history=[
            {"role": "user", "text": "После запуска чёрный экран."},
            {
                "role": "assistant",
                "text": (
                    "Перезапустите AngioPlus Core и дождитесь полной "
                    "загрузки."
                ),
            },
        ],
        last_recommendation="RESTART_SOFTWARE",
        procedure=_procedure(),
    )
    assert obs.problem_status == ProblemStatus.ONGOING

    failed_ids = {
        a.step_id
        for a in obs.reported_actions
        if a.outcome == StepOutcome.FAILED
    }
    # Both attempted actions should surface as FAILED.
    assert {"RESTART_SOFTWARE", "CHECK_CABLE"} <= failed_ids, obs.reported_actions


# ================================================================
# 5. HUMAN REQUEST (in scope)
# ================================================================

@cloud
@needs_creds
def test_cloud_human_request_in_scope():
    obs = _analyze(
        "Это AngioPlus не работает, соедините меня с живым специалистом.",
        history=[{"role": "user", "text": "Чёрный экран на AngioPlus."}],
        last_recommendation="RESTART_SOFTWARE",
        procedure=_procedure(),
    )
    assert obs.scope == ObservationScope.IN_SCOPE
    assert obs.human_requested is True


# ================================================================
# 6. HUMAN REQUEST (out of scope)
# ================================================================

@cloud
@needs_creds
def test_cloud_human_request_out_of_scope():
    obs = _analyze(
        "Соедините меня с оператором, чтобы узнать погоду на завтра.",
        history=[],
        last_recommendation=None,
        procedure=_procedure(),
    )
    assert obs.scope == ObservationScope.OUT_OF_SCOPE
    assert obs.human_requested is True


# ================================================================
# 7. Sanity: the analyzer never emits routing fields
# ================================================================

@cloud
@needs_creds
def test_cloud_no_routing_fields_in_observation():
    import dataclasses

    obs = _analyze(
        "Перезапускал, не помогло",
        history=[],
        last_recommendation="RESTART_SOFTWARE",
        procedure=_procedure(),
    )
    names = {f.name for f in dataclasses.fields(obs)}
    for forbidden in (
        "candidate_l1_action",
        "candidate_action_already_attempted",
        "decision",
        "action",
    ):
        assert forbidden not in names


# ================================================================
# 8. Sanity: allowlist build excludes the graph
# ================================================================

def test_cloud_allowlist_excludes_graph():
    allowed = build_allowed_steps(_procedure())
    for entry in allowed:
        assert set(entry) == {"step_id", "action"}
    # None of the graph vocabulary may leak into the allowlist.
    blob = json.dumps(allowed)
    assert "attempted_failed" not in blob
    assert "transitions" not in blob
    assert "ESCALATE" not in blob


# ================================================================
# 9. Anti-regression: NO legacy Alice pipeline import in this module
# ================================================================

def test_cloud_provider_imports_no_legacy_pipeline():
    """The cloud provider must build the canonical Alice boundary inline.

    This is enforced by AST so the guard runs even WITHOUT --run-cloud
    (unlike the semantic assertions above, which are skipped without the
    flag). The file must never import the legacy support modules:
      * ``yandex.chat.alice_support_agent`` (``apply_support_policy`` etc.)
      * ``yandex.chat.server`` (the production two-stage pipeline)
    """
    import ast
    from pathlib import Path

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden = {
        "yandex.chat.alice_support_agent",
        "yandex.chat.server",
    }
    leaked = {
        module
        for module in imported
        for bad in forbidden
        if module == bad or module.startswith(bad + ".")
    }
    assert not leaked, f"legacy Alice pipeline import leaked: {sorted(leaked)}"

    # And no bare ``import yandex`` that could pull the pipeline in.
    assert not any(
        m == "yandex" or m.startswith("yandex.") for m in imported
    ), f"unexpected yandex import: {sorted(imported)}"


def test_cloud_model_id_is_canonical():
    """The model id is the canonical Alice Flash boundary form."""
    assert _alice_model_id("my-folder") == (
        "gpt://my-folder/aliceai-llm-flash/latest"
    )
