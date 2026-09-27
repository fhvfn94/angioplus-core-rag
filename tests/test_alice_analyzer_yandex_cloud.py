# -*- coding: utf-8 -*-
"""Opt-in REAL-Yandex tests for the Slice 11A.2 analyzer-provider adapter.

Unlike ``tests/test_alice_analyzer_cloud.py`` (which builds a provider
INLINE), these tests exercise the PRODUCTION adapter
(:class:`support.alice_analyzer_yandex.YandexAnalyzerProvider`) end to end:
the adapter -- with a REAL SDK client -- is injected into the FROZEN Slice 4
analyzer (``support.alice_analyzer.analyze``).

They hit the network, so they are gated behind ``@pytest.mark.cloud`` and are
SKIPPED unless you pass ``--run-cloud`` AND set credentials:

    python -m pytest tests/test_alice_analyzer_yandex_cloud.py --run-cloud -q

Budget: deliberately SMALL (two real model calls) -- the adapter is a thin
transport, so one semantic assertion and one raw-shape assertion are enough.
No File Search is performed; only the analyzer boundary is exercised.
"""

from __future__ import annotations

import json
import os

import pytest

from support import alice_analyzer_yandex as ay
from support.alice_analyzer import (
    analyze,
    build_allowed_steps,
    build_analyzer_prompt,
)
from support.observation import (
    ObservationScope,
    ProblemStatus,
    StepOutcome,
)
from support.procedure import load_procedure_from_dict


# ``cloud`` is applied PER network test (never at module level): the pure
# static guard below must ALWAYS run and must not be skippable by omitting
# ``--run-cloud``.

_HAVE_CREDS = bool(
    os.getenv("YANDEX_API_KEY", "").strip()
    and os.getenv("YANDEX_FOLDER_ID", "").strip()
)

needs_creds = pytest.mark.skipif(
    not _HAVE_CREDS,
    reason="YANDEX_API_KEY / YANDEX_FOLDER_ID not set; skipping cloud test.",
)

cloud = pytest.mark.cloud


def _procedure():
    return load_procedure_from_dict(
        {
            "procedure_id": "BLACK_SCREEN",
            "version": 1,
            "issue": "black screen",
            "entry_step": "RESTART_SOFTWARE",
            "terminators": ["ESCALATE", "RESOLVED"],
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
        }
    )


_MESSAGE = "Я уже два раза перезапускал AngioPlus, ничего не изменилось"

_HISTORY = [
    {"role": "user", "text": "После запуска чёрный экран."},
    {
        "role": "assistant",
        "text": "Перезапустите AngioPlus Core и дождитесь полной загрузки.",
    },
]


# ================================================================
# 1. The PRODUCTION adapter feeds the FROZEN analyzer (one real call)
# ================================================================

@cloud
@needs_creds
def test_cloud_adapter_feeds_frozen_analyzer():
    provider = ay.YandexAnalyzerProvider()  # real client, lazy, from env

    observation = analyze(
        message=_MESSAGE,
        history=_HISTORY,
        procedure=_procedure(),
        last_recommendation="RESTART_SOFTWARE",
        provider=provider,
    )

    assert observation.scope == ObservationScope.IN_SCOPE
    assert observation.problem_status == ProblemStatus.ONGOING

    # The restart must be reported FAILED and attributed to its step -- which
    # proves the strict schema + parser contract survives a real round-trip.
    matched = [
        action
        for action in observation.reported_actions
        if action.step_id == "RESTART_SOFTWARE"
        and action.outcome == StepOutcome.FAILED
    ]
    assert matched, observation.reported_actions


# ================================================================
# 2. The adapter returns RAW, strictly-shaped JSON (one real call)
# ================================================================

@cloud
@needs_creds
def test_cloud_adapter_returns_raw_strict_json():
    provider = ay.YandexAnalyzerProvider()

    prompt = build_analyzer_prompt(
        message=_MESSAGE,
        history=_HISTORY,
        allowed_steps=build_allowed_steps(_procedure()),
        last_recommendation="RESTART_SOFTWARE",
    )
    raw = provider(prompt, list(_HISTORY))

    # The adapter returns text, not a parsed object.
    assert isinstance(raw, str)

    data = json.loads(raw)
    assert isinstance(data, dict)
    # The strict seven-field contract (no extras): the adapter still owns no
    # parsing -- this only proves the model honoured the pinned schema.
    assert set(data) == {
        "scope",
        "reported_actions",
        "problem_status",
        "human_requested",
        "clarification_facts",
        "new_issue",
        "request_kind",
    }


# ================================================================
# 3. Pure static guard (ALWAYS runs -- not skippable)
# ================================================================

def test_model_id_is_canonical():
    assert ay.DEFAULT_BASE_URL == "https://ai.api.cloud.yandex.net/v1"
    assert ay.build_analyzer_model_id("my-folder") == (
        "gpt://my-folder/aliceai-llm-flash/latest"
    )
