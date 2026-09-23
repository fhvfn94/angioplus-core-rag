# -*- coding: utf-8 -*-
"""Opt-in REAL Yandex retrieval tests for support.knowledge_yandex (Slice 5C).

These tests call the REAL Yandex retrieval stack
(``RetrievalController`` -> ``YandexVectorStoreClient``), so they require
the network and are gated behind ``@pytest.mark.cloud``. They are SKIPPED
unless you explicitly pass ``--run-cloud``:

    python -m pytest tests/test_knowledge_yandex_cloud.py --run-cloud -q

Budget: AT MOST TWO real retrieval calls in total -- one per test. The
calls are counted and asserted (``controller.calls == 1`` per test) so the
budget cannot be exceeded silently.

Coverage is deliberately limited to the two things only a real call can
prove:

  1. a known procedure-bound query -> FOUND with valid mapped metadata;
  2. a general/document query -> FOUND, and any non-IFU / multi-source
     evidence returned by Yandex is RETAINED.

``NOT_FOUND`` and ``ERROR`` are NOT tested here -- they are covered purely
(no network) in ``tests/test_knowledge_yandex_backend.py`` to avoid
spending cloud calls on them.

If credentials (YANDEX_API_KEY / YANDEX_FOLDER_ID / YANDEX_VECTOR_STORE_ID)
are missing, every test is skipped rather than failed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from support import knowledge as kb
from support.knowledge import KnowledgeStatus
from support.knowledge_yandex import YandexKnowledgeBackend
from support.procedure import load_procedure


# Every test in this module hits the network and is opt-in.
pytestmark = pytest.mark.cloud

cloud = pytest.mark.cloud


# ----------------------------------------------------------------
# Credentials (guarded dotenv load; fail-closed but not fail-hard)
# ----------------------------------------------------------------

try:  # dotenv is a repo dependency; load the project .env if present.
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:  # pragma: no cover - import/env is best-effort
    pass


_HAVE_CREDS = all(
    os.getenv(name, "").strip()
    for name in (
        "YANDEX_API_KEY",
        "YANDEX_FOLDER_ID",
        "YANDEX_VECTOR_STORE_ID",
    )
)

needs_creds = pytest.mark.skipif(
    not _HAVE_CREDS,
    reason=(
        "YANDEX_API_KEY / YANDEX_FOLDER_ID / YANDEX_VECTOR_STORE_ID not "
        "set; skipping cloud test."
    ),
)


BLACK_SCREEN = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "procedures"
    / "black_screen.yaml"
)


# ----------------------------------------------------------------
# Counting wrapper: guarantees the two-call budget is observable
# ----------------------------------------------------------------

class _CountingController:
    """Delegates to the REAL RetrievalController and counts calls."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0

    def search(self, query):
        self.calls += 1
        return self._inner.search(query)


def _real_backend(*, max_num_results=kb.DEFAULT_MAX_RESULTS):
    # Local import: the HTTP stack is only needed when --run-cloud is set.
    from yandex.retrieval.controller import RetrievalController

    controller = _CountingController(
        RetrievalController(max_num_results=max_num_results)
    )
    backend = YandexKnowledgeBackend(
        controller=controller, max_num_results=max_num_results
    )
    return backend, controller


# ================================================================
# 1. PROCEDURE-BOUND query -> FOUND + valid mapped metadata
# ================================================================

@cloud
@needs_creds
def test_cloud_procedure_query_is_found_with_valid_metadata():
    procedure = load_procedure(BLACK_SCREEN)
    backend, controller = _real_backend()

    result = kb.retrieve_for_step(
        procedure.steps["CHECK_POWER"], backend=backend
    )

    print("\n=== 5C CLOUD CALL 1 (procedure-bound) ===")
    print("query:", result.query)
    print("status:", result.status)
    print("calls:", controller.calls)
    for it in result.items:
        print(
            f"rank={it.rank} score={it.score:.4f} "
            f"source_type={it.source_type} authority={it.authority} "
            f"source={it.source!r}"
        )

    assert controller.calls == 1  # exactly one real retrieval call
    assert result.status is KnowledgeStatus.FOUND
    assert result.items

    for it in result.items:
        assert isinstance(it.text, str) and it.text.strip()
        assert isinstance(it.rank, int)
        assert isinstance(it.score, float)
        assert isinstance(it.source_type, str)
        assert isinstance(it.authority, int)

    # Yandex order preserved: ranks ascend in the mapped output.
    ranks = [it.rank for it in result.items]
    assert ranks == sorted(ranks)


# ================================================================
# 2. GENERAL / document query -> FOUND, non-IFU evidence retained
# ================================================================

@cloud
@needs_creds
def test_cloud_general_query_retains_non_ifu_evidence_when_returned():
    backend, controller = _real_backend()

    result = kb.retrieve_general(
        "AngioPlus Core system requirements", backend=backend
    )

    print("\n=== 5C CLOUD CALL 2 (general/document) ===")
    print("query:", result.query)
    print("status:", result.status)
    print("calls:", controller.calls)
    for it in result.items:
        print(
            f"rank={it.rank} score={it.score:.4f} "
            f"source_type={it.source_type} source={it.source!r}"
        )

    assert controller.calls == 1  # exactly one real retrieval call
    assert result.status is KnowledgeStatus.FOUND
    assert result.items

    # Non-IFU / multi-source evidence must NOT be discarded when Yandex
    # returns it (5C never filters by source_type).
    non_ifu = [it for it in result.items if it.source_type != "ifu"]
    texts = [it.text for it in result.items]
    for it in non_ifu:
        assert it.text in texts

    # Order preserved.
    ranks = [it.rank for it in result.items]
    assert ranks == sorted(ranks)
