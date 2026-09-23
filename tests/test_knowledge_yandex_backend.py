# -*- coding: utf-8 -*-
"""Pure/local tests for support.knowledge_yandex (Slice 5C).

NO network, NO credentials, NO Yandex HTTP stack: the controller is a pure
in-memory fake. These tests pin the approved 5C behaviour:

  * the adapter implements the frozen 5A ``KnowledgeBackend`` contract;
  * ``EvidenceChunk`` -> ``KnowledgeItem`` field mapping is verbatim;
  * Yandex relevance order is preserved; a lower score that is earlier stays
    earlier; ``authority=100`` does NOT promote an item; IFU does NOT jump
    above FAQ;
  * deterministic RETAIN-FIRST deduplication (structural + exact-text);
  * distinct chunks of the same document survive;
  * evidence is capped at the requested max / default 5;
  * empty evidence and controller failures flow through the EXISTING 5A
    core as ``NOT_FOUND`` / ``ERROR``;
  * the adapter adds NO second retrieval stack and is NOT wired into any
    production runtime.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from support import knowledge as kb
from support import knowledge_yandex as kyb
from support.knowledge import KnowledgeItem, KnowledgeStatus
from support.knowledge_yandex import YandexKnowledgeBackend
from yandex.retrieval.models import EvidenceChunk


# ----------------------------------------------------------------
# Fakes / helpers
# ----------------------------------------------------------------

class _Response:
    def __init__(self, evidence):
        self.evidence = list(evidence)


class FakeController:
    """In-memory stand-in for RetrievalController. Records calls."""

    def __init__(self, evidence=(), exc=None):
        self._evidence = list(evidence)
        self._exc = exc
        self.calls: list[str] = []

    def search(self, query):  # matches RetrievalController.search(query)
        self.calls.append(query)
        if self._exc is not None:
            raise self._exc
        return _Response(self._evidence)


def chunk(text="evidence text", **kw) -> EvidenceChunk:
    kw.setdefault("rank", 1)
    kw.setdefault("score", 0.5)
    return EvidenceChunk(text=text, **kw)


def backend_with(evidence=(), *, exc=None, max_num_results=kb.DEFAULT_MAX_RESULTS):
    controller = FakeController(evidence=evidence, exc=exc)
    return YandexKnowledgeBackend(
        controller=controller, max_num_results=max_num_results
    ), controller


# ================================================================
# 1. CONTRACT / CONSTRUCTION
# ================================================================

def test_implements_knowledge_backend_contract():
    # The adapter must satisfy the 5A port: search(query, *, max_num_results).
    backend, _ = backend_with([chunk("x")])
    items = backend.search("q", max_num_results=kb.DEFAULT_MAX_RESULTS)
    assert list(items) == [
        KnowledgeItem(text="x", score=0.5, rank=1)
    ]


def test_max_num_results_must_be_positive():
    with pytest.raises(ValueError):
        YandexKnowledgeBackend(max_num_results=0)


def test_lazy_construction_does_not_touch_yandex_on_init():
    # No controller injected, but merely constructing must not import/build
    # the real stack or require credentials.
    YandexKnowledgeBackend()  # must not raise


def test_uses_exactly_one_controller_search():
    backend, controller = backend_with([chunk("a"), chunk("b")])
    result = kb.retrieve_general("black screen", backend=backend)
    assert result.status is KnowledgeStatus.FOUND
    assert controller.calls == ["black screen"]  # exactly one call


# ================================================================
# 2. FIELD MAPPING (verbatim)
# ================================================================

def test_evidence_chunk_maps_to_knowledge_item_verbatim():
    ev = EvidenceChunk(
        rank=3,
        score=0.814,
        text="  restart the software  ",
        source="AngioPlus IFU.pdf",
        source_type="ifu",
        authority=100,
        section="11",
        pages="9-10",
        file_id="f1",
        vector_store_filename="AngioPlus IFU.pdf",
    )
    backend, _ = backend_with([ev])
    items = backend.search("q", max_num_results=kb.DEFAULT_MAX_RESULTS)

    assert len(items) == 1
    item = items[0]
    assert item.text == "  restart the software  "
    assert item.source == "AngioPlus IFU.pdf"
    assert item.source_type == "ifu"
    assert item.authority == 100
    assert item.section == "11"
    assert item.pages == "9-10"
    assert item.file_id == "f1"
    assert item.filename == "AngioPlus IFU.pdf"  # from vector_store_filename
    assert item.score == 0.814
    assert item.rank == 3


# ================================================================
# 3. ORDER & NON-PROMOTION
# ================================================================

def test_yandex_order_is_preserved():
    evidence = [
        chunk("first", rank=1),
        chunk("second", rank=2),
        chunk("third", rank=3),
    ]
    backend, _ = backend_with(evidence)
    items = backend.search("q", max_num_results=kb.DEFAULT_MAX_RESULTS)
    assert [i.text for i in items] == ["first", "second", "third"]


def test_lower_score_earlier_stays_earlier():
    evidence = [
        chunk("low", score=0.10, rank=1),
        chunk("high", score=0.99, rank=2),
    ]
    backend, _ = backend_with(evidence)
    items = backend.search("q", max_num_results=kb.DEFAULT_MAX_RESULTS)
    assert [i.text for i in items] == ["low", "high"]


def test_authority_does_not_promote_an_item():
    evidence = [
        chunk("unknown-low", authority=10, rank=1),
        chunk("ifu-high", authority=100, rank=2),
    ]
    backend, _ = backend_with(evidence)
    items = backend.search("q", max_num_results=kb.DEFAULT_MAX_RESULTS)
    assert [i.text for i in items] == ["unknown-low", "ifu-high"]


def test_ifu_does_not_automatically_move_above_faq():
    evidence = [
        chunk("faq", source_type="faq", authority=70, rank=1),
        chunk("ifu", source_type="ifu", authority=100, rank=2),
    ]
    backend, _ = backend_with(evidence)
    items = backend.search("q", max_num_results=kb.DEFAULT_MAX_RESULTS)
    assert [i.text for i in items] == ["faq", "ifu"]
    assert [i.source_type for i in items] == ["faq", "ifu"]


# ================================================================
# 4. DEDUPLICATION (deterministic, retain-first)
# ================================================================

def test_structural_dedup_retains_first_highest_ranked():
    evidence = [
        chunk("first copy", file_id="doc", section="11", pages="9-10", rank=1),
        chunk("second copy", file_id="doc", section="11", pages="9-10", rank=2),
    ]
    backend, _ = backend_with(evidence)
    items = backend.search("q", max_num_results=kb.DEFAULT_MAX_RESULTS)
    assert [i.text for i in items] == ["first copy"]


def test_same_document_distinct_chunks_survive():
    evidence = [
        chunk("section 11", file_id="doc", section="11", pages="9-10"),
        chunk("section 12", file_id="doc", section="12", pages="11"),
    ]
    backend, _ = backend_with(evidence)
    items = backend.search("q", max_num_results=kb.DEFAULT_MAX_RESULTS)
    assert [i.text for i in items] == ["section 11", "section 12"]


def test_same_document_without_span_is_not_collapsed():
    # section/pages absent -> structural key NOT applied; distinct text kept.
    evidence = [
        chunk("chunk one", file_id="doc"),
        chunk("chunk two", file_id="doc"),
    ]
    backend, _ = backend_with(evidence)
    items = backend.search("q", max_num_results=kb.DEFAULT_MAX_RESULTS)
    assert [i.text for i in items] == ["chunk one", "chunk two"]


def test_exact_text_duplicate_is_collapsed():
    evidence = [
        chunk("same text"),
        chunk("  same   text  "),  # identical after normalization
    ]
    backend, _ = backend_with(evidence)
    items = backend.search("q", max_num_results=kb.DEFAULT_MAX_RESULTS)
    assert len(items) == 1
    assert items[0].text == "same text"


def test_internal_duplicate_is_skipped_not_reordered():
    evidence = [
        chunk("A", file_id="doc", section="1", pages="1-1", rank=1),
        chunk("B", file_id="doc", section="2", pages="2-2", rank=2),
        chunk("A dup", file_id="doc", section="1", pages="1-1", rank=3),
    ]
    backend, _ = backend_with(evidence)
    items = backend.search("q", max_num_results=kb.DEFAULT_MAX_RESULTS)
    assert [i.text for i in items] == ["A", "B"]


# ================================================================
# 5. MAX COUNT
# ================================================================

def test_default_cap_is_five():
    evidence = [chunk(f"e{i}", rank=i + 1) for i in range(7)]
    backend, _ = backend_with(evidence, max_num_results=kb.DEFAULT_MAX_RESULTS)
    items = backend.search("q", max_num_results=kb.DEFAULT_MAX_RESULTS)
    assert len(items) == 5
    assert [i.text for i in items] == ["e0", "e1", "e2", "e3", "e4"]


def test_requested_max_below_default_is_honored():
    evidence = [chunk(f"e{i}", rank=i + 1) for i in range(7)]
    backend, _ = backend_with(evidence, max_num_results=kb.DEFAULT_MAX_RESULTS)
    items = backend.search("q", max_num_results=2)
    assert [i.text for i in items] == ["e0", "e1"]


# ================================================================
# 6. EMPTY / ERROR THROUGH THE EXISTING 5A CORE
# ================================================================

def test_empty_evidence_is_not_found_through_core():
    backend, controller = backend_with([])
    result = kb.retrieve_general("q", backend=backend)
    assert result.status is KnowledgeStatus.NOT_FOUND
    assert result.items == ()
    assert controller.calls == ["q"]  # backend WAS called


def test_blank_text_only_is_not_found_through_core():
    backend, _ = backend_with([chunk(""), chunk("   ")])
    result = kb.retrieve_general("q", backend=backend)
    assert result.status is KnowledgeStatus.NOT_FOUND


def test_controller_exception_is_knowledge_error_through_core():
    backend, _ = backend_with(exc=RuntimeError("transport down"))
    result = kb.retrieve_general("q", backend=backend)
    assert result.status is KnowledgeStatus.ERROR
    assert result.items == ()
    assert result.error is not None
    assert "RuntimeError" in result.error


def test_blank_query_is_not_found_without_controller_call():
    backend, controller = backend_with([chunk("x")])
    result = kb.retrieve_general("   ", backend=backend)
    assert result.status is KnowledgeStatus.NOT_FOUND
    assert controller.calls == []


# ================================================================
# 7. DETERMINISM
# ================================================================

def test_repeated_output_is_deterministic():
    evidence = [
        chunk("b", file_id="d", section="1", pages="1", rank=1),
        chunk("a", file_id="d", section="2", pages="2", rank=2),
        chunk("b", file_id="d", section="1", pages="1", rank=3),
    ]
    backend, _ = backend_with(evidence)
    first = backend.search("q", max_num_results=kb.DEFAULT_MAX_RESULTS)
    second = backend.search("q", max_num_results=kb.DEFAULT_MAX_RESULTS)
    assert [i.text for i in first] == [i.text for i in second] == ["b", "a"]


# ================================================================
# 8. ANTI-REGRESSION GUARDS (static; no runtime import)
# ================================================================

def _module_path() -> Path:
    return Path(kyb.__file__)


def _module_level_imports() -> set[str]:
    tree = ast.parse(_module_path().read_text(encoding="utf-8"))
    found: set[str] = set()
    # ONLY top-level (module-scope) imports: the Yandex import must be lazy.
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_no_top_level_provider_or_http_imports():
    forbidden_roots = {
        "yandex",
        "httpx",
        "requests",
        "aiohttp",
        "openai",
        "psycopg",
        "psycopg2",
        "sqlalchemy",
    }
    for name in _module_level_imports():
        assert name.split(".")[0] not in forbidden_roots, (
            f"forbidden top-level import: {name}"
        )


def test_only_allowed_top_level_imports():
    allowed = {"__future__", "typing", "support"}
    roots = {n.split(".")[0] for n in _module_level_imports()}
    assert roots <= allowed, f"unexpected imports: {roots - allowed}"


def test_does_not_duplicate_retrieval_endpoint():
    source = _module_path().read_text(encoding="utf-8")
    assert "ai.api.cloud.yandex.net" not in source


def test_only_knowledge_core_imported_from_support():
    support_mods = {
        n
        for n in _module_level_imports()
        if n == "support" or n.startswith("support.")
    }
    assert support_mods == {"support.knowledge"}


def test_public_api_surface():
    assert set(kyb.__all__) == {"YandexKnowledgeBackend"}


_RUNTIME_MODULES = (
    Path("yandex/chat/server.py"),
    Path("yandex/chat/alice_support_agent.py"),
    Path("yandex/realtime/web/server.py"),
)


def test_not_wired_into_any_production_runtime():
    root = _module_path().resolve().parents[1]
    for rel in _RUNTIME_MODULES:
        path = root / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        assert "knowledge_yandex" not in text, f"{rel} wires the 5C adapter"
        assert "YandexKnowledgeBackend" not in text, (
            f"{rel} wires the 5C adapter"
        )
