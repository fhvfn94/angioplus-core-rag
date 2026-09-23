# -*- coding: utf-8 -*-
"""Contract tests for support.knowledge (Slice 5A -- PURE: no I/O, no network).

This file pins the approved 5A + 5B contract:

  * ``KnowledgeStatus`` is a closed three-value enum;
  * ``KnowledgeResult`` enforces its FOUND / NOT_FOUND / ERROR invariants;
  * the execution core (``_retrieve``) maps every backend outcome
    (items / empty / blank-only / exception / malformed) to the right
    status and NEVER lets an exception escape;
  * the result carries no usable evidence in NOT_FOUND / ERROR and carries
    internal error information in ERROR;
  * the contract objects are immutable;
  * anti-regression guards: no routing/workflow vocabulary, no forbidden
    imports, no network/provider dependency, and no wiring into
    ``support/__init__.py``;
  * 5B entry points: ``retrieve_for_step`` uses ``ProcedureStep.evidence_query``
    as the authoritative base and may only APPEND a refinement, while
    ``retrieve_general`` normalizes a factual question; both reuse the same
    ``_retrieve`` core, so FOUND/NOT_FOUND/ERROR semantics are untouched.

Source priority (IFU / authority) and the Yandex backend are LATER
checkpoints (5C/5D) and are deliberately NOT covered here.
"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from support import knowledge as kb
from support.knowledge import (
    KnowledgeItem,
    KnowledgeResult,
    KnowledgeStatus,
)
from support.procedure import ProcedureStep, load_procedure


BLACK_SCREEN = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "procedures"
    / "black_screen.yaml"
)


# ----------------------------------------------------------------
# Helpers / fakes
# ----------------------------------------------------------------

def item(text="evidence text", **kw) -> KnowledgeItem:
    return KnowledgeItem(text=text, **kw)


def step(
    step_id="STEP_A",
    evidence_query="test query A",
    action="Generic test action.",
) -> ProcedureStep:
    return ProcedureStep(
        id=step_id,
        action=action,
        evidence_query=evidence_query,
        transitions={"attempted_failed": "ESCALATE", "resolved": "RESOLVED"},
    )


class FakeBackend:
    """In-memory KnowledgeBackend. Records calls; optional canned failure."""

    def __init__(self, items=(), exc=None):
        self._items = list(items)
        self._exc = exc
        self.calls: list[tuple[str, int]] = []

    def search(self, query, *, max_num_results):
        self.calls.append((query, max_num_results))
        if self._exc is not None:
            raise self._exc
        return list(self._items)


# ================================================================
# 1. STATUS VOCABULARY -- closed enum
# ================================================================

def test_status_is_closed_three_value_enum():
    assert {s.value for s in KnowledgeStatus} == {
        "found",
        "not_found",
        "error",
    }


# ================================================================
# 2. CORE MAPPING -- FOUND / NOT_FOUND / ERROR
# ================================================================

def test_non_empty_evidence_is_found():
    backend = FakeBackend(items=[item("restart the software")])
    result = kb._retrieve("black screen", backend=backend)
    assert result.status is KnowledgeStatus.FOUND
    assert len(result.items) == 1
    assert result.items[0].text == "restart the software"
    assert result.error is None
    assert result.query == "black screen"


def test_empty_backend_result_is_not_found():
    backend = FakeBackend(items=[])
    result = kb._retrieve("black screen", backend=backend)
    assert result.status is KnowledgeStatus.NOT_FOUND
    assert result.items == ()
    assert result.error is None


def test_blank_evidence_only_is_not_found():
    backend = FakeBackend(items=[item(""), item("   "), item("\n\t ")])
    result = kb._retrieve("black screen", backend=backend)
    assert result.status is KnowledgeStatus.NOT_FOUND
    assert result.items == ()
    assert backend.calls  # backend WAS called; only the result was unusable


def test_mixed_usable_and_blank_keeps_only_usable():
    backend = FakeBackend(items=[item("  "), item("real evidence"), item("")])
    result = kb._retrieve("q", backend=backend)
    assert result.status is KnowledgeStatus.FOUND
    assert [i.text for i in result.items] == ["real evidence"]


def test_non_item_entries_are_ignored():
    backend = FakeBackend(items=["not an item", 42, None])
    result = kb._retrieve("q", backend=backend)
    assert result.status is KnowledgeStatus.NOT_FOUND
    assert result.items == ()


def test_backend_exception_is_error():
    backend = FakeBackend(exc=RuntimeError("boom"))
    result = kb._retrieve("black screen", backend=backend)
    assert result.status is KnowledgeStatus.ERROR
    assert result.items == ()
    assert result.error is not None
    assert "RuntimeError" in result.error
    assert "boom" in result.error


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("x"),
        ValueError("y"),
        KeyError("z"),
        TypeError("t"),
        TimeoutError("slow"),
    ],
)
def test_no_exception_escapes_from_core(exc):
    backend = FakeBackend(exc=exc)
    result = kb._retrieve("q", backend=backend)  # must not raise
    assert result.status is KnowledgeStatus.ERROR
    assert result.error is not None


def test_malformed_backend_response_is_error_not_crash():
    class BadBackend:
        def search(self, query, *, max_num_results):
            return 12345  # not iterable -> tuple() raises TypeError

    result = kb._retrieve("q", backend=BadBackend())
    assert result.status is KnowledgeStatus.ERROR


def test_blank_query_is_not_found_and_backend_not_called():
    backend = FakeBackend(items=[item("x")])
    result = kb._retrieve("   ", backend=backend)
    assert result.status is KnowledgeStatus.NOT_FOUND
    assert result.query == ""
    assert backend.calls == []


def test_max_num_results_is_passed_through():
    backend = FakeBackend(items=[item("x")])
    kb._retrieve("q", backend=backend, max_num_results=3)
    assert backend.calls == [("q", 3)]


def test_backend_order_is_preserved_no_priority_yet():
    # 5A performs NO source priority / sorting; backend order is authoritative.
    items = [
        item("low", authority=10, score=0.2, rank=1),
        item("high", authority=100, score=0.9, rank=2),
    ]
    result = kb._retrieve("q", backend=FakeBackend(items=items))
    assert [i.text for i in result.items] == ["low", "high"]


def test_item_metadata_round_trips():
    src = KnowledgeItem(
        text="t",
        source="doc.pdf",
        section="11",
        pages="9-10",
        filename="doc.pdf",
        file_id="f1",
        source_type="ifu",
        authority=100,
        score=0.77,
        rank=2,
    )
    result = kb._retrieve("q", backend=FakeBackend(items=[src]))
    assert result.items == (src,)


# ================================================================
# 3. RESULT INVARIANTS
# ================================================================

def test_found_requires_usable_item():
    with pytest.raises(ValueError):
        KnowledgeResult(status=KnowledgeStatus.FOUND, query="q", items=())


def test_found_rejects_only_blank_items():
    with pytest.raises(ValueError):
        KnowledgeResult(
            status=KnowledgeStatus.FOUND, query="q", items=(item("  "),)
        )


def test_found_rejects_error():
    with pytest.raises(ValueError):
        KnowledgeResult(
            status=KnowledgeStatus.FOUND,
            query="q",
            items=(item("x"),),
            error="nope",
        )


def test_not_found_rejects_usable_evidence():
    with pytest.raises(ValueError):
        KnowledgeResult(
            status=KnowledgeStatus.NOT_FOUND, query="q", items=(item("x"),)
        )


def test_not_found_rejects_error():
    with pytest.raises(ValueError):
        KnowledgeResult(
            status=KnowledgeStatus.NOT_FOUND, query="q", error="nope"
        )


def test_error_rejects_evidence():
    with pytest.raises(ValueError):
        KnowledgeResult(
            status=KnowledgeStatus.ERROR,
            query="q",
            items=(item("x"),),
            error="e",
        )


def test_error_requires_error_info():
    with pytest.raises(ValueError):
        KnowledgeResult(status=KnowledgeStatus.ERROR, query="q", error=None)


def test_valid_results_construct():
    assert KnowledgeResult(
        status=KnowledgeStatus.NOT_FOUND, query="q"
    ).items == ()
    assert KnowledgeResult(
        status=KnowledgeStatus.FOUND, query="q", items=(item("x"),)
    ).error is None


# ================================================================
# 4. IMMUTABILITY
# ================================================================

def test_item_is_immutable():
    it = item("x")
    with pytest.raises(FrozenInstanceError):
        it.text = "y"  # type: ignore[misc]


def test_result_is_immutable():
    res = kb._retrieve("q", backend=FakeBackend(items=[item("x")]))
    with pytest.raises(FrozenInstanceError):
        res.status = KnowledgeStatus.NOT_FOUND  # type: ignore[misc]


def test_items_are_tuple():
    res = kb._retrieve("q", backend=FakeBackend(items=[item("x")]))
    assert isinstance(res.items, tuple)


# ================================================================
# 5. ANTI-REGRESSION GUARDS
# ================================================================

def _module_path() -> Path:
    return Path(kb.__file__)


def _module_imports() -> set[str]:
    tree = ast.parse(_module_path().read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_no_forbidden_imports():
    forbidden_roots = {
        "yandex",
        "openai",
        "requests",
        "httpx",
        "aiohttp",
        "psycopg",
        "psycopg2",
        "sqlalchemy",
    }
    forbidden_modules = {
        "support.router",
        "support.procedure_state",
        "support.alice_analyzer",
    }
    for name in _module_imports():
        root = name.split(".")[0]
        assert root not in forbidden_roots, f"forbidden import: {name}"
        for mod in forbidden_modules:
            assert name != mod and not name.startswith(mod + "."), (
                f"forbidden import: {name}"
            )


def test_only_allowed_imports():
    allowed = {"__future__", "dataclasses", "enum", "typing", "support"}
    roots = {name.split(".")[0] for name in _module_imports()}
    assert roots <= allowed, f"unexpected imports: {roots - allowed}"


def test_only_support_procedure_is_imported():
    # 5B may import the declarative ProcedureStep type -- and nothing else from
    # the support package (no router / procedure state / analyzer).
    support_mods = {
        n
        for n in _module_imports()
        if n == "support" or n.startswith("support.")
    }
    assert support_mods == {"support.procedure"}


def test_no_routing_or_workflow_vocabulary():
    source = _module_path().read_text(encoding="utf-8")
    for token in (
        "RoutingDecision",
        "RouteAction",
        "EscalateReason",
        "ClarifyReason",
        "candidate_l1_action",
        "candidate_action_already_attempted",
        "RECOMMEND_STEP",
        "OUT_OF_SCOPE",
        "next_step",
        "transition",
    ):
        assert token not in source, f"forbidden token present: {token}"


def test_not_wired_into_package_init():
    init = (_module_path().parent / "__init__.py").read_text(encoding="utf-8")
    assert "knowledge" not in init


def test_public_api_surface():
    assert set(kb.__all__) == {
        "KnowledgeStatus",
        "KnowledgeItem",
        "KnowledgeResult",
        "KnowledgeBackend",
        "retrieve_for_step",
        "retrieve_general",
    }


# ================================================================
# 6. PROCEDURE-BOUND RETRIEVAL (5B)
# ================================================================

def test_step_evidence_query_is_the_retrieval_base():
    backend = FakeBackend(items=[item("evidence")])
    result = kb.retrieve_for_step(
        step(evidence_query="test query A"), backend=backend
    )
    assert result.status is KnowledgeStatus.FOUND
    assert backend.calls == [("test query A", kb.DEFAULT_MAX_RESULTS)]
    assert result.query == "test query A"


def test_step_query_is_normalized():
    backend = FakeBackend(items=[item("evidence")])
    result = kb.retrieve_for_step(
        step(evidence_query="  test   query   A  "), backend=backend
    )
    assert backend.calls[0][0] == "test query A"
    assert result.query == "test query A"


def test_refinement_is_appended_never_substituted():
    backend = FakeBackend(items=[item("evidence")])
    result = kb.retrieve_for_step(
        step(evidence_query="base objective"),
        refinement="usb-c only",
        backend=backend,
    )
    assert backend.calls[0][0] == "base objective usb-c only"
    assert result.query.startswith("base objective")


def test_refinement_is_normalized_before_appending():
    backend = FakeBackend(items=[item("evidence")])
    kb.retrieve_for_step(
        step(evidence_query="base"),
        refinement="  usb   c  ",
        backend=backend,
    )
    assert backend.calls[0][0] == "base usb c"


@pytest.mark.parametrize("refinement", [None, "", "   ", "\n\t "])
def test_blank_refinement_behaves_as_absent(refinement):
    backend = FakeBackend(items=[item("evidence")])
    kb.retrieve_for_step(
        step(evidence_query="base"), refinement=refinement, backend=backend
    )
    assert backend.calls[0][0] == "base"


@pytest.mark.parametrize("missing", [None, "", "   ", "\n"])
def test_missing_or_blank_evidence_query_is_not_found_without_backend_call(
    missing,
):
    backend = FakeBackend(items=[item("evidence")])
    result = kb.retrieve_for_step(
        step(evidence_query=missing), backend=backend
    )
    assert result.status is KnowledgeStatus.NOT_FOUND
    assert result.items == ()
    assert backend.calls == []


def test_no_fallback_to_step_action():
    backend = FakeBackend(items=[item("evidence")])
    s = ProcedureStep(
        id="STEP_A",
        action="Confirm the power LED is lit.",
        evidence_query=None,
        transitions={"attempted_failed": "ESCALATE", "resolved": "RESOLVED"},
    )
    result = kb.retrieve_for_step(s, backend=backend)
    assert result.status is KnowledgeStatus.NOT_FOUND
    assert backend.calls == []
    assert "power" not in result.query.lower()


def test_requested_step_is_used_and_not_mutated():
    s = step(step_id="CHECK_CABLE", evidence_query="angioplus display cable")
    backend = FakeBackend(items=[item("evidence")])
    result = kb.retrieve_for_step(s, backend=backend)
    assert backend.calls[0][0] == "angioplus display cable"
    assert result.query == "angioplus display cable"
    # the caller's step object is untouched
    assert s.evidence_query == "angioplus display cable"
    assert s.id == "CHECK_CABLE"


def test_step_retrieval_uses_real_procedure_fixture():
    procedure = load_procedure(BLACK_SCREEN)
    backend = FakeBackend(items=[item("doc")])
    result = kb.retrieve_for_step(
        procedure.steps["CHECK_POWER"], backend=backend
    )
    assert result.status is KnowledgeStatus.FOUND
    assert backend.calls == [
        ("angioplus black screen power led", kb.DEFAULT_MAX_RESULTS)
    ]


def test_step_retrieval_maps_through_core_semantics():
    empty = kb.retrieve_for_step(step(), backend=FakeBackend(items=[]))
    assert empty.status is KnowledgeStatus.NOT_FOUND
    failed = kb.retrieve_for_step(
        step(), backend=FakeBackend(exc=RuntimeError("x"))
    )
    assert failed.status is KnowledgeStatus.ERROR
    assert failed.items == ()


def test_step_retrieval_max_num_results_passthrough():
    backend = FakeBackend(items=[item("evidence")])
    kb.retrieve_for_step(step(), backend=backend, max_num_results=2)
    assert backend.calls == [("test query A", 2)]


# ================================================================
# 7. GENERAL KNOWLEDGE (5B)
# ================================================================

def test_general_question_is_normalized_and_sent_to_same_backend():
    backend = FakeBackend(items=[item("evidence")])
    result = kb.retrieve_general(
        "  What   are the system requirements?  ", backend=backend
    )
    assert result.status is KnowledgeStatus.FOUND
    assert backend.calls == [
        ("What are the system requirements?", kb.DEFAULT_MAX_RESULTS)
    ]


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
def test_blank_general_question_is_not_found_without_backend_call(blank):
    backend = FakeBackend(items=[item("evidence")])
    result = kb.retrieve_general(blank, backend=backend)
    assert result.status is KnowledgeStatus.NOT_FOUND
    assert result.query == ""
    assert backend.calls == []


def test_general_semantics_identical_to_core():
    assert (
        kb.retrieve_general("q", backend=FakeBackend(items=[])).status
        is KnowledgeStatus.NOT_FOUND
    )
    err = kb.retrieve_general("q", backend=FakeBackend(exc=ValueError("x")))
    assert err.status is KnowledgeStatus.ERROR
    assert err.items == ()
    assert err.error is not None


def test_general_does_not_require_a_procedure():
    # No ProcedureStep / Procedure involved at all.
    backend = FakeBackend(items=[item("spec")])
    result = kb.retrieve_general(
        "device connection instructions", backend=backend
    )
    assert result.status is KnowledgeStatus.FOUND
    assert backend.calls == [
        ("device connection instructions", kb.DEFAULT_MAX_RESULTS)
    ]


# ================================================================
# 8. RUNTIME ISOLATION GUARD (5B)
# ================================================================

_RUNTIME_MODULES = (
    Path("yandex/chat/server.py"),
    Path("yandex/chat/alice_support_agent.py"),
    Path("yandex/realtime/web/server.py"),
)


def test_production_runtime_does_not_import_knowledge():
    """Static scan only (never imports these modules): the Knowledge layer
    must not be wired into any production runtime path yet."""
    root = Path(kb.__file__).resolve().parents[1]
    for rel in _RUNTIME_MODULES:
        path = root / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        assert "support.knowledge" not in text, f"{rel} imports support.knowledge"
        assert (
            "from support import knowledge" not in text
        ), f"{rel} imports support.knowledge"
