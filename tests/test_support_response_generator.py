# -*- coding: utf-8 -*-
"""Contract tests for support.response_generator (Slice 6A -- PURE).

Slice 6A is provider-free and Knowledge-free: it deterministically renders
an ALREADY-MADE :class:`RoutingDecision` for EVERY
:class:`~support.router.RouteAction` without any LLM, network, DB or
production wiring.

This file pins the approved 6A contract:

  * ``ResponseStatus`` is a closed three-value enum;
  * ``ResponseResult`` is frozen and always carries a non-blank ``text``;
  * ``RECOMMEND_STEP`` renders EXACTLY the selected step's procedure-owned
    ``action`` and can never render a different step;
  * a missing / unresolvable step is ``INTERNAL_ERROR`` whose text stays
    NEUTRAL (no handoff / escalation / ticket / routing claim);
  * every ``ClarifyReason`` / ``EscalateReason`` has a deterministic
    template, and ONLY ``ESCALATE`` templates contain handoff wording;
  * ``RESOLVE`` / ``OUT_OF_SCOPE`` are deterministic;
  * identical input -> identical output; inputs are never mutated;
  * the external-contact directive guard protects procedure-owned text;
  * anti-regression guards: no forbidden imports, no state / observation /
    analyzer usage, no procedure-graph reading, no legacy cascade
    vocabulary, no wiring into a runtime path.

Knowledge- and provider-aware behavior belongs to LATER checkpoints.
"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from support import response_generator as rg
from support.procedure import load_procedure_from_dict
from support.response_generator import (
    ResponseResult,
    ResponseStatus,
    render_response,
)
from support.router import (
    ClarifyReason,
    EscalateReason,
    RouteAction,
    RoutingDecision,
)


# ----------------------------------------------------------------
# Helpers / fakes
# ----------------------------------------------------------------

def _procedure():
    """In-memory procedure (NO frozen fixture touched) with two steps
    whose actions are clearly distinct so a step mix-up is detectable."""
    return load_procedure_from_dict(
        {
            "procedure_id": "BLACK_SCREEN",
            "version": 1,
            "issue": "black screen",
            "entry_step": "STEP_A",
            "steps": [
                {
                    "id": "STEP_A",
                    "action": "Confirm the power LED is lit.",
                    "evidence_query": "angioplus black screen power led",
                    "transitions": {
                        "attempted_failed": "STEP_B",
                        "resolved": "RESOLVED",
                    },
                },
                {
                    "id": "STEP_B",
                    "action": "Reseat the display cable on both ends.",
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


def _recommend(step_id="STEP_A"):
    return RoutingDecision(
        action=RouteAction.RECOMMEND_STEP,
        next_step=step_id,
        transition_type=None,
    )


# Elements that may ONLY appear in ESCALATE rendering.
_HANDOFF_MARKERS = (
    "специалист",
    "переда",
    "поддержк",
    "оператор",
    "заявк",
    "переключ",
    "escalat",
    "handoff",
    "human",
    "ticket",
)


# ================================================================
# 1. STATUS VOCABULARY + RESULT INVARIANTS
# ================================================================

def test_status_is_closed_three_value_enum():
    assert {s.value for s in ResponseStatus} == {
        "rendered",
        "fallback",
        "internal_error",
    }


def test_result_is_frozen():
    res = ResponseResult(status=ResponseStatus.RENDERED, text="hello")
    with pytest.raises(FrozenInstanceError):
        res.text = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
def test_result_rejects_blank_text(blank):
    with pytest.raises(ValueError):
        ResponseResult(status=ResponseStatus.RENDERED, text=blank)


def test_result_rejects_non_str_text():
    with pytest.raises(ValueError):
        ResponseResult(status=ResponseStatus.RENDERED, text=123)  # type: ignore[arg-type]


def test_result_rejects_non_status():
    with pytest.raises(ValueError):
        ResponseResult(status="rendered", text="x")  # type: ignore[arg-type]


# ================================================================
# 2. EVERY ROUTE ACTION PRODUCES A NON-BLANK RESPONSE
# ================================================================

@pytest.mark.parametrize("action", list(RouteAction))
def test_every_route_action_renders_non_blank_text(action):
    proc = _procedure()
    if action is RouteAction.RECOMMEND_STEP:
        decision = _recommend("STEP_A")
    elif action is RouteAction.CLARIFY:
        decision = RoutingDecision(
            action=action, clarify_reason=ClarifyReason.PENDING_CLARIFICATION
        )
    elif action is RouteAction.ESCALATE:
        decision = RoutingDecision(
            action=action, escalate_reason=EscalateReason.PROCEDURE_TERMINAL
        )
    else:
        decision = RoutingDecision(action=action)

    res = render_response(decision, procedure=proc)
    assert isinstance(res, ResponseResult)
    assert res.status in set(ResponseStatus)
    assert isinstance(res.text, str) and res.text.strip()


# ================================================================
# 3. RECOMMEND_STEP -- exactly the selected step
# ================================================================

def test_recommend_step_renders_selected_step_action():
    proc = _procedure()
    res = render_response(_recommend("STEP_A"), procedure=proc)
    assert res.status is ResponseStatus.RENDERED
    assert res.text == "Confirm the power LED is lit."
    assert res.step_id == "STEP_A"


def test_recommend_step_cannot_render_another_step():
    proc = _procedure()
    res = render_response(_recommend("STEP_A"), procedure=proc)
    assert "Reseat the display cable" not in res.text
    assert res.text == proc.steps["STEP_A"].action


def test_recommend_step_whitespace_is_normalized():
    proc = load_procedure_from_dict(
        {
            "procedure_id": "P",
            "version": 1,
            "issue": "x",
            "entry_step": "S",
            "steps": [
                {
                    "id": "S",
                    "action": "  Restart   the   software.  ",
                    "transitions": {"resolved": "RESOLVED"},
                }
            ],
            "terminators": ["RESOLVED"],
        }
    )
    res = render_response(_recommend("S"), procedure=proc)
    assert res.text == "Restart the software."


def test_recommend_step_does_not_read_graph_edges():
    # The output depends ONLY on the selected step's action, never on the
    # step's declared outgoing edges.
    proc = load_procedure_from_dict(
        {
            "procedure_id": "P",
            "version": 1,
            "issue": "x",
            "entry_step": "S",
            "steps": [
                {
                    "id": "S",
                    "action": "Do the thing.",
                    "transitions": {"attempted_failed": "ESCALATE"},
                }
            ],
            "terminators": ["ESCALATE"],
        }
    )
    res = render_response(_recommend("S"), procedure=proc)
    assert res.text == "Do the thing."
    assert res.status is ResponseStatus.RENDERED


# ================================================================
# 4. INTERNAL_ERROR -- missing / unresolvable step (NEUTRAL text)
# ================================================================

def test_missing_next_step_is_internal_error():
    proc = _procedure()
    decision = RoutingDecision(action=RouteAction.RECOMMEND_STEP, next_step=None)
    res = render_response(decision, procedure=proc)
    assert res.status is ResponseStatus.INTERNAL_ERROR
    assert res.text.strip()
    assert res.step_id is None


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_next_step_is_internal_error(blank):
    proc = _procedure()
    decision = RoutingDecision(action=RouteAction.RECOMMEND_STEP, next_step=blank)
    res = render_response(decision, procedure=proc)
    assert res.status is ResponseStatus.INTERNAL_ERROR


def test_unknown_next_step_is_internal_error():
    proc = _procedure()
    res = render_response(_recommend("NOPE"), procedure=proc)
    assert res.status is ResponseStatus.INTERNAL_ERROR
    assert res.step_id is None


def test_missing_procedure_is_internal_error():
    res = render_response(_recommend("STEP_A"), procedure=None)
    assert res.status is ResponseStatus.INTERNAL_ERROR


def test_internal_error_text_is_neutral_no_handoff_claim():
    """INTERNAL_ERROR is NOT an escalation decision: its text must not
    claim (or imply) a handoff, escalation, ticket or routing outcome."""
    proc = _procedure()
    for decision in (
        RoutingDecision(action=RouteAction.RECOMMEND_STEP, next_step=None),
        _recommend("NOPE"),
    ):
        res = render_response(decision, procedure=proc)
        assert res.status is ResponseStatus.INTERNAL_ERROR
        low = res.text.lower()
        for marker in _HANDOFF_MARKERS:
            assert marker not in low, (
                f"INTERNAL_ERROR text claims a handoff via {marker!r}: "
                f"{res.text!r}"
            )
        # and it must not direct the user to an external contact either
        assert not rg._is_external_contact_directive(res.text)
        # and it must not be ESCALATE wording
        assert res.text not in set(rg._ESCALATE_TEMPLATES.values())

    assert "специалист" not in rg._INTERNAL_ERROR_TEXT.lower()
    assert "переда" not in rg._INTERNAL_ERROR_TEXT.lower()
    assert "поддержк" not in rg._INTERNAL_ERROR_TEXT.lower()


# ================================================================
# 5. TEMPLATE COVERAGE + HANDOFF EXCLUSIVITY
# ================================================================

def test_every_clarify_reason_has_template():
    assert set(rg._CLARIFY_TEMPLATES) == set(ClarifyReason)


def test_every_escalate_reason_has_template():
    assert set(rg._ESCALATE_TEMPLATES) == set(EscalateReason)


def test_clarify_renders_reason_template():
    for reason in ClarifyReason:
        decision = RoutingDecision(
            action=RouteAction.CLARIFY, clarify_reason=reason
        )
        res = render_response(decision)
        assert res.status is ResponseStatus.RENDERED
        assert res.text == rg._CLARIFY_TEMPLATES[reason]


def test_escalate_renders_reason_template():
    for reason in EscalateReason:
        decision = RoutingDecision(
            action=RouteAction.ESCALATE, escalate_reason=reason
        )
        res = render_response(decision)
        assert res.status is ResponseStatus.RENDERED
        assert res.text == rg._ESCALATE_TEMPLATES[reason]


def test_only_escalate_templates_contain_handoff_wording():
    non_escalate_texts = (
        list(rg._CLARIFY_TEMPLATES.values())
        + [rg._CLARIFY_DEFAULT_TEXT]
        + [rg._RESOLVE_TEXT]
        + [rg._OUT_OF_SCOPE_TEXT]
        + [rg._INTERNAL_ERROR_TEXT]
        + [rg._UNSAFE_STEP_TEXT]
    )
    for text in non_escalate_texts:
        low = text.lower()
        for marker in _HANDOFF_MARKERS:
            assert marker not in low, (
                f"non-ESCALATE text claims a handoff via {marker!r}: "
                f"{text!r}"
            )

    # And each ESCALATE template DOES carry handoff wording.
    for text in rg._ESCALATE_TEMPLATES.values():
        low = text.lower()
        assert any(marker in low for marker in _HANDOFF_MARKERS), text


def test_clarify_without_reason_uses_safe_default():
    decision = RoutingDecision(action=RouteAction.CLARIFY, clarify_reason=None)
    res = render_response(decision)
    assert res.status is ResponseStatus.RENDERED
    assert res.text == rg._CLARIFY_DEFAULT_TEXT


def test_escalate_without_reason_uses_safe_default():
    decision = RoutingDecision(action=RouteAction.ESCALATE, escalate_reason=None)
    res = render_response(decision)
    assert res.status is ResponseStatus.RENDERED
    assert res.text == rg._ESCALATE_DEFAULT_TEXT


# ================================================================
# 6. RESOLVE / OUT_OF_SCOPE -- deterministic
# ================================================================

def test_resolve_is_deterministic():
    first = render_response(RoutingDecision(action=RouteAction.RESOLVE))
    second = render_response(RoutingDecision(action=RouteAction.RESOLVE))
    assert first == second
    assert first.status is ResponseStatus.RENDERED
    assert first.text == rg._RESOLVE_TEXT


def test_out_of_scope_is_deterministic():
    first = render_response(RoutingDecision(action=RouteAction.OUT_OF_SCOPE))
    second = render_response(RoutingDecision(action=RouteAction.OUT_OF_SCOPE))
    assert first == second
    assert first.status is ResponseStatus.RENDERED
    assert first.text == rg._OUT_OF_SCOPE_TEXT
    # no handoff wording in OUT_OF_SCOPE
    low = first.text.lower()
    for marker in _HANDOFF_MARKERS:
        assert marker not in low


# ================================================================
# 7. PURITY -- identical input / no mutation
# ================================================================

def test_repeated_identical_input_is_identical_output():
    proc = _procedure()
    decision = _recommend("STEP_A")
    assert render_response(decision, procedure=proc) == render_response(
        decision, procedure=proc
    )


def test_input_decision_and_procedure_are_not_mutated():
    proc = _procedure()
    decision = _recommend("STEP_A")
    before_step_a = proc.steps["STEP_A"].action
    render_response(decision, procedure=proc)
    # decision is frozen; procedure untouched.
    assert decision.next_step == "STEP_A"
    assert decision.action is RouteAction.RECOMMEND_STEP
    assert proc.steps["STEP_A"].action == before_step_a
    assert set(proc.steps) == {"STEP_A", "STEP_B"}


# ================================================================
# 8. EXTERNAL-CONTACT DIRECTIVE GUARD
# ================================================================

def _procedure_with_action(action: str):
    return load_procedure_from_dict(
        {
            "procedure_id": "P",
            "version": 1,
            "issue": "x",
            "entry_step": "S",
            "steps": [
                {
                    "id": "S",
                    "action": action,
                    "transitions": {"resolved": "RESOLVED"},
                }
            ],
            "terminators": ["RESOLVED"],
        }
    )


@pytest.mark.parametrize(
    "action",
    [
        "Обратитесь к производителю за заменой.",
        "Свяжитесь с дистрибьютором для получения детали.",
        "Contact the manufacturer for a replacement.",
        "Please reach out to the supplier.",
    ],
)
def test_external_contact_directive_in_action_is_not_rendered(action):
    proc = _procedure_with_action(action)
    res = render_response(_recommend("S"), procedure=proc)
    assert res.status is ResponseStatus.FALLBACK
    assert res.text == rg._UNSAFE_STEP_TEXT
    assert "производител" not in res.text.lower()
    assert "manufacturer" not in res.text.lower()
    assert "Обратитесь" not in res.text


@pytest.mark.parametrize(
    "action",
    [
        "В инструкции производителя указано минимальное напряжение.",
        "According to the manufacturer documentation, use the supplied cable.",
        "The distributor ships the unit with a calibrated sensor.",
    ],
)
def test_factual_manufacturer_mention_is_allowed(action):
    proc = _procedure_with_action(action)
    res = render_response(_recommend("S"), procedure=proc)
    assert res.status is ResponseStatus.RENDERED
    assert res.text == action


def test_guard_blocks_directive_but_not_bare_mention():
    assert rg._is_external_contact_directive("Обратитесь к производителю.")
    assert rg._is_external_contact_directive("Contact the supplier.")
    assert not rg._is_external_contact_directive(
        "According to the manufacturer, this is expected."
    )
    assert not rg._is_external_contact_directive(
        "Обратитесь в службу поддержки."
    )


# ================================================================
# 9. ANTI-REGRESSION GUARDS
# ================================================================

def _module_path() -> Path:
    return Path(rg.__file__)


def _module_source() -> str:
    return _module_path().read_text(encoding="utf-8")


def _module_imports() -> set[str]:
    tree = ast.parse(_module_source())
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
        "support.procedure_state",
        "support.observation",
        "support.alice_analyzer",
        "support.knowledge",
        "support.knowledge_yandex",
        "support.db",
        "support.tracker_service",
    }
    for name in _module_imports():
        root = name.split(".")[0]
        assert root not in forbidden_roots, f"forbidden import: {name}"
        for mod in forbidden_modules:
            assert name != mod and not name.startswith(mod + "."), (
                f"forbidden import: {name}"
            )


def test_only_allowed_imports():
    allowed = {"__future__", "dataclasses", "enum", "re", "typing", "support"}
    roots = {name.split(".")[0] for name in _module_imports()}
    assert roots <= allowed, f"unexpected imports: {roots - allowed}"


def test_only_router_and_procedure_are_imported_from_support():
    support_mods = {
        n
        for n in _module_imports()
        if n == "support" or n.startswith("support.")
    }
    assert support_mods <= {"support.router", "support.procedure"}, support_mods


def test_no_state_or_observation_or_analyzer_usage():
    source = _module_source()
    for token in (
        "ProcedureState",
        "SemanticObservation",
        "alice_analyzer",
        "analyze(",
        "ObservationScope",
        "ProblemStatus",
    ):
        assert token not in source, f"forbidden token present: {token}"


def test_generator_never_reads_procedure_graph_edges():
    source = _module_source()
    # Neither the attribute name nor the concept may appear.
    assert "transitions" not in source
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr != "transitions", "reads procedure graph edges"


def test_generator_never_calls_the_router():
    imported_names: set[str] = set()
    tree = ast.parse(_module_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.name)
    assert "route" not in imported_names, "must not import the router entry point"


def test_no_legacy_cascade_vocabulary():
    source = _module_source()
    for token in (
        "candidate_l1_action",
        "candidate_action_already_attempted",
        "apply_support_policy",
        "live_text_validation",
    ):
        assert token not in source, f"forbidden token present: {token}"


def test_not_wired_into_package_init():
    init = (_module_path().parent / "__init__.py").read_text(encoding="utf-8")
    assert "response_generator" not in init


_RUNTIME_MODULES = (
    Path("yandex/chat/server.py"),
    Path("yandex/chat/alice_support_agent.py"),
    Path("yandex/realtime/web/server.py"),
)


def test_no_runtime_module_imports_response_generator():
    """Static scan only (never imports these modules): the generator must
    not be wired into any production runtime path yet."""
    root = Path(rg.__file__).resolve().parents[1]
    for rel in _RUNTIME_MODULES:
        path = root / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        assert "response_generator" not in text, f"{rel} references the generator"


def test_public_api_surface():
    assert set(rg.__all__) == {
        "ResponseStatus",
        "ResponseResult",
        "render_response",
    }
