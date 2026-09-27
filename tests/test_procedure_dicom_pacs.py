# -*- coding: utf-8 -*-
"""Slice 10 -- the first PRODUCTION procedure: DICOM / PACS connectivity.

This file pins the IFU-grounded declarative graph
(``support/procedures/dicom_pacs_connectivity_v1.yaml``) and proves it is
executable by the FROZEN deterministic stack (router + response generator)
WITHOUT any LLM, network or DB.

AUTHORING PROVENANCE (see also the YAML header comment):
  * AngioPlus Core v2.5 IFU (RU), §12.3.4 "PACS" (pp. 18-20):
      DICOM Connect on the PStation toolbar; AE title + port (defaults
      "μFR" / 1108); register the PStation storage SCP with the AE title
      and the PACS port as a DICOM destination node; add + test the
      registered DICOM node.
  * AngioPlus Core v2.5 IFU (RU), §20 "Поиск и устранение
    неисправностей" (p. 44): restart the software on an error / crash /
    problem.
  * Deliberately NOT copied: the external-contact clause (Shanghai Pulse
    Medical Technology, Inc.) -- expressed as the ESCALATE terminator;
    the "IP-адрес удаленного узла" field and OS-port/firewall advice
    (§8); the "Q&A List ENG.xlsx" (binary, no extracted text).

PURE: no DB, no cloud, no LLM.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from support import procedure_catalog as pc
from support.observation import (
    ObservationScope,
    RequestKind,
    SemanticObservation,
    StepOutcome,
)
from support.procedure import Procedure, load_procedure
from support.procedure_catalog import ProcedureCatalog
from support.procedure_state import ProcedureState, StepAttempt
from support.response_generator import (
    ResponseStatus,
    _is_external_contact_directive,
    render_response,
)
from support.router import (
    EscalateReason,
    RouteAction,
    RoutingDecision,
    route,
)


_YAML = Path(pc.__file__).parent / "procedures" / "dicom_pacs_connectivity_v1.yaml"

_STEP_IDS = (
    "CONFIGURE_DICOM_CONNECT",
    "REGISTER_DESTINATION_NODE",
    "TEST_DICOM_NODE",
    "RESTART_SOFTWARE",
)

_CHAIN = {
    "CONFIGURE_DICOM_CONNECT": "REGISTER_DESTINATION_NODE",
    "REGISTER_DESTINATION_NODE": "TEST_DICOM_NODE",
    "TEST_DICOM_NODE": "RESTART_SOFTWARE",
    "RESTART_SOFTWARE": "ESCALATE",
}


@pytest.fixture(scope="module")
def procedure() -> Procedure:
    return load_procedure(_YAML)


def _obs() -> SemanticObservation:
    """A neutral in-scope reading with no facts (router walks the graph)."""
    return SemanticObservation(
        scope=ObservationScope.IN_SCOPE,
        reported_actions=(),
        problem_status=None,
        human_requested=False,
        clarification_facts=(),
        new_issue=None,
        request_kind=RequestKind.PROBLEM,
    )


def _state(
    procedure: Procedure,
    attempts: tuple[StepAttempt, ...],
    last_recommended: str | None,
) -> ProcedureState:
    return ProcedureState(
        issue=procedure.issue,
        procedure_id=procedure.procedure_id,
        procedure_version=procedure.version,
        last_recommended_step=last_recommended,
        step_attempts=attempts,
    )


def _failed(step_id: str, seq: int) -> StepAttempt:
    return StepAttempt(step_id=step_id, outcome=StepOutcome.FAILED, sequence=seq)


# ================================================================
# 1. IDENTITY + DECLARATIVE SHAPE
# ================================================================

def test_identity_and_entry_step(procedure):
    assert procedure.procedure_id == "DICOM_PACS_CONNECTIVITY"
    assert procedure.version == 1
    assert procedure.entry_step == "CONFIGURE_DICOM_CONNECT"


def test_step_set_is_exactly_the_four_ifu_steps(procedure):
    assert procedure.step_ids() == set(_STEP_IDS)


def test_attempted_failed_chain_follows_the_ifu_order(procedure):
    for step_id, target in _CHAIN.items():
        assert procedure.steps[step_id].transitions["attempted_failed"] == target


def test_every_step_resolves_to_the_terminator(procedure):
    for step_id in _STEP_IDS:
        assert procedure.steps[step_id].transitions["resolved"] == "RESOLVED"


def test_terminators(procedure):
    assert procedure.terminators == {"ESCALATE", "RESOLVED"}


def test_issue_is_framed_as_configuration_connectivity(procedure):
    issue = procedure.issue.lower()
    assert "dicom" in issue and "pacs" in issue
    assert "соединени" in issue


# ================================================================
# 2. IFU-GROUNDED PROVENANCE (evidence_query quotes IFU literals)
# ================================================================

@pytest.mark.parametrize(
    "step_id, tokens",
    [
        (
            "CONFIGURE_DICOM_CONNECT",
            ("dicom", "connect", "pstation", "заголовок", "номера", "порта"),
        ),
        (
            "REGISTER_DESTINATION_NODE",
            ("узел", "назначения", "заголовок", "pstation", "scp"),
        ),
        ("TEST_DICOM_NODE", ("узел", "тестирования", "dicom")),
        ("RESTART_SOFTWARE", ("перезапустите", "проверка", "соединения")),
    ],
)
def test_step_evidence_query_quotes_ifu_literal_tokens(procedure, step_id, tokens):
    query = (procedure.steps[step_id].evidence_query or "").lower()
    for token in tokens:
        assert token in query, (step_id, token)


def test_step_actions_follow_the_ifu_wording(procedure):
    assert "dicom connect" in procedure.steps["CONFIGURE_DICOM_CONNECT"].action.lower()
    assert "1108" in procedure.steps["CONFIGURE_DICOM_CONNECT"].action
    assert "scp" in procedure.steps["REGISTER_DESTINATION_NODE"].action.lower()
    assert "тестирования" in procedure.steps["TEST_DICOM_NODE"].action.lower()
    assert "перезапустите" in procedure.steps["RESTART_SOFTWARE"].action.lower()


def test_yaml_header_documents_model_and_ifu_provenance():
    header = _YAML.read_text(encoding="utf-8")
    assert 'COLD-START SELECTION MODEL = "mvp_lexical_bootstrap"' in header
    assert "IFU" in header
    assert "12.3.4" in header
    assert "ESCALATE" in header  # contact clause -> declarative terminator


# ================================================================
# 3. ANTI-INVENTION GUARDS
# ================================================================

def test_no_invented_or_excluded_step_is_present(procedure):
    blob = " ".join(
        f"{step.id} {step.action} {step.evidence_query or ''}"
        for step in procedure.steps.values()
    ).lower()
    # No invented transfer-status step.
    assert "check_dicom_transfer" not in blob
    # No OS/IT-level advice (IP address / firewall / OS port opening).
    for forbidden in ("ip-адрес", "ip address", "firewall", "брандмауэр", "порт 104"):
        assert forbidden not in blob, forbidden
    # No external-contact copy (it must be ESCALATE, not a step action).
    for forbidden in ("shanghai pulse", "medical technology", "производител"):
        assert forbidden not in blob, forbidden


def test_no_step_action_is_an_external_contact_directive(procedure):
    for step in procedure.steps.values():
        assert not _is_external_contact_directive(step.action), step.id


# ================================================================
# 4. END-TO-END EXECUTION BY THE FROZEN STACK (no LLM / DB)
# ================================================================

def test_first_turn_recommends_the_entry_step(procedure):
    decision = route(
        _state(procedure, (), None), _obs(), procedure
    )
    assert decision.action is RouteAction.RECOMMEND_STEP
    assert decision.next_step == "CONFIGURE_DICOM_CONNECT"
    assert decision.transition_type is None


@pytest.mark.parametrize(
    "index, next_step",
    [
        (0, "REGISTER_DESTINATION_NODE"),
        (1, "TEST_DICOM_NODE"),
        (2, "RESTART_SOFTWARE"),
    ],
)
def test_a_failed_step_advances_via_attempted_failed(procedure, index, next_step):
    attempts = tuple(_failed(_STEP_IDS[i], i) for i in range(index + 1))
    decision = route(
        _state(procedure, attempts, _STEP_IDS[index]), _obs(), procedure
    )
    assert decision.action is RouteAction.RECOMMEND_STEP
    assert decision.next_step == next_step
    assert decision.transition_type == "attempted_failed"


def test_all_steps_failed_escalates_at_the_terminator(procedure):
    attempts = tuple(_failed(step_id, i) for i, step_id in enumerate(_STEP_IDS))
    decision = route(
        _state(procedure, attempts, _STEP_IDS[-1]), _obs(), procedure
    )
    assert decision.action is RouteAction.ESCALATE
    assert decision.escalate_reason is EscalateReason.PROCEDURE_TERMINAL


def test_a_successful_step_resolves(procedure):
    attempts = (
        StepAttempt(
            step_id="CONFIGURE_DICOM_CONNECT",
            outcome=StepOutcome.SUCCEEDED,
            sequence=0,
        ),
    )
    decision = route(
        _state(procedure, attempts, "CONFIGURE_DICOM_CONNECT"), _obs(), procedure
    )
    assert decision.action is RouteAction.RESOLVE


@pytest.mark.parametrize("step_id", _STEP_IDS)
def test_every_step_action_renders_verbatim(procedure, step_id):
    decision = RoutingDecision(
        action=RouteAction.RECOMMEND_STEP, next_step=step_id
    )
    result = render_response(decision, procedure=procedure)
    assert result.status is ResponseStatus.RENDERED
    assert result.step_id == step_id
    assert result.text == " ".join(procedure.steps[step_id].action.split())


def test_the_previously_unsafe_restart_step_now_renders(procedure):
    """The IFU contact clause was removed from the restart step: it renders
    cleanly instead of falling back on the external-contact guard."""
    decision = RoutingDecision(
        action=RouteAction.RECOMMEND_STEP, next_step="RESTART_SOFTWARE"
    )
    result = render_response(decision, procedure=procedure)
    assert result.status is ResponseStatus.RENDERED
    assert "Перезапустите" in result.text


def test_escalate_terminator_renders_handoff(procedure):
    """The IFU contact clause lives on as the ESCALATE terminator text."""
    result = render_response(
        RoutingDecision(
            action=RouteAction.ESCALATE,
            escalate_reason=EscalateReason.PROCEDURE_TERMINAL,
        )
    )
    assert result.status is ResponseStatus.RENDERED
    assert "специалист" in result.text.lower()


# ================================================================
# 5. CATALOG INTEGRATION (identity from a cold-start signal)
# ================================================================

def test_catalog_selects_this_production_procedure(procedure):
    selection = ProcedureCatalog.from_directory().select(
        "не передаются данные DICOM на PACS при съеме"
    )
    assert selection is not None
    assert selection.procedure_id == procedure.procedure_id
    assert selection.version == procedure.version
    assert selection.procedure == procedure
