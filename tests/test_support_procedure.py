"""Support Procedure infrastructure tests (pure: no DB, no network, no LLM)."""

from __future__ import annotations

from pathlib import Path

import pytest

from support.procedure import (
    Procedure,
    ProcedureError,
    load_procedure,
    load_procedure_from_dict,
)


FIXTURE = (
    Path(__file__).parent / "fixtures" / "procedures" / "test_procedure.yaml"
)


def _valid_dict() -> dict:
    return {
        "procedure_id": "TEST_PROCEDURE",
        "version": 1,
        "issue": "synthetic test issue",
        "entry_step": "STEP_A",
        "steps": [
            {
                "id": "STEP_A",
                "action": "Generic test action for STEP_A.",
                "evidence_query": "test query A",
                "transitions": {
                    "attempted_failed": "STEP_B",
                    "resolved": "RESOLVED",
                },
            },
            {
                "id": "STEP_B",
                "action": "Generic test action for STEP_B.",
                "transitions": {
                    "attempted_failed": "ESCALATE",
                    "resolved": "RESOLVED",
                },
            },
        ],
        "terminators": ["ESCALATE", "RESOLVED"],
    }


# ----------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------

def test_valid_procedure_loads_from_fixture():
    procedure = load_procedure(FIXTURE)
    assert isinstance(procedure, Procedure)
    assert procedure.procedure_id == "TEST_PROCEDURE"
    assert procedure.version == 1
    assert procedure.entry_step == "STEP_A"
    assert procedure.step_ids() == {"STEP_A", "STEP_B"}
    assert procedure.terminators == {"ESCALATE", "RESOLVED"}


def test_valid_procedure_loads_from_dict():
    procedure = load_procedure_from_dict(_valid_dict())
    assert procedure.step_ids() == {"STEP_A", "STEP_B"}
    assert procedure.steps["STEP_A"].transitions["attempted_failed"] == "STEP_B"


def test_valid_terminal_transitions_accepted():
    procedure = load_procedure_from_dict(_valid_dict())
    assert procedure.steps["STEP_B"].transitions["attempted_failed"] == "ESCALATE"
    assert procedure.steps["STEP_A"].transitions["resolved"] == "RESOLVED"


def test_cycle_with_terminal_exit_accepted():
    # STEP_A <-> STEP_B cycle is allowed because STEP_B exits to a terminator.
    data = _valid_dict()
    data["steps"] = [
        {
            "id": "STEP_A",
            "action": "A",
            "transitions": {
                "attempted_failed": "STEP_B",
                "resolved": "RESOLVED",
            },
        },
        {
            "id": "STEP_B",
            "action": "B",
            "transitions": {
                "attempted_failed": "STEP_A",
                "resolved": "RESOLVED",
                "user_requested_human": "ESCALATE",
            },
        },
    ]
    procedure = load_procedure_from_dict(data)
    assert procedure.step_ids() == {"STEP_A", "STEP_B"}


# ----------------------------------------------------------------
# Invalid definitions
# ----------------------------------------------------------------

def test_invalid_entry_step_rejected():
    data = _valid_dict()
    data["entry_step"] = "NOPE"
    with pytest.raises(ProcedureError):
        load_procedure_from_dict(data)


def test_dangling_transition_rejected():
    data = _valid_dict()
    data["steps"][0]["transitions"]["attempted_failed"] = "MISSING_STEP"
    with pytest.raises(ProcedureError):
        load_procedure_from_dict(data)


def test_unknown_transition_type_rejected():
    data = _valid_dict()
    data["steps"][0]["transitions"]["made_coffee"] = "STEP_B"
    with pytest.raises(ProcedureError):
        load_procedure_from_dict(data)


def test_unknown_terminator_rejected():
    data = _valid_dict()
    data["terminators"] = ["ESCALATE", "RESOLVED", "TOTALLY_MADE_UP"]
    with pytest.raises(ProcedureError):
        load_procedure_from_dict(data)


def test_undeclared_terminator_target_rejected():
    # Target is a valid vocabulary terminator but not declared by THIS
    # procedure (only ESCALATE declared).
    data = _valid_dict()
    data["terminators"] = ["ESCALATE"]
    data["steps"][0]["transitions"]["resolved"] = "RESOLVED"
    with pytest.raises(ProcedureError):
        load_procedure_from_dict(data)


def test_duplicate_step_rejected():
    data = _valid_dict()
    data["steps"].append(
        {
            "id": "STEP_A",
            "action": "duplicate",
            "transitions": {
                "attempted_failed": "ESCALATE",
                "resolved": "RESOLVED",
            },
        }
    )
    with pytest.raises(ProcedureError):
        load_procedure_from_dict(data)


def test_unreachable_step_rejected():
    data = _valid_dict()
    data["steps"].append(
        {
            "id": "ORPHAN",
            "action": "never reachable",
            "transitions": {
                "attempted_failed": "ESCALATE",
                "resolved": "RESOLVED",
            },
        }
    )
    with pytest.raises(ProcedureError):
        load_procedure_from_dict(data)


def test_closed_cycle_without_terminal_path_rejected():
    # STEP_A <-> STEP_B with NO path to ESCALATE/RESOLVED must be rejected.
    data = _valid_dict()
    data["steps"] = [
        {
            "id": "STEP_A",
            "action": "A",
            "transitions": {
                "attempted_failed": "STEP_B",
                "resolved": "STEP_B",
            },
        },
        {
            "id": "STEP_B",
            "action": "B",
            "transitions": {
                "attempted_failed": "STEP_A",
                "resolved": "STEP_A",
            },
        },
    ]
    with pytest.raises(ProcedureError):
        load_procedure_from_dict(data)


def test_empty_action_rejected():
    data = _valid_dict()
    data["steps"][0]["action"] = "   "
    with pytest.raises(ProcedureError):
        load_procedure_from_dict(data)


def test_empty_transitions_rejected():
    data = _valid_dict()
    data["steps"][0]["transitions"] = {}
    with pytest.raises(ProcedureError):
        load_procedure_from_dict(data)


# ----------------------------------------------------------------
# YAML-level failures
# ----------------------------------------------------------------

def test_malformed_yaml_rejected(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("procedure_id: [unterminated\n", encoding="utf-8")
    with pytest.raises(ProcedureError):
        load_procedure(bad)


def test_duplicate_yaml_key_rejected(tmp_path: Path):
    bad = tmp_path / "dup.yaml"
    bad.write_text(
        "procedure_id: TEST_PROCEDURE\n"
        "procedure_id: SHADOWED\n"
        "version: 1\n"
        "issue: x\n"
        "entry_step: STEP_A\n"
        "steps: []\n",
        encoding="utf-8",
    )
    with pytest.raises(ProcedureError):
        load_procedure(bad)
