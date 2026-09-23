"""
Declarative Support Procedure: typed model + YAML loader + validator.

Infrastructure only. NOT imported by any production runtime path
(server.py, alice_support_agent.py, n8n, Tracker, DB, File Search).
No network, no LLM, no database: reads a YAML file and produces a
validated, typed object.

Domain split (design):
  KNOWLEDGE  = facts/evidence (Yandex File Search)   -> out of scope here
  PROCEDURE  = deterministic troubleshooting graph   -> this module
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml


# ============================================================
# CLOSED VOCABULARIES
# ============================================================
# Minimal infrastructure vocabulary for Slice 1. This is NOT the final
# AngioPlus business schema; it may be extended in later slices after the
# real procedures are designed.

TRANSITION_TYPES = frozenset(
    {
        "attempted_failed",
        "resolved",
        "user_requested_human",
    }
)

TERMINATORS = frozenset(
    {
        "ESCALATE",
        "RESOLVED",
    }
)


class ProcedureError(Exception):
    """Raised for any malformed / invalid procedure definition."""


# ============================================================
# TYPED REPRESENTATION
# ============================================================

@dataclass(frozen=True)
class ProcedureStep:
    """A single node of a support procedure."""

    id: str
    action: str
    evidence_query: Optional[str]
    transitions: Mapping[str, str]  # transition_type -> target (step id | terminator)


@dataclass(frozen=True)
class Procedure:
    """A fully validated, immutable support procedure."""

    procedure_id: str
    version: int
    issue: str
    entry_step: str
    steps: Mapping[str, ProcedureStep]
    terminators: frozenset[str]

    def step_ids(self) -> frozenset[str]:
        return frozenset(self.steps)


# ============================================================
# LOADING (YAML -> raw dict)
# ============================================================

def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Read YAML into a mapping; reject malformed YAML and duplicate keys.

    PyYAML silently keeps the LAST duplicate key by default; a typo could
    therefore shadow a real transition. We fail instead.
    """
    try:
        raw_text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ProcedureError(f"Cannot read procedure file: {exc}") from exc

    class _StrictLoader(yaml.SafeLoader):
        pass

    def _construct_mapping(loader, node, deep=False):
        loader.flatten_mapping(node)
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ProcedureError(
                    f"Duplicate YAML key {key!r} at line "
                    f"{key_node.start_mark.line + 1}"
                )
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    _StrictLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        _construct_mapping,
    )

    try:
        data = yaml.load(raw_text, Loader=_StrictLoader)
    except ProcedureError:
        raise
    except yaml.YAMLError as exc:
        raise ProcedureError(f"Malformed YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise ProcedureError(
            "Procedure root must be a YAML mapping, got "
            f"{type(data).__name__}."
        )
    return data


# ============================================================
# PARSING (raw dict -> typed, structural only)
# ============================================================

def _parse_transitions(step_id: str, raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict) or not raw:
        raise ProcedureError(
            f"Step {step_id!r}: 'transitions' must be a non-empty mapping."
        )
    parsed: dict[str, str] = {}
    for ttype, target in raw.items():
        if ttype not in TRANSITION_TYPES:
            raise ProcedureError(
                f"Step {step_id!r}: unknown transition type {ttype!r}."
            )
        if not isinstance(target, str) or not target.strip():
            raise ProcedureError(
                f"Step {step_id!r}: transition {ttype!r} target must be a "
                "non-empty string."
            )
        parsed[ttype] = target.strip()
    return parsed


def _parse_step(raw: Any, index: int) -> ProcedureStep:
    if not isinstance(raw, dict):
        raise ProcedureError(f"Step #{index} must be a mapping.")

    step_id = raw.get("id")
    if not isinstance(step_id, str) or not step_id.strip():
        raise ProcedureError(f"Step #{index}: missing or empty 'id'.")

    action = raw.get("action")
    if not isinstance(action, str) or not action.strip():
        raise ProcedureError(
            f"Step {step_id!r}: missing or empty 'action'/'instruction'."
        )

    evidence_query = raw.get("evidence_query")
    if evidence_query is not None and (
        not isinstance(evidence_query, str) or not evidence_query.strip()
    ):
        raise ProcedureError(
            f"Step {step_id!r}: 'evidence_query' must be a non-empty "
            "string when present."
        )

    transitions = _parse_transitions(step_id, raw.get("transitions"))

    return ProcedureStep(
        id=step_id.strip(),
        action=action.strip(),
        evidence_query=evidence_query.strip() if evidence_query else None,
        transitions=transitions,
    )


def _parse_procedure(data: Mapping[str, Any]) -> Procedure:
    procedure_id = data.get("procedure_id")
    if not isinstance(procedure_id, str) or not procedure_id.strip():
        raise ProcedureError("Missing or empty 'procedure_id'.")

    version = data.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ProcedureError("'version' must be an integer.")

    issue = data.get("issue")
    if not isinstance(issue, str) or not issue.strip():
        raise ProcedureError("Missing or empty 'issue'.")

    entry_step = data.get("entry_step")
    if not isinstance(entry_step, str) or not entry_step.strip():
        raise ProcedureError("Missing or empty 'entry_step'.")

    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ProcedureError("'steps' must be a non-empty list.")

    steps: dict[str, ProcedureStep] = {}
    for index, raw_step in enumerate(raw_steps):
        step = _parse_step(raw_step, index)
        if step.id in steps:
            raise ProcedureError(f"Duplicate step id {step.id!r}.")
        steps[step.id] = step

    raw_terminators = data.get("terminators", ["ESCALATE", "RESOLVED"])
    if not isinstance(raw_terminators, list) or not raw_terminators:
        raise ProcedureError("'terminators' must be a non-empty list.")
    terminators: set[str] = set()
    for term in raw_terminators:
        if term not in TERMINATORS:
            raise ProcedureError(f"Unknown terminator {term!r}.")
        terminators.add(term)

    return Procedure(
        procedure_id=procedure_id.strip(),
        version=version,
        issue=issue.strip(),
        entry_step=entry_step.strip(),
        steps=steps,
        terminators=frozenset(terminators),
    )


# ============================================================
# VALIDATION (invariants)
# ============================================================

def _reachable_steps(procedure: Procedure) -> set[str]:
    """Steps reachable from entry_step following step->step transitions."""
    step_ids = procedure.step_ids()
    reachable: set[str] = set()
    frontier = [procedure.entry_step]
    while frontier:
        current = frontier.pop()
        if current in reachable:
            continue
        reachable.add(current)
        for target in procedure.steps[current].transitions.values():
            if target in step_ids and target not in reachable:
                frontier.append(target)
    return reachable


def _can_reach_terminator(
    procedure: Procedure, start: str, reachable: set[str]
) -> bool:
    """True if `start` can reach a declared terminator via step->step edges."""
    seen: set[str] = set()
    frontier = [start]
    while frontier:
        current = frontier.pop()
        if current in seen:
            continue
        seen.add(current)
        for target in procedure.steps[current].transitions.values():
            if target in procedure.terminators:
                return True
            if target in reachable and target not in seen:
                frontier.append(target)
    return False


def _validate(procedure: Procedure) -> None:
    step_ids = procedure.step_ids()

    # entry_step must reference an existing step.
    if procedure.entry_step not in step_ids:
        raise ProcedureError(
            f"entry_step {procedure.entry_step!r} does not reference an "
            "existing step."
        )

    # Every transition target must be an existing step OR a declared
    # terminator.
    for step in procedure.steps.values():
        for ttype, target in step.transitions.items():
            if target in step_ids:
                continue
            if target in procedure.terminators:
                continue
            if target in TERMINATORS:
                raise ProcedureError(
                    f"Step {step.id!r}: transition {ttype!r} targets "
                    f"terminator {target!r} which is not declared by this "
                    "procedure."
                )
            raise ProcedureError(
                f"Step {step.id!r}: transition {ttype!r} targets unknown "
                f"identifier {target!r} (not a step or terminator)."
            )

    # Reachability: every step must be reachable from entry_step.
    reachable = _reachable_steps(procedure)
    unreachable = step_ids - reachable
    if unreachable:
        raise ProcedureError(
            f"Unreachable step(s) from entry_step "
            f"{procedure.entry_step!r}: {sorted(unreachable)}."
        )

    # Terminal reachability: every reachable step must have a path to some
    # declared terminator. Cycles are allowed ONLY if a terminal exit exists
    # from the cycle; closed regions with no terminal path are rejected.
    for step_id in sorted(reachable):
        if not _can_reach_terminator(procedure, step_id, reachable):
            raise ProcedureError(
                f"Step {step_id!r} cannot reach any declared terminator "
                f"{sorted(procedure.terminators)}; a closed region without a "
                "terminal path is not allowed."
            )


# ============================================================
# PUBLIC API
# ============================================================

def load_procedure_from_dict(data: Mapping[str, Any]) -> Procedure:
    """Parse + validate a procedure from an in-memory mapping."""
    procedure = _parse_procedure(data)
    _validate(procedure)
    return procedure


def load_procedure(path: str | Path) -> Procedure:
    """Load + parse + validate a procedure from a YAML file path."""
    data = _load_yaml_mapping(path)
    return load_procedure_from_dict(data)


__all__ = [
    "Procedure",
    "ProcedureStep",
    "ProcedureError",
    "TRANSITION_TYPES",
    "TERMINATORS",
    "load_procedure",
    "load_procedure_from_dict",
]
