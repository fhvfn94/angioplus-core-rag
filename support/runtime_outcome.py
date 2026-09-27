"""
Runtime Outcome -- the PURE runtime-facing contract of ONE support turn
(Slice HL-2a).

Infrastructure only. NOT imported by any production runtime path yet
(server.py, alice_support_agent.py, n8n, Tracker). It is a standalone
projection, not wired anywhere.

PURPOSE
-------
Turn the ALREADY-MADE application result of ONE support turn

    SupportTurnOutcome  (support.support_core)
  + ConversationState   (support.domain)
        |
        v
    RuntimeTurn  (this module)

into the SINGLE immutable runtime value a transport layer consumes: the
user-facing ``text``, the channel ``decision`` / ``reason``, the
``case_action`` and the ONE FSM ``fsm_event`` (or ``None``).

SINGLE DECISION OWNER
---------------------
This module DECIDES NOTHING. It delegates ``decision`` / ``reason`` /
``fsm_event`` / ``case_action`` to
:func:`support.lifecycle_projection.project_turn_outcome` (the ONE owner of
that translation) and preserves its result VERBATIM. It adds ONLY:

  * the user-facing ``text`` extraction from the outcome's own native
    payload (never rendered again, never produced by an LLM); and
  * the typed application distinction between a normal ``USER_RESPONSE`` and
    an ``INTERNAL_ERROR`` (an internal-consistency failure that must never be
    mistaken for a semantic decision).

It never reads the raw message or a
:class:`~support.observation.SemanticObservation`, never walks procedure
transitions, and runs no router, no reducer and no LLM.

WHAT IT DELIBERATELY DOES NOT DO (later HL slices own these)
-----------------------------------------------------------
No DB / session work, no message persistence, no idempotency, no FSM
execution, no HTTP serialization, no replay / crash semantics, no server.py
wiring, no n8n / Tracker, no production composition root. It is a pure,
total projection over typed inputs, unit-testable in isolation.

This module carries NO ``retrieval_used`` flag: it is a legacy HTTP
compatibility field that ``SupportTurnOutcome`` does not expose truthfully.
Preserving it is an HL-2c decision, so HL-2a neither hardcodes, infers nor
inspects it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol

from support.clarification import clarify
from support.domain import ConversationState, SupportEvent
from support.lifecycle_projection import (
    CaseAction,
    SupportTurnOutcomeLike,
    project_turn_outcome,
)

# The neutral, deterministic INTERNAL_ERROR wording is REUSED from the single
# wording owner (``support.response_generator``) rather than duplicated: it
# makes NO success, handoff, escalation, ticket or Tracker claim.
from support.response_generator import _INTERNAL_ERROR_TEXT


# ============================================================
# CONSUMED CONTRACT (structural + pinned status literals)
# ============================================================

# The EXISTING boundary status vocabulary owned by ``support.support_core``.
# Pinned here (rather than imported) because the repository architecture guard
# ``test_no_runtime_module_imports_support_core_yet`` forbids ANY sibling module
# from importing the composition boundary. The focused tests build outcomes
# from the REAL ``support_core`` constants, so any drift fails loudly.
_TURN_EXECUTED = "TURN_EXECUTED"
_KNOWLEDGE_QA = "KNOWLEDGE_QA"
_CLARIFICATION = "CLARIFICATION"
_ESCALATE = "ESCALATE"
_OUT_OF_SCOPE = "OUT_OF_SCOPE"
_NO_PROCEDURE = "NO_PROCEDURE"
_PROCEDURE_NOT_FOUND = "PROCEDURE_NOT_FOUND"

# The ONE reason an INTERNAL_ERROR carries (matches the projection owner).
_REASON_PROCEDURE_NOT_FOUND = "PROCEDURE_NOT_FOUND"


class RuntimeTurnOutcomeLike(SupportTurnOutcomeLike, Protocol):
    """Structural view of the outcome fields this projection consumes.

    The real producer is ``support.support_core.SupportTurnOutcome``; it
    satisfies this protocol. Declaring it structurally keeps this PURE module
    free of any dependency on the composition boundary while still
    documenting the exact contract it reads.
    """

    conversation_id: str
    text: Optional[str]
    knowledge: Optional[object]
    clarification: Optional[object]


# ============================================================
# CLOSED VOCABULARY
# ============================================================

class RuntimeTurnKind(str, Enum):
    """The minimum typed application distinction for ONE runtime turn.

    * ``USER_RESPONSE`` -- a normal turn carrying a real channel decision
      (``ANSWER`` / ``CLARIFY`` / ``ESCALATE`` / ``OUT_OF_SCOPE``).
    * ``INTERNAL_ERROR`` -- an internal-consistency failure
      (``PROCEDURE_NOT_FOUND``): NOT a semantic decision, carries NO decision
      and can never move the FSM or create / resume / forward a case.
    """

    USER_RESPONSE = "user_response"
    INTERNAL_ERROR = "internal_error"


# ============================================================
# RUNTIME VALUE OBJECT
# ============================================================

@dataclass(frozen=True)
class RuntimeTurn:
    """The immutable runtime contract of ONE support turn.

    Composition result ONLY -- it introduces no vocabulary of its own: every
    field is either a payload-native user-facing string or the projection
    owner's VERBATIM ``decision`` / ``reason`` / ``fsm_event`` /
    ``case_action``. ``text`` is ALWAYS a non-blank, user-facing string.
    """

    kind: RuntimeTurnKind
    outcome_status: str
    conversation_id: str
    text: str
    decision: Optional[str]
    reason: str
    case_action: CaseAction
    fsm_event: Optional[SupportEvent]

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RuntimeTurnKind):
            raise ValueError("kind must be a RuntimeTurnKind.")
        _require_non_blank(self.outcome_status, "outcome_status")
        _require_non_blank(self.conversation_id, "conversation_id")
        _require_non_blank(self.text, "text")
        _require_non_blank(self.reason, "reason")

        if not isinstance(self.case_action, CaseAction):
            raise ValueError("case_action must be a CaseAction.")
        if self.fsm_event is not None and not isinstance(
            self.fsm_event, SupportEvent
        ):
            raise ValueError("fsm_event must be a SupportEvent or None.")

        if self.kind is RuntimeTurnKind.INTERNAL_ERROR:
            # An internal-consistency failure: no decision, no lifecycle
            # effect, no fabricated vocabulary.
            if self.outcome_status != _PROCEDURE_NOT_FOUND:
                raise ValueError(
                    "INTERNAL_ERROR may only carry a PROCEDURE_NOT_FOUND "
                    f"outcome, got {self.outcome_status!r}."
                )
            if self.decision is not None:
                raise ValueError(
                    "INTERNAL_ERROR must not carry a decision."
                )
            if self.reason != _REASON_PROCEDURE_NOT_FOUND:
                raise ValueError(
                    "INTERNAL_ERROR reason must be PROCEDURE_NOT_FOUND."
                )
            if self.fsm_event is not None:
                raise ValueError("INTERNAL_ERROR must not move the FSM.")
            if self.case_action is not CaseAction.NONE:
                raise ValueError(
                    "INTERNAL_ERROR must not create / resume / forward a case."
                )
        else:
            # A normal user-facing turn MUST carry a real channel decision.
            _require_non_blank(self.decision, "decision")


def _require_non_blank(value: Optional[str], field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-blank string.")


# ============================================================
# PROJECTION
# ============================================================

def adapt_turn_outcome(
    outcome: RuntimeTurnOutcomeLike,
    state: ConversationState,
) -> RuntimeTurn:
    """Project ONE already-made support turn into the runtime contract.

    Pure, total and deterministic over the closed vocabularies: it reads only
    the typed ``outcome`` and the CURRENT FSM ``state``. It re-derives NO
    decision -- ``decision`` / ``reason`` / ``fsm_event`` / ``case_action``
    come VERBATIM from
    :func:`support.lifecycle_projection.project_turn_outcome`, and the ONLY
    value added here is the user-facing ``text`` plus the typed
    ``USER_RESPONSE`` / ``INTERNAL_ERROR`` distinction.
    """
    projection = project_turn_outcome(outcome, state)
    return RuntimeTurn(
        kind=_kind_for(projection.decision),
        outcome_status=projection.outcome_status,
        conversation_id=outcome.conversation_id,
        text=_extract_text(outcome, projection.outcome_status),
        decision=projection.decision,
        reason=projection.reason,
        case_action=projection.case_action,
        fsm_event=projection.fsm_event,
    )


def _kind_for(decision: Optional[str]) -> RuntimeTurnKind:
    """The typed kind is a mechanical relabeling of the projection's verdict:
    a missing decision is EXACTLY the ``PROCEDURE_NOT_FOUND`` internal
    failure, and every other projection carries a real decision."""
    if decision is None:
        return RuntimeTurnKind.INTERNAL_ERROR
    return RuntimeTurnKind.USER_RESPONSE


def _extract_text(outcome: RuntimeTurnOutcomeLike, status: str) -> str:
    """Extract the user-facing text from the outcome's OWN native payload.

    No re-rendering, no LLM, no provider. Each family uses its own frozen
    text; ``NO_PROCEDURE`` reuses the existing deterministic clarification
    capability and ``PROCEDURE_NOT_FOUND`` reuses the existing neutral
    internal-error wording.
    """
    if status == _TURN_EXECUTED:
        text = _payload_member(outcome.persisted, ("turn", "assistant_text"), "persisted")
    elif status == _KNOWLEDGE_QA:
        text = _payload_member(outcome.knowledge, ("text",), "knowledge")
    elif status == _CLARIFICATION:
        text = _payload_member(outcome.clarification, ("text",), "clarification")
    elif status in (_ESCALATE, _OUT_OF_SCOPE):
        text = outcome.text
    elif status == _NO_PROCEDURE:
        text = clarify().text
    elif status == _PROCEDURE_NOT_FOUND:
        text = _INTERNAL_ERROR_TEXT
    else:
        raise ValueError(f"unhandled SupportTurnOutcome.status {status!r}")

    _require_non_blank(text, "text")
    return text


def _payload_member(root: Optional[object], path: tuple[str, ...], label: str):
    """Walk a required payload path, failing loudly if it is absent."""
    value: Optional[object] = root
    for attribute in path:
        if value is None:
            raise ValueError(f"{label} payload is missing its {attribute!r}.")
        value = getattr(value, attribute, None)
    return value


__all__ = [
    "RuntimeTurn",
    "RuntimeTurnKind",
    "RuntimeTurnOutcomeLike",
    "adapt_turn_outcome",
]
