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
    RuntimeTurn  (this module)     -- a genuine NEW turn
  or RuntimeReplay (this module)   -- an ENGINE REPLAY (already applied)

into the SINGLE immutable runtime value a transport layer consumes: either
the user-facing ``text`` + channel ``decision`` / ``reason`` +
``case_action`` + the ONE FSM ``fsm_event`` (or ``None``) of a genuine turn
(:class:`RuntimeTurn`), or the content-free "already applied -- do NOT
re-apply" marker of an engine replay (:class:`RuntimeReplay`).

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

The engine-replay detection (HL-2c.2) is likewise mechanical: it classifies
the ALREADY-TYPED ``persisted.replayed`` flag, adds no policy, and
short-circuits BEFORE the projection owner is consulted (see below).

It never reads the raw message or a
:class:`~support.observation.SemanticObservation`, never walks procedure
transitions, and runs no router, no reducer and no LLM.

ENGINE REPLAY (HL-2c.2)
-----------------------
Replay is EXECUTION SEMANTICS, not a business decision. When the ENGINE
already applied this execution (HL-2b idempotency, surfaced by HL-2c.1), the
outcome is a ``TURN_EXECUTED`` carrying a ``PersistedTurnResult`` with
``replayed`` ``True`` and ``turn`` ``None``. Such an outcome is projected to
:class:`RuntimeReplay` and NEVER reaches
:func:`support.lifecycle_projection.project_turn_outcome` -- so a replay can
produce NO decision, NO reason, NO FSM event, NO case action and NO
(re-generated) text. This is LEVEL B (engine / procedure replay); the
separate LEVEL A (a finalized inbound-event replay in the runtime) is NOT
handled here.

WHAT IT DELIBERATELY DOES NOT DO (later HL slices own these)
-----------------------------------------------------------
No DB / session work, no message persistence, no inbound-event idempotency,
no FSM execution, no HTTP serialization, no finalized-runtime replay, no
crash-recovery semantics, no server.py wiring, no n8n / Tracker, no
production composition root. It is a pure, total projection over typed
inputs, unit-testable in isolation.

This module carries NO ``retrieval_used`` flag: it is a legacy HTTP
compatibility field that ``SupportTurnOutcome`` does not expose truthfully.
Preserving it is an HL-2c decision, so HL-2a neither hardcodes, infers nor
inspects it. HL-2c.2 does NOT resolve it either: a runtime replay must not
hardcode, infer or regenerate ``retrieval_used`` from the decision. Exact
replay fidelity for this field requires the runtime's finalized idempotency
storage to persist it first (documented follow-up, before the production
cutover).
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
# ``test_only_the_sanctioned_runtime_module_imports_support_core`` forbids ANY
# OTHER sibling module from importing the composition boundary. The focused
# tests build outcomes from the REAL ``support_core`` constants, so any drift
# fails loudly.
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
# RUNTIME REPLAY (HL-2c.2)
# ============================================================

@dataclass(frozen=True)
class RuntimeReplay:
    """The runtime contract of ONE engine replay (an ALREADY-applied turn).

    NOT a turn. It carries the SMALLEST possible identity and NOTHING else:
    no ``text``, no ``decision``, no ``reason``, no ``case_action``, no
    ``fsm_event`` and no ``RuntimeTurnKind``. Those effect-bearing fields are
    STRUCTURALLY ABSENT, so a replay can never be mistaken for, coerced into
    or rendered as a new turn, and no FSM / case / wording effect is even
    expressible from it. It exists ONLY to tell the runtime: "this execution
    was already applied -- do NOT re-apply it".

    The answer text for a duplicate DELIVERY is deliberately NOT here: the
    engine produced NO wording for a replay (``persisted.turn`` is ``None``),
    and reconstructing the original response body is the runtime's OWN
    finalized-idempotency concern (LEVEL A), not this engine-level value.

    ``RuntimeReplay`` itself is the discriminator: ``isinstance(result,
    RuntimeReplay)`` distinguishes a replay from a genuine
    :class:`RuntimeTurn`; no shared ``kind`` enum member is introduced.
    """

    outcome_status: str
    conversation_id: str

    def __post_init__(self) -> None:
        _require_non_blank(self.outcome_status, "outcome_status")
        _require_non_blank(self.conversation_id, "conversation_id")


# The closed runtime RESULT vocabulary of ONE support-turn attempt: EITHER a
# genuine new turn OR an engine replay -- never both, never anything else.
RuntimeResult = RuntimeTurn | RuntimeReplay


# ============================================================
# PROJECTION
# ============================================================

_ABSENT = object()


def _is_engine_replay(outcome: RuntimeTurnOutcomeLike) -> bool:
    """Is this the ONE legitimate HL-2b / HL-2c.1 engine-replay shape?

    A replayed ``TURN_EXECUTED`` carries a ``PersistedTurnResult`` whose
    ``replayed`` is ``True`` and whose ``turn`` is ``None`` -- the ONLY such
    combination the ``PersistedTurnResult`` invariant admits (a replayed
    result carries no recomputed turn; a non-replayed one requires a turn).

    EVERY other shape is NOT a replay and follows the existing projection
    path UNCHANGED (fail closed), so a malformed executed payload can never
    be silently reclassified: a ``turn`` of ``None`` WITHOUT the replay flag,
    or a contradictory replay flag WITH a turn, are both rejected here.
    """
    if outcome.status != _TURN_EXECUTED:
        return False
    persisted = outcome.persisted
    if persisted is None:
        return False
    return (
        getattr(persisted, "replayed", _ABSENT) is True
        and getattr(persisted, "turn", _ABSENT) is None
    )


def adapt_turn_outcome(
    outcome: RuntimeTurnOutcomeLike,
    state: ConversationState,
) -> RuntimeResult:
    """Project ONE already-made support turn into the runtime contract.

    Returns EITHER a :class:`RuntimeTurn` (a genuine new application turn) OR
    a :class:`RuntimeReplay` (an engine replay).

    The engine-replay branch is detected FIRST and returns BEFORE
    :func:`support.lifecycle_projection.project_turn_outcome` is consulted,
    so an engine replay NEVER reaches LifecycleProjection and can produce NO
    decision, NO reason, NO FSM event, NO case action, and NO
    re-rendered text.

    For every NON-replay outcome the behaviour is EXACTLY as before: pure,
    total and deterministic over the closed vocabularies, it reads only the
    typed ``outcome`` and the CURRENT FSM ``state``, re-derives NO decision --
    ``decision`` / ``reason`` / ``fsm_event`` / ``case_action`` come VERBATIM
    from :func:`support.lifecycle_projection.project_turn_outcome` -- and the
    ONLY value added is the user-facing ``text`` plus the typed
    ``USER_RESPONSE`` / ``INTERNAL_ERROR`` distinction.
    """
    if _is_engine_replay(outcome):
        return RuntimeReplay(
            outcome_status=outcome.status,
            conversation_id=outcome.conversation_id,
        )
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
    "RuntimeReplay",
    "RuntimeResult",
    "RuntimeTurn",
    "RuntimeTurnKind",
    "RuntimeTurnOutcomeLike",
    "adapt_turn_outcome",
]
