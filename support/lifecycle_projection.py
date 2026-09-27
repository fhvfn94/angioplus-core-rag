"""
Human Lifecycle Projection (Slice HL-1) -- translate an ALREADY-MADE support
decision into the EXISTING human-lifecycle / runtime vocabulary.

Infrastructure only. Pure and total over the closed vocabularies it consumes.

WHAT THIS MODULE IS
-------------------
A PURE, DETERMINISTIC PROJECTION layer between the already-typed
:class:`~support.support_core.SupportTurnOutcome` produced by the application
boundary and the EXISTING conversation FSM (``support.domain``) and the
EXISTING channel/n8n runtime vocabulary (``decision`` / ``reason`` /
``case_action``).

It is deliberately NOT a second Router. ``SupportCore`` has already decided
WHAT the turn means and WHICH action was selected; this module only translates
that frozen decision into the existing lifecycle vocabulary. It NEVER:

  * reads the raw message or a :class:`~support.observation.SemanticObservation`;
  * performs any semantic classification or precedence gate;
  * reads procedure transitions or re-derives "exhausted" / "terminal" state;
  * executes a transition, mutates state, renders wording or calls anything;
  * constructs a ``SupportCore`` or touches DB / HTTP / provider / Tracker.

It is fed two already-typed objects and returns one immutable
:class:`LifecycleProjection`. It has no message text, no side effects and no
I/O of any kind.

INPUTS (typed only)
-------------------
  * ``SupportTurnOutcome`` (``support.support_core``) -- the boundary result,
    carrying exactly one family payload and the router's ``RoutingDecision``
    VERBATIM inside ``persisted.turn.decision`` for a ``TURN_EXECUTED`` turn.
  * ``ConversationState`` (``support.domain``) -- the CURRENT FSM triple
    ``(conversation_mode, support_status, handoff_status)``. Only the mode and
    the handoff status are consulted.

OUTPUT
------
A :class:`LifecycleProjection`:

  * ``outcome_status``  -- the ``SupportTurnOutcome.status`` VERBATIM;
  * ``decision``        -- the EXISTING channel/n8n decision string (or
                           ``None`` for an internal-consistency failure);
  * ``reason``          -- the engine reason / outcome status, VERBATIM;
  * ``fsm_event``       -- the EXISTING ``SupportEvent`` to apply, or ``None``
                           when the projection must NOT move the FSM;
  * ``case_action``     -- the EXISTING runtime case action
                           (``CREATE_NEW`` / ``RESUME`` / ``NONE``).

MAPPING (see the report; audit-approved)
----------------------------------------
  Global ESCALATE / TURN_EXECUTED ESCALATE  -> ``ESCALATE_NEW`` /
      ``ESCALATE_RESUME`` (``RESUME`` only from ``waiting_user``) and
      ``CREATE_NEW`` / ``RESUME``; a human-owned conversation is a defensive
      ``USER_MESSAGE_HUMAN`` / ``NONE``.
  AI-owned non-handoff families (RECOMMEND_STEP / CLARIFY / KNOWLEDGE_QA /
      CLARIFICATION / OUT_OF_SCOPE) -> ``USER_MESSAGE_AI`` / ``NONE``.
  RESOLVE -> ``AI_RESOLVED`` / ``NONE`` (a self-loop) on plain ``AI_ACTIVE``,
      and deliberately NO FSM event from ``waiting_user`` (the current FSM has
      no legal AI-resolution-confirmation transition there and this layer must
      never emit a knowingly invalid event).
  NO_PROCEDURE / PROCEDURE_NOT_FOUND -> NO FSM event and ``NONE``; the latter
      additionally carries ``decision=None`` (an internal consistency failure,
      not a semantic/runtime decision).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol

from support.domain import (
    ConversationMode,
    ConversationState,
    HandoffStatus,
    SupportEvent,
)
from support.router import ClarifyReason, EscalateReason, RouteAction


# ============================================================
# CONSUMED CONTRACT (structural)
# ============================================================

# These literals are the EXISTING boundary status vocabulary owned by
# ``support.support_core``. They are pinned here (rather than imported)
# because the repository architecture guard
# ``test_no_runtime_module_imports_support_core_yet`` forbids ANY sibling
# module from importing the composition boundary. The focused tests build
# outcomes from the REAL ``support_core`` constants, so any drift fails
# loudly.
_TURN_EXECUTED = "TURN_EXECUTED"
_KNOWLEDGE_QA = "KNOWLEDGE_QA"
_CLARIFICATION = "CLARIFICATION"
_ESCALATE = "ESCALATE"
_OUT_OF_SCOPE = "OUT_OF_SCOPE"
_NO_PROCEDURE = "NO_PROCEDURE"
_PROCEDURE_NOT_FOUND = "PROCEDURE_NOT_FOUND"


class SupportTurnOutcomeLike(Protocol):
    """Structural view of the ONLY fields this projection consumes.

    The real producer is ``support.support_core.SupportTurnOutcome``; it
    satisfies this protocol. Declaring it structurally keeps this PURE module
    free of any dependency on the composition boundary while still documenting
    the exact contract it reads (``status`` and, for ``TURN_EXECUTED``,
    ``persisted.turn.decision``).
    """

    status: str
    persisted: Optional[object]


# ============================================================
# CLOSED VOCABULARY (existing runtime / n8n case actions)
# ============================================================

class CaseAction(str, Enum):
    """The EXISTING channel/n8n case action vocabulary.

    ``HUMAN_FORWARD`` belongs to the runtime's human-forward path, which runs
    BEFORE the engine; :func:`project_turn_outcome` never emits it (it is kept
    here so the closed vocabulary stays in one place).
    """

    NONE = "NONE"
    HUMAN_FORWARD = "HUMAN_FORWARD"
    CREATE_NEW = "CREATE_NEW"
    RESUME = "RESUME"


# --- EXISTING channel/n8n ``decision`` strings ----------------------------
_DECISION_ANSWER = "ANSWER"
_DECISION_CLARIFY = "CLARIFY"
_DECISION_ESCALATE = "ESCALATE"
_DECISION_OUT_OF_SCOPE = "OUT_OF_SCOPE"

# --- EXISTING runtime ``reason`` vocabulary -------------------------------
# Global ESCALATE has exactly ONE cause (a user request) and reuses the
# EXISTING legacy runtime reason; a procedure ESCALATE preserves the engine's
# own ``EscalateReason`` value VERBATIM.
_REASON_GLOBAL_ESCALATE = "USER_REQUESTED_HUMAN"
_REASON_RESOLVED = "RESOLVED"
_REASON_PROCEDURE_STEP = "PROCEDURE_STEP"
_REASON_OUT_OF_SCOPE = "OUT_OF_SCOPE"
_REASON_KNOWLEDGE_QA = "KNOWLEDGE_QA"
_REASON_CLARIFICATION = "CLARIFICATION"
_REASON_NO_PROCEDURE = "NO_PROCEDURE"
_REASON_PROCEDURE_NOT_FOUND = "PROCEDURE_NOT_FOUND"

_HUMAN_OWNED_EVENT = SupportEvent.USER_MESSAGE_HUMAN
_AI_OWNED_EVENT = SupportEvent.USER_MESSAGE_AI

_ESCALATION_EVENTS = frozenset(
    {SupportEvent.ESCALATE_NEW, SupportEvent.ESCALATE_RESUME}
)
_ESCALATION_CASE_ACTIONS = frozenset(
    {CaseAction.CREATE_NEW, CaseAction.RESUME}
)


# ============================================================
# INTERNAL LIFECYCLE INTENT (state-independent classification)
# ============================================================

class _LifecycleKind(str, Enum):
    """The lifecycle effect an outcome implies, BEFORE the current state is
    consulted. Internal only; never exposed and never a semantic decision."""

    ESCALATE = "escalate"    # hand to a human / terminal support
    AI_TURN = "ai_turn"      # AI-owned turn, no handoff
    RESOLVE = "resolve"      # the problem is resolved
    NO_EFFECT = "no_effect"  # must not move the FSM at all


@dataclass(frozen=True)
class _Classification:
    decision: Optional[str]
    reason: str
    kind: _LifecycleKind
    procedure_action: Optional[RouteAction] = None
    escalation_reason: Optional[EscalateReason] = None
    clarify_reason: Optional[ClarifyReason] = None


# ============================================================
# PROJECTION VALUE OBJECT
# ============================================================

@dataclass(frozen=True)
class LifecycleProjection:
    """An immutable projection of ONE already-made support decision.

    Composition result ONLY -- it introduces no new vocabulary, owns no
    wording and performs no transition. ``decision`` is ``None`` exactly for a
    ``PROCEDURE_NOT_FOUND`` internal-consistency failure (HL-2 owns the
    runtime/HTTP treatment).
    """

    outcome_status: str
    decision: Optional[str]
    reason: str
    fsm_event: Optional[SupportEvent]
    case_action: CaseAction
    procedure_action: Optional[RouteAction] = None
    escalation_reason: Optional[EscalateReason] = None
    clarify_reason: Optional[ClarifyReason] = None

    def __post_init__(self) -> None:
        _require_non_blank(self.outcome_status, "outcome_status")
        if self.decision is not None:
            _require_non_blank(self.decision, "decision")
        _require_non_blank(self.reason, "reason")

        if self.fsm_event is not None and not isinstance(
            self.fsm_event, SupportEvent
        ):
            raise ValueError("fsm_event must be a SupportEvent or None.")
        if not isinstance(self.case_action, CaseAction):
            raise ValueError("case_action must be a CaseAction.")
        if self.procedure_action is not None and not isinstance(
            self.procedure_action, RouteAction
        ):
            raise ValueError("procedure_action must be a RouteAction or None.")
        if self.escalation_reason is not None and not isinstance(
            self.escalation_reason, EscalateReason
        ):
            raise ValueError(
                "escalation_reason must be an EscalateReason or None."
            )
        if self.clarify_reason is not None and not isinstance(
            self.clarify_reason, ClarifyReason
        ):
            raise ValueError("clarify_reason must be a ClarifyReason or None.")

        # Pairing invariant: an escalation event requires an escalation
        # case action; every other projection carries NONE (or the runtime's
        # pre-engine HUMAN_FORWARD, which this layer never emits).
        if self.fsm_event in _ESCALATION_EVENTS:
            if self.case_action not in _ESCALATION_CASE_ACTIONS:
                raise ValueError(
                    f"{self.fsm_event.value} requires a CREATE_NEW/RESUME "
                    f"case action, got {self.case_action.value}."
                )
        elif self.case_action not in (CaseAction.NONE, CaseAction.HUMAN_FORWARD):
            raise ValueError(
                f"{self.case_action.value} requires an escalation event."
            )


def _require_non_blank(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-blank string.")


# ============================================================
# PROJECTION
# ============================================================

def project_turn_outcome(
    outcome: SupportTurnOutcomeLike,
    state: ConversationState,
) -> LifecycleProjection:
    """Project ONE already-made decision into the lifecycle vocabulary.

    Pure, total and deterministic over the closed vocabularies: it reads only
    the typed ``outcome`` and the current FSM ``state`` and returns an
    immutable :class:`LifecycleProjection`. It performs no I/O, no rendering,
    no transition and no state mutation, and it re-derives NO semantic
    decision -- every field is a mechanical relabeling of an existing one.
    """
    classification = _classify(outcome)
    fsm_event, case_action = _project_events(classification.kind, state)
    return LifecycleProjection(
        outcome_status=outcome.status,
        decision=classification.decision,
        reason=classification.reason,
        fsm_event=fsm_event,
        case_action=case_action,
        procedure_action=classification.procedure_action,
        escalation_reason=classification.escalation_reason,
        clarify_reason=classification.clarify_reason,
    )


def _classify(outcome: SupportTurnOutcomeLike) -> _Classification:
    """Map the outcome's ALREADY-MADE decision to a decision/reason + intent.

    No message text is available or consulted; every branch keys on a frozen
    enum member or a closed status constant.
    """
    status = outcome.status

    if status == _TURN_EXECUTED:
        decision = outcome.persisted.turn.decision
        action = decision.action
        if action is RouteAction.RECOMMEND_STEP:
            return _Classification(
                _DECISION_ANSWER,
                _REASON_PROCEDURE_STEP,
                _LifecycleKind.AI_TURN,
                procedure_action=action,
            )
        if action is RouteAction.CLARIFY:
            return _Classification(
                _DECISION_CLARIFY,
                _reason_value(decision.clarify_reason, action),
                _LifecycleKind.AI_TURN,
                procedure_action=action,
                clarify_reason=decision.clarify_reason,
            )
        if action is RouteAction.ESCALATE:
            return _Classification(
                _DECISION_ESCALATE,
                _reason_value(decision.escalate_reason, action),
                _LifecycleKind.ESCALATE,
                procedure_action=action,
                escalation_reason=decision.escalate_reason,
            )
        if action is RouteAction.RESOLVE:
            return _Classification(
                _DECISION_ANSWER,
                _REASON_RESOLVED,
                _LifecycleKind.RESOLVE,
                procedure_action=action,
            )
        if action is RouteAction.OUT_OF_SCOPE:
            return _Classification(
                _DECISION_OUT_OF_SCOPE,
                _REASON_OUT_OF_SCOPE,
                _LifecycleKind.AI_TURN,
                procedure_action=action,
            )
        raise ValueError(f"unhandled RouteAction {action!r}")

    if status == _ESCALATE:
        return _Classification(
            _DECISION_ESCALATE,
            _REASON_GLOBAL_ESCALATE,
            _LifecycleKind.ESCALATE,
            escalation_reason=EscalateReason.USER_REQUESTED_HUMAN,
        )
    if status == _OUT_OF_SCOPE:
        return _Classification(
            _DECISION_OUT_OF_SCOPE, _REASON_OUT_OF_SCOPE, _LifecycleKind.AI_TURN
        )
    if status == _KNOWLEDGE_QA:
        return _Classification(
            _DECISION_ANSWER, _REASON_KNOWLEDGE_QA, _LifecycleKind.AI_TURN
        )
    if status == _CLARIFICATION:
        return _Classification(
            _DECISION_CLARIFY, _REASON_CLARIFICATION, _LifecycleKind.AI_TURN
        )
    if status == _NO_PROCEDURE:
        # No capability ran and nothing was persisted: NO FSM event.
        return _Classification(
            _DECISION_CLARIFY, _REASON_NO_PROCEDURE, _LifecycleKind.NO_EFFECT
        )
    if status == _PROCEDURE_NOT_FOUND:
        # Internal consistency failure: not a semantic/runtime decision.
        return _Classification(
            None, _REASON_PROCEDURE_NOT_FOUND, _LifecycleKind.NO_EFFECT
        )

    raise ValueError(f"unhandled SupportTurnOutcome.status {status!r}")


def _reason_value(reason, action: RouteAction) -> str:
    """Preserve the router's own reason value VERBATIM (fail closed if the
    frozen ``RoutingDecision`` invariant is violated)."""
    if reason is None:
        raise ValueError(f"RoutingDecision(action={action.value}) has no reason.")
    return reason.value


def _project_events(
    kind: _LifecycleKind, state: ConversationState
) -> tuple[Optional[SupportEvent], CaseAction]:
    """Select the FSM event + case action for the intent and CURRENT state.

    State is consulted ONLY for ownership and for the ``waiting_user`` resume
    distinction; this is a lookup, not a decision.
    """
    # A NO_EFFECT intent (NO_PROCEDURE / PROCEDURE_NOT_FOUND) never moves the
    # FSM, regardless of the current state.
    if kind is _LifecycleKind.NO_EFFECT:
        return None, CaseAction.NONE

    mode = state[0]
    handoff = state[2]

    # Defensive: the engine should not run while a human owns the
    # conversation; mirror the EXISTING runtime human-message event and create
    # no case action.
    if mode is ConversationMode.HUMAN:
        return _HUMAN_OWNED_EVENT, CaseAction.NONE

    if kind is _LifecycleKind.ESCALATE:
        if handoff is HandoffStatus.WAITING_USER:
            return SupportEvent.ESCALATE_RESUME, CaseAction.RESUME
        return SupportEvent.ESCALATE_NEW, CaseAction.CREATE_NEW

    if kind is _LifecycleKind.AI_TURN:
        return _AI_OWNED_EVENT, CaseAction.NONE

    if kind is _LifecycleKind.RESOLVE:
        # waiting_user is AI-owned but the FSM has NO legal
        # AI-resolution-confirmation transition there: emit NO event rather
        # than a knowingly invalid one (a proper AI confirmation lifecycle is
        # deferred).
        if handoff is HandoffStatus.WAITING_USER:
            return None, CaseAction.NONE
        return SupportEvent.AI_RESOLVED, CaseAction.NONE

    raise ValueError(f"unhandled lifecycle kind {kind!r}")


__all__ = [
    "CaseAction",
    "LifecycleProjection",
    "SupportTurnOutcomeLike",
    "project_turn_outcome",
]
