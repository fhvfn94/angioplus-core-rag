"""
Turn Recovery -- the DURABLE recovery contract for ONE applied turn execution
(HL-2c.3a).

Infrastructure only. NOT wired into any production runtime path yet
(server.py, alice_support_agent.py, n8n, Tracker). Pure value object + pure
reproducer; the persistence that stores it lives in
:mod:`support.turn_recovery_store` and the table in :mod:`support.models`.

WHY IT EXISTS (the W3 crash window)
-----------------------------------
:class:`~support.turn_executor.SupportTurnExecutor` commits the ProcedureState
CAS in ONE short transaction; the runtime (a LATER slice) then persists the
user / assistant messages, applies the FSM event and finalizes the inbound
event in its OWN transaction. A crash BETWEEN the CAS commit and that
finalization leaves a duplicate delivery that is a legitimately idempotent
REPLAY -- but the runtime then holds only the content-free
:class:`~support.runtime_outcome.RuntimeReplay`
(``PersistedTurnResult.turn`` is ``None``), so it can neither re-render the
answer (forbidden: no LLM / wording rerun) nor finish the original request
from nothing.

This module closes exactly that gap WITHOUT running anything a second time: it
defines the small, STABLE, scalar payload that is stored ATOMICALLY with the
CAS, and a pure, deterministic reproducer of the ORIGINAL
:class:`~support.runtime_outcome.RuntimeTurn` from that payload.

OWNERSHIP (deliberate)
----------------------
``ProcedureState`` is TECHNICAL diagnostic / procedure progress.
``ConversationState`` is RUNTIME / human lifecycle (mode, support_status,
handoff_status). They are different concerns and are persisted in different
places: the recovery record lives in its OWN table keyed by ``execution_id``
(:class:`~support.models.TurnRecoveryRow`), written by
:class:`~support.turn_recovery_store.TurnRecoveryStore` -- never by
``ProcedureStateStore`` and never as ``procedure_states`` columns.

WHAT IS STORED (and why it is the minimum)
------------------------------------------
For the ONLY path that CASes (PROCEDURE_SUPPORT) the runtime projection
:func:`support.runtime_outcome.adapt_turn_outcome` needs exactly:

  * the user-facing ``text`` -- the engine's own ``TurnResult.assistant_text``;
  * the router's ``RoutingDecision`` -- but ONLY ``action`` /
    ``escalate_reason`` / ``clarify_reason`` (``next_step`` /
    ``transition_type`` are NOT consulted by the projection and are NOT
    stored);
  * the runtime PROJECTION SNAPSHOT -- ``projection_mode`` /
    ``projection_support_status`` / ``projection_handoff_status``. These are
    SNAPSHOT INPUTS captured at CAS time because the projected ``fsm_event`` /
    ``case_action`` are STATE-DEPENDENT (e.g. ``ESCALATE_RESUME`` vs
    ``ESCALATE_NEW``): re-deriving them from the CURRENT conversation state at
    recovery time could yield a DIFFERENT result, so the projection state is
    snapshotted rather than re-read. The triple is carried as an OPAQUE
    snapshot -- it is NOT modelled as, or persisted as, ProcedureState.

SEMANTICS (non-negotiable)
--------------------------
  * NO LLM / retrieval rerun, NO wording regeneration, NO second procedure
    execution -- recovery is a pure function of the STORED payload.
  * NO dependence on the mutable CURRENT conversation state.
  * ABSENT and CORRUPT are different: "no record" is legitimate and is
    represented EXPLICITLY as ``None``; an existing partial / malformed record
    raises :class:`TurnRecoveryCorruption` (fail closed LOUDLY). A malformed
    durable record is NEVER silently coerced to ``None``.

The reproducer is a SEPARATE concept from the NEW-EXECUTION fold
(:func:`support.runtime_outcome.adapt_turn_outcome`) and from the
finalized-HTTP replay of a LATER slice: it does NOT touch
:class:`~support.runtime_outcome.RuntimeReplay` (which stays content-free) and
adds no field to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from support.domain import (
    ConversationMode,
    ConversationState,
    HandoffStatus,
    SupportStatus,
)
from support.router import (
    ClarifyReason,
    EscalateReason,
    RouteAction,
    RoutingDecision,
)
from support.runtime_outcome import RuntimeTurn, adapt_turn_outcome


# The ONE boundary status a CAS-persisted turn can carry: the executor's CAS
# runs ONLY on the PROCEDURE_SUPPORT path (every other family either writes
# nothing or terminates before the executor at all).
_OUTCOME_STATUS = "TURN_EXECUTED"


class TurnRecoveryCorruption(RuntimeError):
    """A durable recovery record EXISTS but is partial / malformed.

    Raised LOUDLY (never swallowed, never degraded to ``None``): a record that
    exists but cannot be interpreted means durable recovery data was damaged,
    and silently pretending the record is absent would hide the corruption.
    Absence of a record is a DIFFERENT, legitimate condition (``None``).
    """


def _require_non_blank(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TurnRecoveryCorruption(
            f"recovery field {field!r} must be a non-blank string."
        )
    return value


@dataclass(frozen=True)
class TurnRecoveryPayload:
    """The canonical DURABLE recovery payload of ONE applied turn execution.

    Every field is an EXPLICIT, STABLE SCALAR (an enum ``.value`` string, or
    ``None``) -- no Python object, no ``repr``, nothing that could drift across
    versions. The payload is written ATOMICALLY with the CAS (the SAME
    transaction) and, on an idempotent replay, read back VERBATIM.

    ``execution_id`` is the KEY (the opaque idempotency identity already used
    by ``ProcedureStateStore``); ``conversation_id`` is carried so a reader can
    refuse to serve a record that does not belong to the conversation asking.
    """

    execution_id: str
    conversation_id: str
    text: str
    action: str
    projection_mode: str
    projection_support_status: str
    escalate_reason: Optional[str] = None
    clarify_reason: Optional[str] = None
    projection_handoff_status: Optional[str] = None

    def __post_init__(self) -> None:
        for name in (
            "execution_id",
            "conversation_id",
            "text",
            "action",
            "projection_mode",
            "projection_support_status",
        ):
            _require_non_blank(getattr(self, name), name)
        for name in (
            "escalate_reason",
            "clarify_reason",
            "projection_handoff_status",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_non_blank(value, name)

    # ---------------------------------------------------------
    # CAPTURE (engine-side, at the CAS point)
    # ---------------------------------------------------------

    @classmethod
    def capture(
        cls,
        *,
        execution_id: str,
        conversation_id: str,
        turn,
        projection_state: ConversationState,
    ) -> "TurnRecoveryPayload":
        """Capture the canonical engine result of ONE applied turn.

        ``turn`` is the executor's own turn result (the engine's
        ``TurnResult``); ``projection_state`` is the caller-supplied immutable
        :class:`~support.domain.ConversationState` the runtime projection will
        consult. Pure scalar extraction: no I/O, no rendering and no second
        analysis: this module never imports, builds or runs the engine.
        """
        decision = turn.decision
        return cls(
            execution_id=execution_id,
            conversation_id=conversation_id,
            text=turn.assistant_text,
            action=decision.action.value,
            projection_mode=projection_state[0].value,
            projection_support_status=projection_state[1].value,
            escalate_reason=(
                None
                if decision.escalate_reason is None
                else decision.escalate_reason.value
            ),
            clarify_reason=(
                None
                if decision.clarify_reason is None
                else decision.clarify_reason.value
            ),
            projection_handoff_status=(
                None
                if projection_state[2] is None
                else projection_state[2].value
            ),
        )

    # ---------------------------------------------------------
    # TYPED RECONSTRUCTION
    # ---------------------------------------------------------

    def conversation_state(self) -> ConversationState:
        """The snapshotted FSM triple, as a domain value object."""
        try:
            return (
                ConversationMode(self.projection_mode),
                SupportStatus(self.projection_support_status),
                (
                    None
                    if self.projection_handoff_status is None
                    else HandoffStatus(self.projection_handoff_status)
                ),
            )
        except ValueError as exc:
            raise TurnRecoveryCorruption(
                f"recovery projection snapshot is not a valid "
                f"ConversationState: {exc}"
            ) from exc

    def routing_decision(self) -> RoutingDecision:
        """The captured router verdict (only the projection-relevant fields).

        ``next_step`` / ``transition_type`` are deliberately absent: the
        projection never reads them, so storing them would be redundant.
        """
        try:
            return RoutingDecision(
                action=RouteAction(self.action),
                escalate_reason=(
                    None
                    if self.escalate_reason is None
                    else EscalateReason(self.escalate_reason)
                ),
                clarify_reason=(
                    None
                    if self.clarify_reason is None
                    else ClarifyReason(self.clarify_reason)
                ),
            )
        except ValueError as exc:
            raise TurnRecoveryCorruption(
                f"recovery engine result is not a valid RoutingDecision: {exc}"
            ) from exc


# ============================================================
# REPRODUCTION
# ============================================================

@dataclass(frozen=True)
class _RecoveredTurn:
    """Structural ``PersistedTurnResult.turn`` view for the projection.

    Deliberately NOT the real ``TurnResult`` (which is not persistable and
    whose other fields -- ``observation`` / ``response`` / ``knowledge`` --
    the projection never reads). It carries EXACTLY the two members the
    projection owner consumes from a ``TURN_EXECUTED`` turn.
    """

    assistant_text: str
    decision: RoutingDecision


@dataclass(frozen=True)
class _RecoveredPersisted:
    """Structural ``PersistedTurnResult`` view: a NON-replay executed turn."""

    turn: _RecoveredTurn
    replayed: bool = False


@dataclass(frozen=True)
class _RecoveredOutcome:
    """Structural ``SupportTurnOutcome`` view fed to the projection owner.

    Only the members the projection actually reads for the ``TURN_EXECUTED``
    path are provided (``status`` / ``conversation_id`` / ``persisted``);
    declaring the real bearer structurally keeps this PURE module free of any
    dependency on the composition boundary.
    """

    status: str
    conversation_id: str
    persisted: _RecoveredPersisted


def reproduce_runtime_turn(payload: TurnRecoveryPayload) -> RuntimeTurn:
    """Rebuild the ORIGINAL :class:`RuntimeTurn` from a stored payload.

    Pure, deterministic and total over a VALID payload: it reconstructs the
    typed ``RoutingDecision`` and ``ConversationState`` and reuses the SINGLE
    projection owner (:func:`support.runtime_outcome.adapt_turn_outcome`) --
    the same code path a NEW execution uses -- so ``decision`` / ``reason`` /
    ``fsm_event`` / ``case_action`` / ``text`` are reproduced EXACTLY, with no
    LLM, no retrieval, no procedure execution and no re-rendering.

    It NEVER consults the current conversation state (the projection inputs
    come from the payload alone) and never produces a
    :class:`~support.runtime_outcome.RuntimeReplay` -- a stored payload is by
    construction a genuine, already-applied turn, so an internal-consistency
    failure raises rather than fabricating a replay.
    """
    outcome = _RecoveredOutcome(
        status=_OUTCOME_STATUS,
        conversation_id=payload.conversation_id,
        persisted=_RecoveredPersisted(
            turn=_RecoveredTurn(
                assistant_text=payload.text,
                decision=payload.routing_decision(),
            )
        ),
    )
    result = adapt_turn_outcome(outcome, payload.conversation_state())
    if not isinstance(result, RuntimeTurn):
        raise TurnRecoveryCorruption(
            "a recovery payload must reproduce a RuntimeTurn, got "
            f"{type(result).__name__}."
        )
    return result


__all__ = [
    "TurnRecoveryCorruption",
    "TurnRecoveryPayload",
    "reproduce_runtime_turn",
]
