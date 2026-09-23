"""
Deterministic Support Router -- the ONLY transition reader.

Infrastructure only. NOT imported by any production runtime path
(server.py, alice_support_agent.py, n8n, Tracker, DB, File Search).

STRICT STATE / ROUTER SEPARATION
--------------------------------
The reducer (``support.procedure_state``) records WHAT happened and must
never read ``procedure.transitions``; it has no cursor. THIS module is its
counterpart:

  Reducer                               Router (this module)
  -----------------------------------   ------------------------------------
  records WHAT the observer reported     decides WHAT to do next
  never reads procedure transitions      is the ONLY transition reader
  never moves a cursor                   owns the cursor concept
  never derives "exhausted"              derives exhausted / terminator state
  never clears state from scope alone    decides to switch procedure

The router is a PURE, TOTAL, DETERMINISTIC function. It does no I/O, calls
no LLM, touches no DB and no framework. Given the same inputs it always
returns the same :class:`RoutingDecision` and it NEVER raises.

It derives the next action by walking the procedure's DECLARATIVE
``transitions`` table -- a table walk, not an LLM-driven priority list.
There is deliberately NO ``candidate_l1_action`` /
``candidate_action_already_attempted`` vocabulary here: that legacy
string-priority cascade (``apply_support_policy``) is exactly the
anti-pattern this router must not reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from support.observation import (
    ObservationScope,
    ProblemStatus,
    SemanticObservation,
    StepOutcome,
)
from support.procedure import Procedure
from support.procedure_state import ProcedureState


# ============================================================
# CLOSED VOCABULARIES
# ============================================================

class _StrEnum(str, Enum):
    """Python 3.10 compatible str-enum (values are plain strings)."""

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return str(self.value)


class RouteAction(_StrEnum):
    """The procedure-level action the router recommends.

    This is NOT a channel-level ``decision``/``reason``: transport mapping
    (Alice / n8n / Tracker) is a later slice.
    """

    RECOMMEND_STEP = "recommend_step"  # advise a specific procedure step
    CLARIFY = "clarify"                # ask the user for missing information
    ESCALATE = "escalate"              # hand to a human / terminal support
    RESOLVE = "resolve"                # the problem is resolved
    OUT_OF_SCOPE = "out_of_scope"      # not about a supported procedure


class EscalateReason(_StrEnum):
    """Why the router escalated. Mutually exclusive by construction.

    * USER_REQUESTED_HUMAN -- the user explicitly asked for a human.
    * PROCEDURE_TERMINAL   -- the walk followed a declarative edge to an
                              ESCALATE terminator (a NORMAL, expected end
                              of the procedure graph).
    * STEPS_EXHAUSTED      -- the walk dead-ended structurally: an
                              authoritative FAILED step with no outgoing
                              ``attempted_failed`` edge and no reachable
                              terminator (a malformed / stalled procedure).
    * INTERNAL_INCONSISTENCY -- the state cannot be trusted against the
                              procedure (id / version mismatch, dangling
                              recommended step). Users cannot fix this.
    """

    USER_REQUESTED_HUMAN = "user_requested_human"
    PROCEDURE_TERMINAL = "procedure_terminal"
    STEPS_EXHAUSTED = "steps_exhausted"
    INTERNAL_INCONSISTENCY = "internal_inconsistency"


class ClarifyReason(_StrEnum):
    """Why the router asked for clarification.

    * PENDING_CLARIFICATION -- the reducer already flagged an unattributed
                               report.
    * NO_PROCEDURE          -- no procedure is selected yet.
    * NEW_ISSUE_RESEELECT   -- the user raised a different problem; a new
                               procedure must be selected.
    * STEP_OUTCOME_UNKNOWN  -- the current step's outcome is inconclusive
                               (SKIPPED or DONE_UNKNOWN); the walk cannot
                               follow a transition until it is definitive.
    """

    PENDING_CLARIFICATION = "pending_clarification"
    NO_PROCEDURE = "no_procedure"
    NEW_ISSUE_RESEELECT = "new_issue_reselect"
    STEP_OUTCOME_UNKNOWN = "step_outcome_unknown"


# ============================================================
# DECISION VALUE OBJECT
# ============================================================

@dataclass(frozen=True)
class RoutingDecision:
    """The router's immutable verdict for one (state, observation, procedure).

    Invariant: exactly ONE reason field matching ``action`` is populated;
    every other optional field is ``None``.

      * ``action == RECOMMEND_STEP`` -> ``next_step`` set, ``transition_type``
        set (the fired edge kind), reasons ``None``.
      * ``action == ESCALATE``       -> ``escalate_reason`` set, rest ``None``.
      * ``action == CLARIFY``        -> ``clarify_reason`` set, rest ``None``.
      * ``action == RESOLVE``        -> all optional fields ``None``.
      * ``action == OUT_OF_SCOPE``   -> all optional fields ``None``.
    """

    action: RouteAction
    next_step: Optional[str] = None
    transition_type: Optional[str] = None
    escalate_reason: Optional[EscalateReason] = None
    clarify_reason: Optional[ClarifyReason] = None


# ============================================================
# PURE HELPERS
# ============================================================

def authoritative_outcome(
    state: ProcedureState, step_id: str
) -> Optional[StepOutcome]:
    """The latest recorded outcome for ``step_id``, or ``None``.

    The latest :class:`~support.procedure_state.StepAttempt` (highest
    ``sequence``) is authoritative. ``sequence`` is unique/contiguous per
    the Slice 2 reducer, so there are no ties. Reads no transitions.
    """
    latest: Optional[StepOutcome] = None
    latest_seq = -1
    for attempt in state.step_attempts:
        if attempt.step_id == step_id and attempt.sequence >= latest_seq:
            latest_seq = attempt.sequence
            latest = attempt.outcome
    return latest


def _escalate(
    reason: EscalateReason,
) -> RoutingDecision:
    return RoutingDecision(action=RouteAction.ESCALATE, escalate_reason=reason)


def _clarify(reason: ClarifyReason) -> RoutingDecision:
    return RoutingDecision(action=RouteAction.CLARIFY, clarify_reason=reason)


# ============================================================
# ROUTER
# ============================================================

def route(
    state: ProcedureState,
    observation: SemanticObservation,
    procedure: Optional[Procedure],
) -> RoutingDecision:
    """Pure, total, deterministic procedure router. Never raises.

    Precedence (first match wins):

      1. scope == OUT_OF_SCOPE            -> OUT_OF_SCOPE
      2. human_requested                  -> ESCALATE(USER_REQUESTED_HUMAN)
      3. problem_status == RESOLVED       -> RESOLVE
      4. procedure is None                -> CLARIFY(NO_PROCEDURE)
      5. identity / dangling cursor       -> ESCALATE(INTERNAL_INCONSISTENCY)
      6. observation.new_issue present    -> CLARIFY(NEW_ISSUE_RESEELECT)
      7. state.pending_clarification      -> CLARIFY(PENDING_CLARIFICATION)
      8. initial entry                    -> RECOMMEND_STEP(entry_step)
      9. deterministic graph walk         -> RECOMMEND_STEP / RESOLVE /
                                             ESCALATE / CLARIFY
     10. fallback                         -> ESCALATE(STEPS_EXHAUSTED) if the
                                             procedure declares ESCALATE, else
                                             CLARIFY(PENDING_CLARIFICATION)
    """

    # --- Gate 1: scope. OUT_OF_SCOPE is the OUTER gate; it wins even over
    #     an explicit human request (audit-approved outer precedence). ---
    if observation.scope == ObservationScope.OUT_OF_SCOPE:
        return RoutingDecision(action=RouteAction.OUT_OF_SCOPE)

    # --- Gate 2: explicit human request. A GLOBAL gate: it does not depend
    #     on the procedure declaring a user_requested_human edge and can
    #     never degrade to CLARIFY. (Scope already cleared above.) ---
    if observation.human_requested:
        return _escalate(EscalateReason.USER_REQUESTED_HUMAN)

    # --- Gate 3: problem resolved. ---
    if (
        observation.problem_status == ProblemStatus.RESOLVED
        or state.problem_status == ProblemStatus.RESOLVED
    ):
        return RoutingDecision(action=RouteAction.RESOLVE)

    # --- Gate 4: no procedure selected yet. Evaluated AFTER handoff and
    #     resolution so an explicit request / resolution wins over
    #     "no procedure". ---
    if procedure is None:
        return _clarify(ClarifyReason.NO_PROCEDURE)

    # --- Gate 5: procedure identity must match the state; a user cannot
    #     fix an internal inconsistency, so fail safe to ESCALATE. ---
    step_ids = procedure.step_ids()
    if (
        state.procedure_id is not None
        and state.procedure_id != procedure.procedure_id
    ):
        return _escalate(EscalateReason.INTERNAL_INCONSISTENCY)
    if (
        state.procedure_version is not None
        and state.procedure_version != procedure.version
    ):
        return _escalate(EscalateReason.INTERNAL_INCONSISTENCY)
    if (
        state.last_recommended_step is not None
        and state.last_recommended_step not in step_ids
    ):
        return _escalate(EscalateReason.INTERNAL_INCONSISTENCY)

    # --- Gate 6: a different problem was raised; reselect a procedure.
    #     The router neither continues the old procedure nor selects a new
    #     one (a pure function mutates nothing). ---
    if observation.new_issue and observation.new_issue.strip():
        return _clarify(ClarifyReason.NEW_ISSUE_RESEELECT)

    # --- Gate 7: the reducer flagged an unattributed report. ---
    if state.pending_clarification:
        return _clarify(ClarifyReason.PENDING_CLARIFICATION)

    # --- Gate 8: first contact with the procedure. ---
    if not state.step_attempts and state.last_recommended_step is None:
        return RoutingDecision(
            action=RouteAction.RECOMMEND_STEP,
            next_step=procedure.entry_step,
            transition_type=None,
        )

    # --- Gate 9: deterministic single-hop graph walk. ---
    return _walk(state, procedure)


def _walk(state: ProcedureState, procedure: Procedure) -> RoutingDecision:
    """Walk the declarative transition table, anchored on the cursor.

    Anchor: ``last_recommended_step`` else ``entry_step``. At each node use
    the AUTHORITATIVE outcome:

      * no attempt yet           -> RECOMMEND_STEP(node).
      * SUCCEEDED                -> follow ``resolved`` edge.
      * FAILED                   -> follow ``attempted_failed`` edge.
      * SKIPPED / DONE_UNKNOWN   -> inconclusive: CLARIFY(STEP_OUTCOME_UNKNOWN);
                                    never follow an edge.

    A step with an authoritative outcome is NEVER re-recommended; its
    outcome is used to continue the walk. Bounded by ``len(procedure.steps)``
    hops for loop safety. If a node with an authoritative FAILED outcome has
    no ``attempted_failed`` edge (structural dead-end), escalate.
    """
    step_ids = procedure.step_ids()
    current = state.last_recommended_step or procedure.entry_step
    fired_edge: Optional[str] = None

    for _ in range(len(procedure.steps) + 1):
        if current not in step_ids:
            # Defensive: should be unreachable after gate 5, but never raise.
            return _escalate(EscalateReason.INTERNAL_INCONSISTENCY)

        outcome = authoritative_outcome(state, current)

        # No authoritative outcome -> recommend THIS step. If we arrived
        # here by following an edge, report the fired edge kind so the
        # caller can see WHY this step was recommended.
        if outcome is None:
            return RoutingDecision(
                action=RouteAction.RECOMMEND_STEP,
                next_step=current,
                transition_type=fired_edge,
            )

        # Inconclusive outcome -> cannot follow any edge.
        if outcome in (StepOutcome.SKIPPED, StepOutcome.DONE_UNKNOWN):
            return _clarify(ClarifyReason.STEP_OUTCOME_UNKNOWN)

        step = procedure.steps[current]

        if outcome == StepOutcome.SUCCEEDED:
            transition_type = "resolved"
        elif outcome == StepOutcome.FAILED:
            transition_type = "attempted_failed"
        else:  # pragma: no cover - exhaustive over StepOutcome
            return _escalate(EscalateReason.INTERNAL_INCONSISTENCY)

        target = step.transitions.get(transition_type)

        # Structural dead-end: an authoritative outcome with no edge for it.
        if target is None:
            return _escalate(EscalateReason.STEPS_EXHAUSTED)

        # Reached a declarative terminator.
        if target in procedure.terminators:
            if target == "ESCALATE":
                return _escalate(EscalateReason.PROCEDURE_TERMINAL)
            if target == "RESOLVED":
                return RoutingDecision(action=RouteAction.RESOLVE)
            # A declared-but-unknown terminator: treat as exhaustion.
            return _escalate(EscalateReason.STEPS_EXHAUSTED)  # pragma: no cover

        # Follow the edge to the next step and continue the walk, remembering
        # the edge we just traversed.
        fired_edge = transition_type
        current = target

    # Loop safety: bounded walk did not converge.
    return _escalate(EscalateReason.STEPS_EXHAUSTED)  # pragma: no cover


__all__ = [
    "RouteAction",
    "EscalateReason",
    "ClarifyReason",
    "RoutingDecision",
    "authoritative_outcome",
    "route",
]
