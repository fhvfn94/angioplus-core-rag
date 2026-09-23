"""
Procedure State -- deterministic bookkeeping of troubleshooting progress.

Infrastructure only. NOT imported by any production runtime path
(server.py, alice_support_agent.py, n8n, Tracker, DB, File Search).

STRICT STATE / ROUTER SEPARATION
--------------------------------
This module is a pure REDUCER, not a router.

  Reducer (this module)                 Router (a LATER slice)
  -----------------------------------   ------------------------------------
  records WHAT the observer reported     decides WHAT to do next
  about steps and the problem            (recommend, advance cursor, resolve,
                                          escalate, clarify, switch procedure)
  never reads procedure transitions      reads procedure transitions
  never moves a cursor                   owns the cursor concept
  never derives "exhausted"              derives exhausted / terminator state
  never clears state from scope alone    decides to switch procedure

Consequently the reducer:

  * MUST NOT read ``procedure.transitions`` (of :class:`support.procedure.Procedure`);
  * MUST NOT follow ``attempted_failed`` / ``resolved`` / ``user_requested_human``
    edges;
  * MUST NOT advance, clear, or introduce any cursor (there is no
    ``current_step`` field);
  * MUST NOT set ``procedure_id`` / ``procedure_version`` (Router owns
    procedure selection; the reducer only carries them through);
  * MUST NOT derive ``procedure_exhausted`` or any terminator state;
  * MUST NOT implement replay/idempotency (transport idempotency already
    exists in ``support.idempotency``; a different observation is appended
    even if its content repeats).

The ``procedure`` argument is used ONLY for step-id validation (checking
that a reported step belongs to the procedure), never for transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from support.observation import (
    ProblemStatus,
    ReportedAction,
    SemanticObservation,
    StepOutcome,
)
from support.procedure import Procedure


# No I/O, no ORM, no framework imports.


# ============================================================
# STATE MODEL
# ============================================================

@dataclass(frozen=True)
class StepAttempt:
    """One reported attempt at one procedure step. Append-only."""

    step_id: str        # a step id of the current procedure (when known)
    outcome: StepOutcome
    # 0-based position within this state's attempt log (ordering only;
    # NOT a replay/idempotency key).
    sequence: int


@dataclass(frozen=True)
class ProcedureState:
    """Where the user is inside a troubleshooting procedure.

    This is the orthogonal axis to the conversation FSM
    (``support.domain``): the FSM owns WHO talks (ai/human) and the case
    lifecycle; ProcedureState owns WHERE the user is inside a procedure.
    Neither imports the other.
    """

    # --- which procedure is being walked (identity only; no cursor) ---
    issue: Optional[str] = None                  # problem label as understood
    procedure_id: Optional[str] = None           # Router-owned (carried through)
    procedure_version: Optional[int] = None      # Router-owned (carried through)

    # --- what the assistant last advised (a FACT about the dialogue) ---
    # Router-owned; the reducer only preserves it.
    last_recommended_step: Optional[str] = None

    # --- reported progress (the single source of truth) ---
    step_attempts: tuple[StepAttempt, ...] = ()

    # --- problem-level fact ---
    problem_status: ProblemStatus = ProblemStatus.UNKNOWN

    # --- clarification fact ---
    pending_clarification: bool = False
    clarification_topic: Optional[str] = None

    # --- schema ---
    schema_version: int = 1

    # ------------------------------------------------------------
    # DERIVED VIEWS (never stored)
    # ------------------------------------------------------------

    def attempted_step_ids(self) -> frozenset[str]:
        """Step ids that have at least one recorded attempt (derived)."""
        return frozenset(a.step_id for a in self.step_attempts)

    def last_attempt(self) -> Optional[StepAttempt]:
        """The most recently appended attempt, or None."""
        return self.step_attempts[-1] if self.step_attempts else None

    def has_attempt(self, step_id: str) -> bool:
        return any(a.step_id == step_id for a in self.step_attempts)


def initial_state() -> ProcedureState:
    """A fresh, empty procedure state."""
    return ProcedureState()


# ============================================================
# REDUCER
# ============================================================

def _clarification_context(state: ProcedureState) -> Optional[str]:
    """The context to point a clarification at.

    Prefers ``last_recommended_step``; falls back to ``issue``.
    """
    if state.last_recommended_step:
        return state.last_recommended_step
    if state.issue:
        return state.issue
    return None


def _dedup_actions(
    actions: tuple[ReportedAction, ...],
) -> list[ReportedAction]:
    """Drop exact duplicate ReportedActions within a single observation.

    This is LOCAL input de-duplication only (e.g. the same step reported
    twice in one message). It is NOT cross-turn / cross-event idempotency.
    Order of first appearance is preserved.
    """
    seen: set[tuple[Optional[str], Optional[StepOutcome]]] = set()
    result: list[ReportedAction] = []
    for action in actions:
        key = (action.step_id, action.outcome)
        if key in seen:
            continue
        seen.add(key)
        result.append(action)
    return result


def update_procedure_state(
    previous_state: ProcedureState,
    observation: SemanticObservation,
    procedure: Optional[Procedure],
) -> ProcedureState:
    """Pure, deterministic reducer. No I/O, no LLM, no DB, no routing.

    Returns the next :class:`ProcedureState` given the previous state, a
    routing-free :class:`~support.observation.SemanticObservation`, and the
    current (optional) :class:`~support.procedure.Procedure`.

    The ``procedure`` is used ONLY to validate that a reported step id
    exists; its ``transitions`` are never read.

    Behaviour summary (see the module docstring for the State/Router split):

      * Appends a :class:`StepAttempt` for each reported action whose
        ``step_id`` is a valid step of ``procedure`` (when known).
      * An unknown / invalid ``step_id`` is NEVER appended and NEVER
        silently dropped: it raises the clarification flag instead.
      * Updates ``problem_status`` from the observation.
      * Sets ``pending_clarification`` / ``clarification_topic`` when a
        report could not be attributed.
      * Handles a bare "didn't help" by attributing FAILED to the
        outstanding recommendation (context resolution, not routing).
      * Carries through ``procedure_id``, ``procedure_version`` and
        ``last_recommended_step`` unchanged.
      * Never reads transitions, never advances a cursor, never switches
        procedure because of ``scope`` / ``new_issue``.
    """
    # --- carry-through (Router-owned fields are preserved verbatim) ---
    new_issue = previous_state.issue
    new_problem_status = previous_state.problem_status
    new_pending = previous_state.pending_clarification
    new_topic = previous_state.clarification_topic

    actions = _dedup_actions(observation.reported_actions)

    # ------------------------------------------------------------
    # 1. Refine the issue label from observation facts (informational).
    #    NOTE: this never changes procedure_id / procedure_version and
    #    never destroys attempts (a "new issue" is informational; the
    #    Router decides whether to switch procedures).
    # ------------------------------------------------------------
    if observation.new_issue:
        stripped = observation.new_issue.strip()
        if stripped:
            new_issue = stripped

    # ------------------------------------------------------------
    # 2. Problem status from the observation (or keep previous).
    # ------------------------------------------------------------
    if observation.problem_status is not None:
        new_problem_status = observation.problem_status

    # ------------------------------------------------------------
    # 3. Record reported attempts for valid, mappable steps.
    # ------------------------------------------------------------
    step_ids = procedure.step_ids() if procedure is not None else None

    attempts: list[StepAttempt] = list(previous_state.step_attempts)
    attempts_before = len(attempts)
    next_sequence = len(attempts)

    invalid_step_reported = False
    unmappable_action_reported = False

    for action in actions:
        if action.step_id is None:
            # Unmappable action: cannot attribute it to a step.
            unmappable_action_reported = True
            continue

        if step_ids is None or action.step_id not in step_ids:
            # Unknown / invalid step id: do NOT append, do NOT guess,
            # do NOT silently drop -- flag for clarification.
            invalid_step_reported = True
            continue

        # Valid step: append one attempt (local de-dup already applied).
        attempts.append(
            StepAttempt(
                step_id=action.step_id,
                outcome=action.outcome or StepOutcome.DONE_UNKNOWN,
                sequence=next_sequence,
            )
        )
        next_sequence += 1

    # ------------------------------------------------------------
    # 4. Bare "didn't help": context resolution (NOT routing).
    #
    #    If the observation reports an ongoing problem but names no
    #    mappable step, AND there is an outstanding recommendation with no
    #    recorded outcome yet, attribute FAILED to it. Otherwise, record
    #    ambiguity via the clarification flag.
    # ------------------------------------------------------------
    attributed_no_help = False
    if (
        not actions
        and new_problem_status == ProblemStatus.ONGOING
        and previous_state.last_recommended_step is not None
        and not any(
            a.step_id == previous_state.last_recommended_step for a in attempts
        )
        and (
            step_ids is None
            or previous_state.last_recommended_step in step_ids
        )
    ):
        attempts.append(
            StepAttempt(
                step_id=previous_state.last_recommended_step,
                outcome=StepOutcome.FAILED,
                sequence=next_sequence,
            )
        )
        next_sequence += 1
        attributed_no_help = True

    # ------------------------------------------------------------
    # 5. Clarification flag.
    #
    #    Set when:
    #      * an invalid step id was reported, OR
    #      * an unmappable action was reported, OR
    #      * the user reported an ongoing problem we could not attribute
    #        and there was no outstanding recommendation to blame.
    #    A successful, fully-attributed update clears the flag.
    # ------------------------------------------------------------
    needs_clarification = (
        invalid_step_reported or unmappable_action_reported
    )

    if (
        not actions
        and not attributed_no_help
        and new_problem_status == ProblemStatus.ONGOING
        and len(attempts) == attempts_before
        and not needs_clarification
    ):
        # Ongoing problem, nothing to attribute, no new attempt recorded.
        needs_clarification = True

    if needs_clarification:
        new_pending = True
        if new_topic is None:
            new_topic = _clarification_context(previous_state)
    else:
        # A fully-attributed observation clears a stale clarification.
        new_pending = False
        new_topic = None

    return replace(
        previous_state,
        issue=new_issue,
        step_attempts=tuple(attempts),
        problem_status=new_problem_status,
        pending_clarification=new_pending,
        clarification_topic=new_topic,
    )


__all__ = [
    "StepAttempt",
    "ProcedureState",
    "initial_state",
    "update_procedure_state",
]
