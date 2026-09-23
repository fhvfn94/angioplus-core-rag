"""
Semantic Observation -- what an LLM (Alice) understood from a user message.

Infrastructure only. NOT imported by any production runtime path
(server.py, alice_support_agent.py, n8n, Tracker, DB, File Search).

An observation describes FACTS ONLY:

  * what the message appears to be about (scope)
  * which procedure step(s) the user reported acting on
  * the reported outcome(s) of those actions
  * the reported problem status
  * whether the user explicitly asked for a human
  * free-form clarification facts
  * an optionally reported new issue

It is ROUTING-FREE: it carries no candidate action, no ANSWER/CLARIFY/
ESCALATE decision, and no verdict on "already attempted". Turning facts
into a next action is the Router's job (a later slice).

This module has no I/O, no ORM, no framework imports, so it can be
unit-tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class _StrEnum(str, Enum):
    """Python 3.10 compatible str-enum (values are plain strings)."""

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return str(self.value)


class ObservationScope(_StrEnum):
    """What the message appears to be about (a reading, not a decision)."""

    IN_SCOPE = "in_scope"          # plausibly about a supported procedure
    OUT_OF_SCOPE = "out_of_scope"  # unrelated topic / non-support
    UNCLEAR = "unclear"            # cannot tell


class ProblemStatus(_StrEnum):
    """Problem-level fact as reported by the user on its own axis.

    Deliberately NOT the same concept as the conversation FSM's lifecycle:
    this only tracks whether the *technical problem* is still present.
    """

    UNKNOWN = "unknown"    # nothing heard yet / cannot determine
    ONGOING = "ongoing"    # problem still present after the last action
    RESOLVED = "resolved"  # user says the problem is gone


class StepOutcome(_StrEnum):
    """Reported result of performing a step."""

    FAILED = "failed"              # user did it and the problem persists
    SUCCEEDED = "succeeded"        # user did it and it fixed the problem
    SKIPPED = "skipped"            # user says already done / cannot do / N/A
    DONE_UNKNOWN = "done_unknown"  # user did it, result not yet clear


@dataclass(frozen=True)
class ReportedAction:
    """One action the user reports having performed in a single message.

    ``step_id`` is a procedure step id when the LLM could map the reported
    action to one, or ``None`` when it could not. ``outcome`` is the
    reported result, or ``None`` when the user did not state one.

    This is a FACT object: it makes no decision and carries no candidate
    action.
    """

    step_id: Optional[str]
    outcome: Optional[StepOutcome]


@dataclass(frozen=True)
class SemanticObservation:
    """Routing-free reading of ONE user message (plus recent context).

    Contains facts only. The deterministic reducer consumes this to update
    :class:`support.procedure_state.ProcedureState`, and, later, the Router
    will consume both to decide the next action. The LLM must NOT output
    any action or routing decision here.
    """

    # What the message appears to be about.
    scope: ObservationScope

    # All actions the user reported performing in this single message.
    # Covers single-action, multi-action and unmappable cases uniformly:
    #   one action   -> one element
    #   many actions -> many elements
    #   unmappable   -> ReportedAction(step_id=None, outcome=...)
    reported_actions: tuple[ReportedAction, ...]

    # Problem-level fact reported by the user, or None if not stated.
    problem_status: Optional[ProblemStatus]

    # Did the user explicitly ask for a human? A FACT, not an ESCALATE
    # decision -- the Router turns this into an action later.
    human_requested: bool

    # Free-form facts useful for clarification (model, LED color, ...).
    # Kept unstructured on purpose in this slice.
    clarification_facts: tuple[str, ...]

    # A newly mentioned problem label, if the user appears to raise a
    # different problem. INFORMATIONAL ONLY: this never by itself mutates
    # or destroys the current procedure state; switching procedures is a
    # Router / orchestrator decision.
    new_issue: Optional[str]


__all__ = [
    "ObservationScope",
    "ProblemStatus",
    "StepOutcome",
    "ReportedAction",
    "SemanticObservation",
]
