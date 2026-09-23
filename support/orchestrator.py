"""
Deterministic Support Orchestrator (Slice 7) -- runtime composition ONLY.

Infrastructure only. NOT imported by any production runtime path
(server.py, alice_support_agent.py, n8n, Tracker, DB, File Search).

PURPOSE
-------
This module composes the frozen, already-total slices into ONE deterministic
support turn. It is glue and bookkeeping ONLY: it makes no decisions, assigns
no meaning and owns no policy.

RESPONSIBILITY SPLIT (frozen boundaries)
----------------------------------------
  Alice Analyzer (4)   turn ONE message into a routing-free SemanticObservation
                       (facts only; never an action).
  Reducer (2)          record WHAT the observer reported (no cursor, no routing).
  Router (3)           decide WHAT to do next (the ONLY transition reader; owns
                       the cursor CONCEPT and every routing gate).
  Knowledge (5)        evidence for an ALREADY-CHOSEN step.
  Generator (6)        render an ALREADY-MADE decision (the ONLY renderer and the
                       only owner of user-facing wording).
  Orchestrator (7)     call the above in a FIXED order and mechanically project
                       the Router's own output back into the returned state.

The orchestrator contains NO branch on ``scope``, ``human_requested``,
``problem_status``, ``new_issue`` or any step outcome -- those are Router gates.
It never invents a next action, never chooses or switches a procedure, never
re-checks escalation, never inspects transitions, and never duplicates a
user-facing template (the Generator owns all wording).

CANONICAL LIFECYCLE (each step exactly once, in order)
------------------------------------------------------
  previous_state
    -> analyze(...)                  -> SemanticObservation
    -> update_procedure_state(...)   -> working_state   (the SINGLE mutation)
    -> route(...)                    -> RoutingDecision
    -> retrieve_for_step(...)        -> evidence        (ONLY for an authorized
                                                         RECOMMEND_STEP)
    -> render_response(...)          -> ResponseResult
    -> project cursor (mechanical)   -> next_state
    -> TurnResult

Reduction happens BEFORE routing on purpose: the frozen cross-layer contracts
verify that a "didn't help" reference, a same-turn outcome and an uncertain
observation only converge correctly when the Router reads the REDUCED state
(uncertain observation -> ``pending_clarification`` -> ``CLARIFY``).

FAIL SAFETY
-----------
Every boundary above is already total: the analyzer fails safe to an unmappable
observation (-> ``CLARIFY``), the Router is total and never raises, retrieval
maps any backend failure to ``KnowledgeStatus.ERROR``, and the Generator falls
back to the deterministic procedure-owned action. This module therefore adds NO
``try/except`` and fabricates NO fallback decision.

NON-GOALS (deliberately absent)
-------------------------------
No PostgreSQL, no FSM (``support.domain``), no Tracker, no n8n, no Telegram, no
Yandex runtime/server wiring, no persistence, no logging of raw provider text,
no provider SDK, and no conversation-history ownership.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Optional, Sequence

from support.alice_analyzer import AnalyzerProvider, analyze
from support.knowledge import (
    KnowledgeBackend,
    KnowledgeResult,
    retrieve_for_step,
)
from support.observation import SemanticObservation
from support.procedure import Procedure
from support.procedure_state import (
    ProcedureState,
    initial_state,
    update_procedure_state,
)
from support.response_generator import (
    ResponseProvider,
    ResponseResult,
    render_response,
)
from support.router import (
    EscalateReason,
    RouteAction,
    RoutingDecision,
    route,
)


# ============================================================
# RESULT VALUE OBJECT
# ============================================================

@dataclass(frozen=True)
class TurnResult:
    """The immutable outcome of ONE support turn.

    Composition result ONLY -- it introduces no new vocabulary:

      * ``assistant_text`` -- the single user-facing string (== ``response.text``).
      * ``decision``       -- the Router's verdict, exposed VERBATIM. The
                              Orchestrator never post-processes it.
      * ``state``          -- the ``next_state`` for the caller's next turn.
      * ``observation``    -- the SANITIZED :class:`SemanticObservation` (facts
                              only). Raw analyzer/provider text is never exposed.
      * ``response``       -- the Generator's :class:`ResponseResult` (provenance:
                              status / reason / provider_called / step_id).
      * ``knowledge``      -- the evidence retrieved this turn, or ``None``.
      * ``escalation_reason`` -- a direct projection of ``decision`` (set only
                              for ESCALATE); no Tracker/FSM side effect is created.
    """

    assistant_text: str
    decision: RoutingDecision
    state: ProcedureState
    observation: SemanticObservation
    response: ResponseResult
    knowledge: Optional[KnowledgeResult] = None
    escalation_reason: Optional[EscalateReason] = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.assistant_text, str)
            or not self.assistant_text.strip()
        ):
            raise ValueError("assistant_text must be a non-blank string.")
        if not isinstance(self.decision, RoutingDecision):
            raise ValueError("decision must be a RoutingDecision.")
        if not isinstance(self.state, ProcedureState):
            raise ValueError("state must be a ProcedureState.")
        if not isinstance(self.observation, SemanticObservation):
            raise ValueError("observation must be a SemanticObservation.")
        if not isinstance(self.response, ResponseResult):
            raise ValueError("response must be a ResponseResult.")


# ============================================================
# MECHANICAL PROJECTIONS (no policy)
# ============================================================

def _bind_procedure_identity(
    state: ProcedureState, procedure: Optional[Procedure]
) -> ProcedureState:
    """Bind an UNBOUND state to the caller-supplied procedure (D2).

    Continuity bookkeeping ONLY. This NEVER selects a procedure and NEVER
    overwrites a non-None identity: a genuine identity mismatch must stay
    visible to the Router's identity gate (``EscalateReason.
    INTERNAL_INCONSISTENCY``), which is exactly what becomes meaningful once
    the identity is bound. Purely mechanical: it sets the stored identity of
    the procedure the caller already handed in, and changes no frozen
    reducer/router behaviour.
    """
    if procedure is None or state.procedure_id is not None:
        return state
    return replace(
        state,
        procedure_id=procedure.procedure_id,
        procedure_version=procedure.version,
    )


def _last_recommendation(
    previous_state: ProcedureState, procedure: Optional[Procedure]
) -> Optional[str]:
    """Read-only Alice context, derived ONLY from the previous state (D5).

    ``previous_state.last_recommended_step`` -> the corresponding
    :class:`ProcedureStep.action` (when the step exists) -> otherwise ``None``.

    Single source of truth: there is deliberately NO public override argument.
    """
    if procedure is None or previous_state.last_recommended_step is None:
        return None
    step = procedure.steps.get(previous_state.last_recommended_step)
    return step.action if step is not None else None


def _authorized_evidence(
    decision: RoutingDecision,
    procedure: Optional[Procedure],
    backend: Optional[KnowledgeBackend],
) -> Optional[KnowledgeResult]:
    """Retrieve evidence ONLY for an authorized RECOMMEND_STEP.

    Returns ``None`` for every other action, and for a missing backend, a
    missing procedure or an unresolvable step. The lookup guard is the ONE
    defensive check this module needs: it prevents a ``KeyError`` on an
    unknown step, after which the Generator renders its neutral
    ``INTERNAL_ERROR``. The Orchestrator owns ONLY this timing/gating (D6);
    Knowledge stays evidence-only and can never influence the decision.
    """
    if backend is None or decision.action is not RouteAction.RECOMMEND_STEP:
        return None
    if procedure is None or decision.next_step is None:
        return None
    step = procedure.steps.get(decision.next_step)
    if step is None:
        return None
    return retrieve_for_step(step, backend=backend)


def _project_next_state(
    working_state: ProcedureState, decision: RoutingDecision
) -> ProcedureState:
    """Mechanical cursor projection for the returned state (D1).

    For an authorized ``RECOMMEND_STEP`` only, record the recommended step as
    ``last_recommended_step`` so the next turn's reducer can attribute a bare
    "didn't help" and the Router can continue its walk. This adds NO routing
    policy; for every other action the state is returned unchanged.
    """
    if (
        decision.action is RouteAction.RECOMMEND_STEP
        and decision.next_step is not None
    ):
        return replace(
            working_state, last_recommended_step=decision.next_step
        )
    return working_state


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def process_support_turn(
    *,
    message: str,
    history: Sequence[Mapping[str, str]] = (),
    state: Optional[ProcedureState] = None,
    procedure: Optional[Procedure] = None,
    analyzer_provider: AnalyzerProvider,
    knowledge_backend: Optional[KnowledgeBackend] = None,
    response_provider: Optional[ResponseProvider] = None,
) -> TurnResult:
    """Compose the frozen slices into ONE deterministic support turn.

    Parameters
    ----------
    message:
        The raw current user message.
    history:
        Recent dialogue, in the Analyzer's existing ``{"role", "text"}`` shape.
        An OPAQUE pass-through (D7): never normalized, translated or extended.
    state:
        The previous :class:`ProcedureState` (``None`` -> a fresh initial state).
        Never mutated.
    procedure:
        The CURRENT procedure, supplied by the caller. The Orchestrator never
        selects or switches one; it only carries it through.
    analyzer_provider:
        REQUIRED (D4). A missing provider is a wiring error, not semantic input.
    knowledge_backend:
        Optional retrieval port. Consulted ONLY for an authorized RECOMMEND_STEP.
    response_provider:
        Optional explanation provider, forwarded to the Generator's grounded
        path. The Generator alone decides when it is consulted.

    Returns
    -------
    TurnResult
        Total: never raises for any analyzer/knowledge/provider outcome, because
        every callee is already fail-safe.

    The lifecycle is fixed and each step runs exactly once:

      analyze -> update_procedure_state -> route -> (retrieve?) -> render ->
      project cursor -> TurnResult.
    """
    # 1. Continuity inputs (the previous state is NEVER mutated).
    previous_state = state if state is not None else initial_state()
    previous_state = _bind_procedure_identity(previous_state, procedure)

    # 2. Message understanding (frozen, fail-safe Analyzer boundary).
    observation = analyze(
        message=message,
        history=history,
        procedure=procedure,
        last_recommendation=_last_recommendation(previous_state, procedure),
        provider=analyzer_provider,
    )

    # 3. Record WHAT happened -- EXACTLY once (the single mutation).
    working_state = update_procedure_state(
        previous_state, observation, procedure
    )

    # 4. Decide WHAT to do next (the ONLY decision maker; reads, never mutates).
    decision = route(working_state, observation, procedure)

    # 5. Evidence ONLY for an authorized RECOMMEND_STEP (invariant L).
    knowledge = _authorized_evidence(decision, procedure, knowledge_backend)

    # 6. Render the MADE decision (the ONLY renderer; never alters the decision).
    response = render_response(
        decision,
        procedure=procedure,
        knowledge=knowledge,
        provider=response_provider,
    )

    # 7. Mechanical projection of the Router's own output into the next state.
    next_state = _project_next_state(working_state, decision)

    return TurnResult(
        assistant_text=response.text,
        decision=decision,
        state=next_state,
        observation=observation,
        response=response,
        knowledge=knowledge,
        escalation_reason=decision.escalate_reason,
    )


__all__ = [
    "TurnResult",
    "process_support_turn",
]
