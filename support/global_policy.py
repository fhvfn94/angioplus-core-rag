"""
Global Semantic Policy -- the ONE owner of the UNIVERSAL terminal conditions
(Slice 11B.1b.4d).

Infrastructure only. NOT imported by any production runtime path (server.py,
alice_support_agent.py, n8n, Tracker, DB, retrieval, File Search). Since Slice
11B.2 it IS applied by the dispatcher (``support.turn_dispatch``) AFTER the
ONE analysis and BEFORE family selection; it is NOT wired into any runtime.

WHAT THIS MODULE IS
-------------------
A single PURE, TOTAL, deterministic function that inspects TWO universal
semantic facts of an already-produced
:class:`~support.observation.SemanticObservation` and answers ONE question:
"must this turn be terminated BEFORE any capability/family dispatch?"

    SemanticObservation
      -> evaluate_global_policy(...)
      -> GlobalAction.CONTINUE | ESCALATE | OUT_OF_SCOPE

Precedence (first match wins, mirroring the frozen Router gates 1 -> 2):

    1. scope == OUT_OF_SCOPE   -> GlobalAction.OUT_OF_SCOPE
    2. human_requested         -> GlobalAction.ESCALATE
    3. otherwise               -> GlobalAction.CONTINUE

READS EXACTLY TWO FACTS
-----------------------
``scope`` and ``human_requested`` ONLY. It deliberately does NOT read
``request_kind`` (that is the family-selector's fact), ``problem_status``
(procedure-specific: RESOLVED closes a procedure walk, it is NOT a global
terminal), ``new_issue`` (procedure-specific), ``reported_actions`` (reducer
input) or ``clarification_facts`` (informational). No raw text, no regex, no
lexical heuristics, no second classifier.

WHY IT IS GLOBAL
----------------
The frozen Router evaluates ``scope`` / ``human_requested`` -- but only from
INSIDE the procedure pipeline. Knowledge QA and Clarification never call the
Router, so today those same facts would behave differently by family
(``OUT_OF_SCOPE + INFORMATION`` or ``human_requested + INFORMATION`` would
fall through to Knowledge QA). Hoisting the two universal facts into ONE
earlier owner removes that family-dependent divergence.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
  * NO user-facing prose and NO rendering (a later projector reuses the
    existing response generator; the generator stays the ONLY wording owner);
  * NO Tracker, NO FSM, NO operator-case creation, NO HTTP, NO channel
    metadata, NO case_action, NO confidence / score / threshold;
  * NO LLM, NO network, NO DB, NO Procedure, NO Knowledge, NO retrieval;
  * NO import of ``support.router`` / ``support.turn_family`` / the
    orchestrator: global policy is not coupled to procedure routing or to
    capability selection.

RELATIONSHIP TO TURN FAMILY / ROUTER
------------------------------------
Global policy runs conceptually BEFORE :func:`support.turn_family.
select_turn_family` (a terminal preempts capability choice); it is NOT part
of the family selector and does NOT know about families. The Router keeps
gate 3+ (procedure-specific policy, e.g. ``RESOLVED`` -> RESOLVE).

TRANSITIONAL NOTE (legacy Router gates 1-2)
-------------------------------------------
The frozen Router still owns its OWN gates 1-2 for the LEGACY procedure path
(``process_support_turn`` / ``execute_persisted_turn``). On the NEW path the
dispatcher applies this module's GLOBAL policy BEFORE family selection, so the
two universal facts (``scope`` / ``human_requested``) have ONE owner there.
Removing the duplicated legacy gates is deliberately DEFERRED until the legacy
path is retired: ``support/router.py`` is UNCHANGED in Slice 11B.2.

RESOLVED + procedure is None (Slice 11B.2 decision)
---------------------------------------------------
Global policy ignores ``problem_status`` entirely, so it never resolves such a
turn. On the new path a ``request_kind=PROBLEM`` turn with ``procedure=None``
returns ``NoProcedureDispatch`` (-> ``NO_PROCEDURE``) and does NOT import the
legacy Router to resolve it; the ``RESOLVED`` + no-procedure lifecycle /
acknowledgement case is DEFERRED (NOT solved by importing the legacy Router).
An ``INFORMATION`` turn with ``procedure=None`` still reaches Knowledge QA.
"""

from __future__ import annotations

from enum import Enum

from support.observation import ObservationScope, SemanticObservation


class GlobalAction(str, Enum):
    """The ONE global terminal verdict for a turn. Closed, exactly 3 values.

      * ``CONTINUE``     -- no universal terminal condition: proceed to
                            capability / family dispatch. Carries no reason
                            (the absence of a terminal condition is not one).
      * ``ESCALATE``     -- an explicit human request (its ONLY global cause,
                            so the reason is implied and the later projector
                            maps it 1:1 to the existing
                            ``EscalateReason.USER_REQUESTED_HUMAN``).
      * ``OUT_OF_SCOPE`` -- the message is not about the product at all.

    There is deliberately NO separate ``reason`` field/enum: with exactly one
    trigger per terminal action, a reason would carry zero information. The
    existing string values (``"escalate"`` / ``"out_of_scope"``) are reused
    for continuity WITHOUT importing ``support.router``.
    """

    CONTINUE = "continue"
    ESCALATE = "escalate"
    OUT_OF_SCOPE = "out_of_scope"


def evaluate_global_policy(observation: SemanticObservation) -> GlobalAction:
    """Return the global terminal verdict for ONE observation. Pure and total.

    Reads EXACTLY ``observation.scope`` and ``observation.human_requested``.
    Never raises; never mutates; deterministic. ``OUT_OF_SCOPE`` is the OUTER
    gate and wins over an explicit human request (the existing, audit-approved
    outer precedence).
    """
    if observation.scope is ObservationScope.OUT_OF_SCOPE:
        return GlobalAction.OUT_OF_SCOPE
    if observation.human_requested:
        return GlobalAction.ESCALATE
    return GlobalAction.CONTINUE


__all__ = [
    "GlobalAction",
    "evaluate_global_policy",
]
