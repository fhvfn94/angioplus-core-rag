"""
Turn Projection -- project typed GLOBAL terminal results into the EXISTING
deterministic user-facing wording (Slice 11B.2).

Infrastructure only. NOT imported by any production runtime path (server.py,
alice_support_agent.py, yandex/realtime, n8n, Tracker, DB, retrieval). It is
imported by :mod:`support.turn_dispatch` so the dispatcher can render a GLOBAL
terminal WITHOUT the dispatcher itself importing the legacy Router.

WHY THIS MODULE EXISTS
----------------------
The ONE owner of user-facing wording is the deterministic Response Generator
(``support.response_generator``), which renders a
:class:`~support.router.RoutingDecision`. Global Semantic Policy
(``support.global_policy``) deliberately does NOT import the Router and does
NOT produce prose -- it returns a closed
:class:`~support.global_policy.GlobalAction` verdict. The dispatcher must not
import the Router either.

This module is the SMALL, isolated bridge: it maps a GLOBAL terminal
``GlobalAction`` to the EXISTING ``RoutingDecision`` the generator already
renders, and returns that generator text VERBATIM. It owns NO wording of its
own -- the generator stays the single wording owner -- and it introduces no new
vocabulary.

MAPPING (1:1, audit-approved)
-----------------------------
  GlobalAction.ESCALATE     -> RouteAction.ESCALATE with
                               EscalateReason.USER_REQUESTED_HUMAN
                               (the ONLY global cause of an ESCALATE)
  GlobalAction.OUT_OF_SCOPE -> RouteAction.OUT_OF_SCOPE

``GlobalAction.CONTINUE`` is NOT a terminal and is rejected: a continuing turn
is never projected here (it goes to family dispatch).

No I/O: no network, no provider, no DB, no LLM, no ProcedureState. Pure and
total for the two terminal actions.
"""

from __future__ import annotations

from support.global_policy import GlobalAction
from support.response_generator import render_response
from support.router import EscalateReason, RouteAction, RoutingDecision


def project_global_terminal(action: GlobalAction) -> str:
    """Render the EXISTING deterministic wording for a GLOBAL terminal.

    Returns the Response Generator's text for the matching, already-frozen
    decision. Raises ``ValueError`` for a non-terminal action
    (``GlobalAction.CONTINUE``), which must never be projected here.
    """
    if action is GlobalAction.ESCALATE:
        decision = RoutingDecision(
            action=RouteAction.ESCALATE,
            escalate_reason=EscalateReason.USER_REQUESTED_HUMAN,
        )
    elif action is GlobalAction.OUT_OF_SCOPE:
        decision = RoutingDecision(action=RouteAction.OUT_OF_SCOPE)
    else:
        raise ValueError(
            f"{action!r} is not a global terminal and cannot be projected."
        )
    return render_response(decision).text


__all__ = ["project_global_terminal"]
