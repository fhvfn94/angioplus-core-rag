"""
Turn Family -- the deterministic ENGINE-FAMILY selector (Slice 11B.1b.2).

Infrastructure only. NOT imported by any production runtime path
(server.py, alice_support_agent.py, the Support Core composition root, n8n,
Tracker, DB, retrieval, File Search).

A single PURE, total function maps ONE semantic fact -- the observation's
``request_kind`` -- to the processing FAMILY that should handle the turn:

    RequestKind.PROBLEM      -> TurnFamily.PROCEDURE_SUPPORT
    RequestKind.INFORMATION  -> TurnFamily.KNOWLEDGE_QA
    RequestKind.UNCLEAR      -> TurnFamily.CLARIFICATION

The selector reads EXACTLY ONE input dimension: ``observation.request_kind``.
It never inspects any other observation field, ProcedureState / conversation
state, the router, retrieval or the LLM.

``TurnFamily`` names a PROCESSING FAMILY, not an action or a routing decision
(it is not ANSWER / CLARIFY / ESCALATE). The Router remains the only component
that turns facts into a next action, so there is no separate "reason" field:
the already-present ``RequestKind`` IS the explanation.

No parsing happens here -- no ``getattr``, no raw-string handling, no fallback
dictionary, no regex. ``SemanticObservation`` is the typed boundary; malformed
/ failed provider output is ALREADY converted by the analyzer fail-safe into
``RequestKind.UNCLEAR``, so the safe path is:

    provider/parser failure -> SemanticObservation -> RequestKind.UNCLEAR
                            -> TurnFamily.CLARIFICATION

No I/O, no ORM, no network, no framework imports.
"""

from __future__ import annotations

from enum import Enum

from support.observation import RequestKind, SemanticObservation


class TurnFamily(str, Enum):
    """Which processing family owns the turn (a family, not an action)."""

    PROCEDURE_SUPPORT = "procedure_support"  # a reported problem -> walk a procedure
    KNOWLEDGE_QA = "knowledge_qa"            # an information/factual ask -> knowledge
    CLARIFICATION = "clarification"          # request_kind unclear -> ask, do not guess


def select_turn_family(observation: SemanticObservation) -> TurnFamily:
    """Map the ONE semantic fact ``request_kind`` to a ``TurnFamily``.

    Total and deterministic. The final arm is the closed-enum member
    ``RequestKind.UNCLEAR`` -- so the analyzer fail-safe path lands on
    ``TurnFamily.CLARIFICATION`` with NO defensive parsing here.
    """
    request_kind = observation.request_kind
    if request_kind is RequestKind.PROBLEM:
        return TurnFamily.PROCEDURE_SUPPORT
    if request_kind is RequestKind.INFORMATION:
        return TurnFamily.KNOWLEDGE_QA
    return TurnFamily.CLARIFICATION


__all__ = [
    "TurnFamily",
    "select_turn_family",
]
