"""
Clarification Responder -- the deterministic, LLM-free CLARIFICATION family
(Slice 11B.1b.4b).

Infrastructure only. NOT imported by any production runtime path
(server.py, alice_support_agent.py, the Support Core composition root, n8n,
Tracker, DB, retrieval, File Search).

RESPONSIBILITY (exactly one)
----------------------------
When the semantic request kind is UNCLEAR -- i.e. the analyzer could not
safely read WHAT the user wants -- produce ONE neutral, deterministic,
user-facing prompt that asks the user to clarify. It is the THIRD engine
family, alongside Procedure Support and Knowledge QA:

    TurnFamily.PROCEDURE_SUPPORT -> the Procedure Engine
    TurnFamily.KNOWLEDGE_QA      -> the Knowledge QA engine
    TurnFamily.CLARIFICATION     -> THIS responder

WHAT IT IS NOT
--------------
Not escalation, not handoff, not an operator request, not a knowledge
answer, not procedure selection, not procedure continuation, not a routing
decision, not error recovery and not a retry policy. The text makes NO claim
that the system cannot help, that no information exists, that an operator
will be contacted, that a support case will be created, that the request is
out of scope, or that an error occurred.

BOUNDARY
--------
Pure and total: NO LLM/provider, NO network, NO DB, NO procedure, NO
knowledge, NO router, NO FSM/Tracker, NO observation and NO
TurnFamily/RequestKind. It does not decide WHEN it is called -- the future
family dispatcher owns that decision. It never imports its siblings' types.
"""

from __future__ import annotations

from dataclasses import dataclass

# The ONE neutral template. Deliberately a single string: no template set,
# no random selection, no confidence/score and no reason taxonomy. It must
# read naturally whether the user is ASKING for information or REPORTING a
# problem, so it names both possibilities without presupposing either.
_CLARIFICATION_TEXT = (
    "Уточните, пожалуйста, что именно вы хотите узнать "
    "или какая проблема возникла."
)


@dataclass(frozen=True)
class ClarificationResult:
    """Immutable result of the CLARIFICATION family.

    One field: ``text`` is ALWAYS a non-blank, user-facing string. There is
    deliberately NO status, NO reason and NO provider flag -- the family has
    exactly one outcome.
    """

    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a non-blank string.")


def clarify() -> ClarificationResult:
    """Return the ONE deterministic clarification prompt. Pure and total."""
    return ClarificationResult(text=_CLARIFICATION_TEXT)


__all__ = [
    "ClarificationResult",
    "clarify",
]
