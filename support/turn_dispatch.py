"""
Turn Dispatch -- the SINGLE family-dispatch boundary (Slice 11B.1b.4c),
extended in Slice 11B.2 to run GLOBAL SEMANTIC POLICY before family selection.

Infrastructure only. NOT imported by any production runtime path (server.py,
alice_support_agent.py, n8n, Tracker, DB, retrieval, File Search). Since Slice
11B.2 it IS composed by the Support Core application boundary
(``support.support_core``), which owns procedure resolution; runtime / channel
wiring remains a LATER slice.

WHAT THIS MODULE IS
-------------------
ONE incoming support turn is understood ONCE, its GLOBAL terminal gate is
evaluated, its processing FAMILY is chosen DETERMINISTICALLY, and EXACTLY ONE
capability runs:

    message, state, procedure
      -> analyze_support_turn(...)            (EXACTLY ONE analysis)
      -> SemanticObservation
      -> evaluate_global_policy(observation)  (the ONLY global terminal gate)
           |- ESCALATE / OUT_OF_SCOPE -> project + GlobalTerminalDispatch
           |- CONTINUE                -> select_turn_family(observation)
                                         (the ONLY family decision)
                -> EXACTLY ONE family branch
                     |- PROCEDURE_SUPPORT -> SupportTurnExecutor
                     |                        .execute_persisted_turn_from_observation
                     |                        (the SAME observation; single CAS writer)
                     |                        -> ExecutedDispatch  (a NEW turn ran)
                     |                        -> ReplayedDispatch  (an HL-2b idempotent
                     |                           replay: the persisted winner, NO new text)
                     |- KNOWLEDGE_QA      -> answer_question(...)
                     |- CLARIFICATION     -> clarify()

SINGLE-ANALYSIS INVARIANT (non-negotiable)
------------------------------------------
The analyzer runs AT MOST ONCE per turn. The procedure branch consumes the
SAME observation VERBATIM through the executor's precomputed-observation
path, which has NO analyzer argument, so ``analyze -> executor -> analyze``
cannot be expressed. KNOWLEDGE_QA and CLARIFICATION never touch the analyzer
or the procedure executor.

OWNS / DOES NOT OWN
-------------------
OWNS: the ONE observation, deterministic family selection, delegation to
exactly one family.
DOES NOT OWN: procedure reducer logic, procedure routing policy, the CAS
implementation, knowledge retrieval implementation, response generation
implementation, procedure resolution / selection (the caller supplies the
already-resolved ``state`` / ``procedure``), HTTP / runtime / channel
behaviour, FSM / Tracker, message persistence, idempotency. The dispatcher
FORWARDS one caller-supplied OPAQUE ``execution_id`` to the procedure executor
and never interprets it: the idempotent CAS semantics stay owned by HL-2b
(:mod:`support.turn_executor` / ``support.procedure_state_store``).

GLOBAL SEMANTIC POLICY (wired here -- Slice 11B.2)
--------------------------------------------------
Immediately AFTER the ONE analysis and BEFORE family selection, the dispatcher
asks the SINGLE global owner (:func:`support.global_policy.
evaluate_global_policy`) whether the turn must TERMINATE before any capability
runs. A terminal verdict -- ``ESCALATE`` (explicit human request) or
``OUT_OF_SCOPE`` -- preempts family dispatch and returns a typed
:class:`GlobalTerminalDispatch`; NO family capability executes. The dispatcher
does NOT duplicate that policy and does NOT import the legacy Router: the
EXISTING wording for a terminal is projected by :mod:
``support.turn_projection`` (the Response Generator stays the single wording
owner).

NO-PROCEDURE OUTCOME
--------------------
A PROBLEM turn with NO resolved procedure candidate runs NO capability: the
dispatcher returns :class:`NoProcedureDispatch` -- no wording invented, no
executor call without a ``Procedure``, and NOT re-classified as CLARIFICATION
(the request kind was PROBLEM, not UNCLEAR). The application boundary maps it
to its EXISTING ``NO_PROCEDURE`` outcome.

PROCEDURE REPLAY (HL-2c.1)
--------------------------
The executor is idempotent (HL-2b): repeating the SAME ``execution_id``
returns the ALREADY persisted winner with ``replayed`` ``True`` and ``turn``
``None`` (either BEFORE PROCESS, or after losing a same-id race at the CAS).
A replay is NOT a second user-visible answer event, so this boundary does NOT
re-render, guess or invent assistant text: it returns the typed, internal
:class:`ReplayedDispatch`, which carries the persisted winner and has NO
``text`` field. A NEW turn keeps returning :class:`ExecutedDispatch` with its
real, non-blank ``text`` (that invariant is NOT relaxed). Neither the family
decision nor the ONE analysis changes: the typed ``replayed`` flag is the
ONLY authoritative discriminator.

ASYNC
-----
``dispatch`` is async because the procedure branch awaits the async executor.
The analyzer provider and the Knowledge QA path are SYNC and network-backed;
they are offloaded with ``asyncio.to_thread`` so they never block the event
loop. ``clarify()`` is pure and is called directly.

ERROR SEMANTICS
---------------
No ``try/except`` and NO catch-all policy is added here. Every branch is
already total except the executor's optimistic CAS, whose
:class:`~support.procedure_state_store.ProcedureStateConflict` propagates
UNCHANGED (no retry, no overwrite). Knowledge statuses (``ANSWERED`` /
``NOT_ANSWERED`` / ``INTERNAL_ERROR``) are preserved intact; the analyzer
fail-safe (``request_kind=UNCLEAR``) naturally lands on CLARIFICATION.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Union

from support.alice_analyzer import AnalyzerProvider
from support.clarification import ClarificationResult, clarify
from support.global_policy import GlobalAction, evaluate_global_policy
from support.knowledge import KnowledgeBackend
from support.knowledge_qa import (
    KnowledgeQAProvider,
    KnowledgeQAResult,
    answer_question,
)
from support.observation import SemanticObservation
from support.orchestrator import analyze_support_turn
from support.procedure import Procedure
from support.procedure_state import ProcedureState
from support.response_generator import ResponseProvider
from support.turn_executor import PersistedTurnResult, SupportTurnExecutor
from support.turn_family import TurnFamily, select_turn_family
from support.turn_projection import project_global_terminal


# ============================================================
# DISPATCH RESULT (four thin types, no ``None`` policy signal)
# ============================================================

@dataclass(frozen=True)
class ExecutedDispatch:
    """A capability DID run.

    The family-native result is preserved INTACT behind ``family`` and
    ``text`` is that family's single user-facing string. ``text`` is ALWAYS
    non-blank (enforced) and EXACTLY the one payload matching ``family`` is
    set -- so an executed result can never be text-less and ``None`` can never
    act as a hidden policy signal.

    No HTTP / case_action / FSM / Tracker / channel / dedup / runtime
    vocabulary is introduced: this is internal support-domain output only.
    """

    family: TurnFamily
    text: str
    procedure_result: Optional[PersistedTurnResult] = None
    knowledge_result: Optional[KnowledgeQAResult] = None
    clarification_result: Optional[ClarificationResult] = None

    def __post_init__(self) -> None:
        if not isinstance(self.family, TurnFamily):
            raise ValueError("family must be a TurnFamily.")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a non-blank string.")

        expected = {
            TurnFamily.PROCEDURE_SUPPORT: PersistedTurnResult,
            TurnFamily.KNOWLEDGE_QA: KnowledgeQAResult,
            TurnFamily.CLARIFICATION: ClarificationResult,
        }[self.family]
        payloads = {
            "procedure_result": self.procedure_result,
            "knowledge_result": self.knowledge_result,
            "clarification_result": self.clarification_result,
        }
        set_names = [name for name, value in payloads.items() if value is not None]
        if len(set_names) != 1 or not isinstance(
            payloads[set_names[0]], expected
        ):
            raise ValueError(
                f"{self.family} requires exactly its one matching payload."
            )


@dataclass(frozen=True)
class ReplayedDispatch:
    """A PROCEDURE_SUPPORT turn that was an HL-2b IDEMPOTENT REPLAY.

    The durable CAS had ALREADY advanced the state under THIS execution id
    (BEFORE PROCESS, or after losing a same-id race at the CAS), so NO new
    turn was computed and NO assistant text exists for THIS call. The
    ACTUALLY persisted winner is carried VERBATIM inside ``procedure_result``.

    There is deliberately NO ``text`` field: this is NOT a second
    user-visible answer event and this boundary never fabricates wording. An
    empty, placeholder, reconstructed or re-rendered string would lie about
    having produced a new text result. It is INTERNAL engine vocabulary only
    -- NOT an HTTP response, NOT a :class:`~support.runtime_outcome.RuntimeTurn`,
    NOT a lifecycle projection, NOT an FSM state, NOT a
    ``SupportTurnOutcome`` status and NOT a new user-facing response type.

    It is the sibling of :class:`ExecutedDispatch`, whose non-blank ``text``
    invariant stays INTACT: an executed result ALWAYS carries real text, a
    replayed result ALWAYS carries the persisted winner instead.
    """

    family: TurnFamily
    procedure_result: PersistedTurnResult

    def __post_init__(self) -> None:
        if self.family is not TurnFamily.PROCEDURE_SUPPORT:
            raise ValueError(
                "a replay can only be a PROCEDURE_SUPPORT result."
            )
        if not isinstance(self.procedure_result, PersistedTurnResult):
            raise ValueError(
                "procedure_result must be a PersistedTurnResult."
            )
        # ``PersistedTurnResult`` ALREADY guarantees
        # ``replayed=True <-> turn is None``; this boundary adds no second,
        # competing rule -- it only refuses a non-replay here.
        if not self.procedure_result.replayed:
            raise ValueError(
                "ReplayedDispatch requires a replayed PersistedTurnResult "
                "(the persisted winner; ``turn`` is already ``None``)."
            )


@dataclass(frozen=True)
class GlobalTerminalDispatch:
    """A GLOBAL terminal preempted every capability/family -- Slice 11B.2.

    The ONE analysis already ran; Global Semantic Policy then terminated the
turn BEFORE family selection, so NO family ran, NO procedure executor was
called and NO ProcedureState was written. ``action`` is the closed
:class:`~support.global_policy.GlobalAction` verdict (``ESCALATE`` or
``OUT_OF_SCOPE``) and ``text`` is the EXISTING deterministic wording produced
by :func:`support.turn_projection.project_global_terminal` (the Response
Generator stays the single wording owner).
    """

    action: GlobalAction
    text: str

    def __post_init__(self) -> None:
        if self.action not in (
            GlobalAction.ESCALATE,
            GlobalAction.OUT_OF_SCOPE,
        ):
            raise ValueError(
                "action must be a global terminal (ESCALATE / OUT_OF_SCOPE)."
            )
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a non-blank string.")


@dataclass(frozen=True)
class NoProcedureDispatch:
    """A PROBLEM turn resolved to NO procedure candidate.

    NO capability ran: no executor call (there is no ``Procedure``), no
    wording invented, no ``None`` policy signal. It is deliberately NOT the
    CLARIFICATION family. The application boundary maps it to its EXISTING
    ``NO_PROCEDURE`` outcome (nothing persisted). Carries no fields -- a
    type-level marker.
    """


# What ``SupportTurnDispatcher.dispatch`` returns.
DispatchResult = Union[
    ExecutedDispatch,
    ReplayedDispatch,
    GlobalTerminalDispatch,
    NoProcedureDispatch,
]


# ============================================================
# DISPATCHER
# ============================================================

class SupportTurnDispatcher:
    """Understand ONE turn ONCE, choose a family, run exactly one capability.

    Dependency-injected and side-effect-free beyond the ONE delegated
    family call. It owns NO persistence (the executor does) and NO policy
    beyond the deterministic :func:`select_turn_family`. The caller supplies
    the already-resolved ``state`` (read-only context) and ``procedure``
    (resolved candidate) -- there is NO ProcedureCatalog / resolver here.
    """

    def __init__(
        self,
        *,
        analyzer_provider: AnalyzerProvider,
        procedure_executor: SupportTurnExecutor,
        knowledge_backend: KnowledgeBackend,
        knowledge_qa_provider: KnowledgeQAProvider,
        response_provider: Optional[ResponseProvider] = None,
    ) -> None:
        self._analyzer_provider = analyzer_provider
        self._procedure_executor = procedure_executor
        self._knowledge_backend = knowledge_backend
        self._knowledge_qa_provider = knowledge_qa_provider
        self._response_provider = response_provider

    async def dispatch(
        self,
        *,
        conversation_id: str,
        message: str,
        history: Sequence[Mapping[str, str]] = (),
        state: Optional[ProcedureState] = None,
        procedure: Optional[Procedure] = None,
        execution_id: Optional[str] = None,
    ) -> DispatchResult:
        """Analyze ONCE, select the family, and delegate to exactly one.

        ``state`` / ``procedure`` are the already-resolved read-only context
        and candidate produced by the application boundary; they are NEVER
        mutated and never re-derived here. ``history`` is an opaque
        pass-through to the analyzer.

        ``execution_id`` is an OPTIONAL, OPAQUE idempotency identity. It is
        forwarded UNCHANGED to the procedure executor ONLY and is NEVER
        parsed, normalised, hashed, truncated, combined, generated, validated
        semantically or inferred from ``conversation_id`` / ``message`` /
        ``history``. No other family can observe it.
        """
        # 1. The ONE analysis. The provider is sync + network-backed, so it
        #    is offloaded to a thread to keep the event loop free.
        observation: SemanticObservation = await asyncio.to_thread(
            analyze_support_turn,
            message=message,
            history=history,
            state=state,
            procedure=procedure,
            analyzer_provider=self._analyzer_provider,
        )

        # 2. The ONE global terminal gate (reads scope + human_requested),
        #    BEFORE any capability. A terminal preempts family selection and
        #    runs no family at all; its EXISTING wording is projected.
        global_action = evaluate_global_policy(observation)
        if global_action is not GlobalAction.CONTINUE:
            return GlobalTerminalDispatch(
                action=global_action,
                text=project_global_terminal(global_action),
            )

        # 3. The ONLY family decision (reads request_kind alone).
        family = select_turn_family(observation)

        # 4. Delegate to EXACTLY ONE family, using the SAME observation.
        if family is TurnFamily.PROCEDURE_SUPPORT:
            return await self._dispatch_procedure(
                conversation_id, observation, procedure, execution_id
            )
        if family is TurnFamily.KNOWLEDGE_QA:
            return await self._dispatch_knowledge(message)
        return self._dispatch_clarification()

    # ------------------------------------------------------------
    # FAMILY BRANCHES
    # ------------------------------------------------------------

    async def _dispatch_procedure(
        self,
        conversation_id: str,
        observation: SemanticObservation,
        procedure: Optional[Procedure],
        execution_id: Optional[str] = None,
    ) -> DispatchResult:
        """Run the procedure pipeline from the SAME observation.

        Never call the executor without a ``Procedure``: a missing candidate
        is a NO-PROCEDURE outcome, not a fabricated turn.

        The executor's TYPED ``replayed`` flag is the authoritative
        discriminator, so a pre-PROCESS same-id hit and a post-CAS same-id
        race are handled IDENTICALLY (and ``turn`` is never dereferenced on
        the replay path).
        """
        if procedure is None:
            return NoProcedureDispatch()

        kwargs: dict = {
            "conversation_id": conversation_id,
            "observation": observation,
            "procedure": procedure,
            "knowledge_backend": self._knowledge_backend,
            "response_provider": self._response_provider,
        }
        if execution_id is not None:
            # HL-2b's compatibility convention: supply the OPAQUE id ONLY
            # when it is present, so a caller that does not use idempotency
            # keeps the EXACT previous executor call shape.
            kwargs["execution_id"] = execution_id

        persisted = (
            await self._procedure_executor.execute_persisted_turn_from_observation(
                **kwargs
            )
        )

        # A replay means the durable state was ALREADY advanced under this
        # execution id: no new turn exists, so NO text is invented. The
        # persisted winner is projected verbatim.
        if persisted.replayed:
            return ReplayedDispatch(
                family=TurnFamily.PROCEDURE_SUPPORT,
                procedure_result=persisted,
            )
        return ExecutedDispatch(
            family=TurnFamily.PROCEDURE_SUPPORT,
            text=persisted.turn.assistant_text,
            procedure_result=persisted,
        )

    async def _dispatch_knowledge(self, message: str) -> DispatchResult:
        """Answer the factual question. Retrieval + provider are sync, so the
        whole QA call is offloaded to a thread. No procedure, no persistence."""
        result = await asyncio.to_thread(
            answer_question,
            message,
            backend=self._knowledge_backend,
            provider=self._knowledge_qa_provider,
        )
        return ExecutedDispatch(
            family=TurnFamily.KNOWLEDGE_QA,
            text=result.text,
            knowledge_result=result,
        )

    def _dispatch_clarification(self) -> DispatchResult:
        """Return the deterministic clarification prompt (pure, no I/O)."""
        result = clarify()
        return ExecutedDispatch(
            family=TurnFamily.CLARIFICATION,
            text=result.text,
            clarification_result=result,
        )


__all__ = [
    "DispatchResult",
    "ExecutedDispatch",
    "GlobalTerminalDispatch",
    "NoProcedureDispatch",
    "ReplayedDispatch",
    "SupportTurnDispatcher",
]
