"""
Support Core -- production dependency COMPOSITION (Slice 11A.3a) and the
SINGLE support-turn EXECUTION flow (Slices 11A.3b -> 11B.2).

Infrastructure only. This module is the single place where the
ALREADY-EXISTING Support Core building blocks are constructed and held
together, and the single place where ONE support turn is driven end to end.

    conversation_id + raw message + history
      -> ONE short, read-only read of the persisted ProcedureState
      -> resolve the CURRENT Procedure CANDIDATE (frozen collaborators)
      -> SupportTurnDispatcher.dispatch(...)
           -> analyze ONCE -> evaluate_global_policy(...)
                -> ESCALATE / OUT_OF_SCOPE -> GlobalTerminalDispatch
                -> CONTINUE -> select_turn_family(...)
                     -> PROCEDURE_SUPPORT -> SupportTurnExecutor
                     |                       .execute_persisted_turn_from_observation
                     |                       (persisted; the single CAS writer)
                     |                       -> ExecutedDispatch (a NEW turn ran)
                     |                       -> ReplayedDispatch (HL-2b replay: the
                     |                          persisted winner, NO new text)
                     -> KNOWLEDGE_QA      -> KnowledgeQAResult
                     -> CLARIFICATION     -> ClarificationResult
                     -> (PROBLEM + no candidate) -> NoProcedureDispatch
      -> SupportTurnOutcome (mechanical projection)

It stays a COMPOSITION / APPLICATION BOUNDARY: it owns no wording, no
policy and no decision logic. Global Semantic Policy is owned by
``support.global_policy`` and applied INSIDE the dispatcher (after the ONE
analysis); family selection is owned by ``support.turn_family``; the durable
state lifecycle is owned by ``support.turn_executor``.

PROCEDURE RESOLUTION (delegated -- this module has no selection algorithm)
-------------------------------------------------------------------------
  * COLD START (no persisted ``procedure_id``) -> ``ProcedureCatalog.select(
    message)``, the ONLY cold-start mapper: exactly one qualifying procedure
    -> that candidate, otherwise ``None``. The candidate is then handed to the
    dispatcher, which STILL analyses the message: a PROBLEM turn with no
    candidate is the dispatcher's ``NoProcedureDispatch`` (-> ``NO_PROCEDURE``),
    while an INFORMATION turn is answered by Knowledge QA even with no
    candidate. No ranking, no best-match, no default procedure, no ``latest``
    and no model-chosen ``procedure_id``.
  * CONTINUATION (persisted ``procedure_id`` + ``procedure_version``) ->
    ``ProcedureCatalog.get(procedure_id, procedure_version)`` -- the EXACT
    pair only. ``select`` is deliberately NOT consulted again for a bound
    conversation, the version is never upgraded, never fallen back to another
    version and never replaced by ``latest``, and the identity is never
    inferred from the current text. An un-rehydratable identity ->
    ``PROCEDURE_NOT_FOUND``.

STATE SOURCE (exactly one writer)
---------------------------------
  * :class:`~support.turn_executor.SupportTurnExecutor` keeps owning the
    durable READ -> PROCESS -> CAS WRITE lifecycle. ``SupportCore`` NEVER
    writes ``ProcedureState``, never commits/rolls back and opens no second
    persistence path.
  * Resolving a BOUND identity needs the persisted identity BEFORE the
    executor runs, so ``SupportCore`` performs ONE additional short,
    read-only, LOCK-FREE ``ProcedureStateStore.read`` and closes it before
    any provider call. That READ is deliberately duplicated (the executor
    reads again for its own CAS); the WRITE is not: there is exactly one
    writer, and it is the executor.
  * No transaction is held across any analyzer / knowledge / generator call.

CONCURRENCY
-----------
  A CAS miss surfaces as the EXISTING typed
  :class:`~support.procedure_state_store.ProcedureStateConflict`, propagated
  UNCHANGED: no automatic retry, no provider re-run, no overwrite of newer
  state, and no duplicate outcome vocabulary invented here.

IDEMPOTENT REPLAY (HL-2c.1)
---------------------------
  ``handle_turn`` accepts an OPTIONAL, OPAQUE ``execution_id`` and forwards it
  UNCHANGED to the dispatcher (and on to the executor); it is never parsed,
  normalised, hashed, generated, truncated, combined, derived or inferred from
  the conversation / message / history, and it is never STORED on this
  boundary. When the executor reports an HL-2b idempotent REPLAY, the
  dispatcher returns its typed
  :class:`~support.turn_dispatch.ReplayedDispatch` and this boundary projects
  it onto the EXISTING ``TURN_EXECUTED`` status carrying the ACTUALLY persisted
  :class:`~support.turn_executor.PersistedTurnResult` (``replayed`` ``True``,
  ``turn`` ``None``). NO new status is invented and NO user-facing wording is
  produced for a replay: this boundary never re-renders, guesses or fabricates
  assistant text.

:class:`SupportCore` itself remains an injectable, side-effect-free
container:

  * NO side effects at import time;
  * NO network call, NO credential read, NO provider call and NO database
    connection at construction time -- every heavyweight dependency is
    resolved LAZILY by the adapter that owns it (and by the turn flow);
  * EVERY dependency is INJECTABLE, so focused tests compose the core with
    in-memory doubles and need NO network, NO credentials and NO real DB.

WHAT IT COMPOSES (existing, frozen components -- nothing is re-implemented)

    ``ProcedureCatalog``          -- support.procedure_catalog (cold-start)
    ``YandexAnalyzerProvider``    -- support.alice_analyzer_yandex
    ``YandexKnowledgeBackend``    -- support.knowledge_yandex
    ``YandexKnowledgeQAProvider`` -- support.knowledge_qa_yandex
    ``YandexExplanationProvider`` -- support.response_generator_yandex
    ``ProcedureStateStore``       -- support.procedure_state_store (factory)
    ``SupportTurnExecutor``       -- support.turn_executor
    ``SupportTurnDispatcher``     -- support.turn_dispatch (family dispatch)

WHAT IT DELIBERATELY DOES NOT DO (later sub-slices / other slices own it)

  * no runtime wiring and no server / channel / transport integration;
  * NO execution POLICY beyond identity resolution and the mechanical
    projection of the dispatcher's already-typed results: no retry, no
    automatic re-run, no procedure switching, no version upgrade/fallback and
    no safe-failure policy that invents a decision;
  * no HTTP, no FSM, no Tracker, no retries, no procedure switching and no
    conversation / message ownership ("cold start" here means ONLY "the
    persisted state carries no procedure identity yet");
  * no routing, no reducing, no retrieval and no response generation: the
    Router, the frozen orchestrator, the catalog, the dispatcher, the family
    selector and the renderer keep every one of those responsibilities;
  * it does NOT duplicate the frozen ``process_support_turn`` helper, does
    NOT call the orchestrator directly and does NOT re-implement the durable
    turn boundary -- it COMPOSES the dispatcher, which composes and calls ONLY
    the existing :class:`~support.turn_executor.SupportTurnExecutor`.
  * ``ProcedureCatalog`` remains the ONLY component that maps a cold-start
    signal to a procedure identity, and ``SupportTurnExecutor`` remains the
    only path to the orchestrator; this module performs neither mapping nor
    orchestration itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from support.alice_analyzer import AnalyzerProvider
from support.alice_analyzer_yandex import YandexAnalyzerProvider
from support.clarification import ClarificationResult
from support.db import get_session_factory
from support.global_policy import GlobalAction
from support.knowledge import KnowledgeBackend
from support.knowledge_qa import KnowledgeQAProvider, KnowledgeQAResult
from support.knowledge_qa_yandex import YandexKnowledgeQAProvider
from support.knowledge_yandex import YandexKnowledgeBackend
from support.procedure_catalog import ProcedureCatalog
from support.procedure_state_store import (
    PersistedProcedureState,
    ProcedureStateStore,
)
from support.response_generator import ResponseProvider
from support.response_generator_yandex import YandexExplanationProvider
from support.turn_dispatch import (
    DispatchResult,
    ExecutedDispatch,
    GlobalTerminalDispatch,
    NoProcedureDispatch,
    ReplayedDispatch,
    SupportTurnDispatcher,
)
from support.turn_executor import PersistedTurnResult, SupportTurnExecutor
from support.turn_family import TurnFamily


# ============================================================
# TURN OUTCOME (Slice 11A.3b, extended in Slice 11B.2)
# ============================================================

# The closed set of ways ONE support turn can end AT THIS APPLICATION
# BOUNDARY. They are composition outcomes ONLY: never an FSM event, never a
# Tracker action, never an HTTP status and never user-facing policy wording.
TURN_EXECUTED = "TURN_EXECUTED"              # PROCEDURE_SUPPORT + persisted
KNOWLEDGE_QA = "KNOWLEDGE_QA"                # KNOWLEDGE_QA family answered
CLARIFICATION = "CLARIFICATION"              # CLARIFICATION family prompted
ESCALATE = "ESCALATE"                        # global terminal: human asked
OUT_OF_SCOPE = "OUT_OF_SCOPE"                # global terminal: out of scope
NO_PROCEDURE = "NO_PROCEDURE"                # PROBLEM turn, no candidate
PROCEDURE_NOT_FOUND = "PROCEDURE_NOT_FOUND"  # bound identity unresolvable

_TURN_STATUSES = (
    TURN_EXECUTED,
    KNOWLEDGE_QA,
    CLARIFICATION,
    ESCALATE,
    OUT_OF_SCOPE,
    NO_PROCEDURE,
    PROCEDURE_NOT_FOUND,
)

# Exactly the ONE payload each status carries. Statuses with no payload
# (NO_PROCEDURE / PROCEDURE_NOT_FOUND) are absent and required to be empty.
_PAYLOAD_FIELD = {
    TURN_EXECUTED: "persisted",
    KNOWLEDGE_QA: "knowledge",
    CLARIFICATION: "clarification",
    ESCALATE: "text",
    OUT_OF_SCOPE: "text",
}


@dataclass(frozen=True)
class SupportTurnOutcome:
    """What :meth:`SupportCore.handle_turn` produced for ONE user turn.

      * ``TURN_EXECUTED`` -- PROCEDURE_SUPPORT ran and persisted its next
        state; ``persisted`` carries the EXISTING
        :class:`~support.turn_executor.PersistedTurnResult` VERBATIM (the
        caller projects ``persisted.turn`` / ``persisted.state``).
      * ``KNOWLEDGE_QA`` -- the KNOWLEDGE_QA family answered; ``knowledge``
        carries the family's :class:`~support.knowledge_qa.KnowledgeQAResult`
        (its status is preserved intact).
      * ``CLARIFICATION`` -- the CLARIFICATION family prompted;
        ``clarification`` carries the family's
        :class:`~support.clarification.ClarificationResult`.
      * ``ESCALATE`` / ``OUT_OF_SCOPE`` -- a GLOBAL terminal preempted family
        dispatch after the ONE analysis; ``text`` is the EXISTING
        deterministic wording projected by ``support.turn_projection`` (this
        boundary invents NO wording of its own).
      * ``NO_PROCEDURE`` -- a PROBLEM turn with no resolved procedure
        candidate: no capability ran, no procedure was invented/defaulted and
        NO user-facing wording is produced here.
      * ``PROCEDURE_NOT_FOUND`` -- the persisted identity (the EXACT
        ``procedure_id`` / ``procedure_version`` pair) could not be
        rehydrated; fail closed BEFORE any analysis. No fallback, no version
        upgrade, no ``latest`` and no identity inference from the current
        text.

    Exactly the ONE payload matching ``status`` is set; every other payload is
    ``None``. A concurrency conflict is deliberately NOT modelled here: the
    existing typed :class:`~support.procedure_state_store.ProcedureStateConflict`
    crosses this boundary UNCHANGED, so no duplicate vocabulary is invented.
    """

    status: str
    conversation_id: str
    text: Optional[str] = None
    persisted: Optional[PersistedTurnResult] = None
    knowledge: Optional[KnowledgeQAResult] = None
    clarification: Optional[ClarificationResult] = None

    def __post_init__(self) -> None:
        if self.status not in _TURN_STATUSES:
            raise ValueError(f"unknown turn status {self.status!r}.")

        supplied = {
            name: value
            for name, value in (
                ("text", self.text),
                ("persisted", self.persisted),
                ("knowledge", self.knowledge),
                ("clarification", self.clarification),
            )
            if value is not None
        }
        expected = _PAYLOAD_FIELD.get(self.status)
        if expected is None:
            if supplied:
                raise ValueError(f"{self.status} must not carry a payload.")
            return
        if list(supplied) != [expected]:
            raise ValueError(
                f"{self.status} requires exactly its {expected!r} payload."
            )
        if expected == "text":
            if not isinstance(self.text, str) or not self.text.strip():
                raise ValueError("text must be a non-blank string.")
        elif expected == "persisted":
            if not isinstance(self.persisted, PersistedTurnResult):
                raise ValueError(
                    "persisted must be a PersistedTurnResult."
                )
        elif expected == "knowledge":
            if not isinstance(self.knowledge, KnowledgeQAResult):
                raise ValueError("knowledge must be a KnowledgeQAResult.")
        elif expected == "clarification":
            if not isinstance(self.clarification, ClarificationResult):
                raise ValueError(
                    "clarification must be a ClarificationResult."
                )


class SupportCore:
    """Composition container + the single support-turn application boundary.

    Every collaborator is injectable; the DEFAULT for each is the real
    production component, so ``SupportCore()`` is the real production
    composition and can still be exercised offline by injecting doubles:

        SupportCore(
            procedure_catalog=...,
            analyzer_provider=...,
            knowledge_backend=...,
            knowledge_qa_provider=...,
            response_provider=...,
            session_factory=...,
            store_factory=...,
            turn_executor=...,
            turn_dispatcher=...,
        )

    Construction only HOLDS the composed dependencies (plus the read-only
    ``ProcedureCatalog`` lookup used by :meth:`handle_turn`). The database
    session factory and the turn executor are resolved lazily, so composing
    the core never requires database configuration and never opens a session.
    """

    def __init__(
        self,
        *,
        procedure_catalog: Optional[ProcedureCatalog] = None,
        analyzer_provider: Optional[AnalyzerProvider] = None,
        knowledge_backend: Optional[KnowledgeBackend] = None,
        knowledge_qa_provider: Optional[KnowledgeQAProvider] = None,
        response_provider: Optional[ResponseProvider] = None,
        session_factory: Optional[
            async_sessionmaker[AsyncSession]
        ] = None,
        store_factory: Callable[
            [AsyncSession], ProcedureStateStore
        ] = ProcedureStateStore,
        turn_executor: Optional[SupportTurnExecutor] = None,
        turn_dispatcher: Optional[SupportTurnDispatcher] = None,
    ) -> None:
        self._procedure_catalog = (
            ProcedureCatalog.from_directory()
            if procedure_catalog is None
            else procedure_catalog
        )
        self._analyzer_provider = (
            YandexAnalyzerProvider()
            if analyzer_provider is None
            else analyzer_provider
        )
        self._knowledge_backend = (
            YandexKnowledgeBackend()
            if knowledge_backend is None
            else knowledge_backend
        )
        self._knowledge_qa_provider = (
            YandexKnowledgeQAProvider()
            if knowledge_qa_provider is None
            else knowledge_qa_provider
        )
        self._response_provider = (
            YandexExplanationProvider()
            if response_provider is None
            else response_provider
        )
        self._session_factory = session_factory
        self._store_factory = store_factory
        self._turn_executor = turn_executor
        self._turn_dispatcher = turn_dispatcher

    # ---------------------------------------------------------
    # COMPOSED DEPENDENCIES (read-only views)
    # ---------------------------------------------------------

    @property
    def procedure_catalog(self) -> ProcedureCatalog:
        """The cold-start procedure catalog (the ONLY selector)."""
        return self._procedure_catalog

    @property
    def analyzer_provider(self) -> AnalyzerProvider:
        """The frozen analyzer port implementation."""
        return self._analyzer_provider

    @property
    def knowledge_backend(self) -> KnowledgeBackend:
        """The frozen knowledge port implementation."""
        return self._knowledge_backend

    @property
    def knowledge_qa_provider(self) -> KnowledgeQAProvider:
        """The frozen Knowledge QA provider port implementation."""
        return self._knowledge_qa_provider

    @property
    def response_provider(self) -> ResponseProvider:
        """The frozen response-generator port implementation."""
        return self._response_provider

    @property
    def store_factory(self) -> Callable[[AsyncSession], ProcedureStateStore]:
        """The per-session ``ProcedureStateStore`` factory."""
        return self._store_factory

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """The session factory, resolved from ``support.db`` on first use."""
        if self._session_factory is None:
            self._session_factory = get_session_factory()
        return self._session_factory

    @property
    def turn_executor(self) -> SupportTurnExecutor:
        """The durable turn boundary, composed once on first use.

        Built from the already-resolved session factory and store factory;
        constructing it is pure (no connection is opened).
        """
        if self._turn_executor is None:
            self._turn_executor = SupportTurnExecutor(
                session_factory=self.session_factory,
                store_factory=self._store_factory,
            )
        return self._turn_executor

    @property
    def turn_dispatcher(self) -> SupportTurnDispatcher:
        """The single family-dispatch boundary, composed once on first use.

        Built from the already-resolved ports and the (lazily resolved)
        executor; constructing it is pure (no connection, no provider call).
        """
        if self._turn_dispatcher is None:
            self._turn_dispatcher = SupportTurnDispatcher(
                analyzer_provider=self._analyzer_provider,
                procedure_executor=self.turn_executor,
                knowledge_backend=self._knowledge_backend,
                knowledge_qa_provider=self._knowledge_qa_provider,
                response_provider=self._response_provider,
            )
        return self._turn_dispatcher

    # ---------------------------------------------------------
    # TURN EXECUTION (Slices 11A.3b -> 11B.2)
    # ---------------------------------------------------------

    async def handle_turn(
        self,
        *,
        conversation_id: str,
        message: str,
        history: Sequence[Mapping[str, str]] = (),
        execution_id: Optional[str] = None,
    ) -> SupportTurnOutcome:
        """Resolve the candidate Procedure, then dispatch ONE support turn.

        The ONLY decision taken here is "cold start vs continuation", and
        both of its branches are delegated verbatim to frozen collaborators:

          * unbound conversation -> ``ProcedureCatalog.select(message)`` --
            the candidate MAY be ``None``; the dispatcher STILL analyses the
            message (an INFORMATION turn reaches Knowledge QA with or without
            a candidate);
          * bound conversation   -> ``ProcedureCatalog.get(procedure_id,
            procedure_version)`` -- the EXACT persisted pair (no ``select``
            re-run, no version upgrade, no fallback, no ``latest``).

        The candidate (and the read-only persisted state) is handed to the
        composed :class:`~support.turn_dispatch.SupportTurnDispatcher`, which
        owns the ONE analysis, Global Semantic Policy and the ONE family
        capability -- including the durable READ -> PROCESS -> CAS WRITE
        lifecycle of :class:`~support.turn_executor.SupportTurnExecutor`.
        This method never writes ``ProcedureState`` itself, never runs an
        analyzer and never calls the orchestrator / router.

        ``history`` is an opaque pass-through: it is forwarded unchanged.

        ``execution_id`` is an OPTIONAL, OPAQUE idempotency identity (HL-2c.1).
        It is forwarded UNCHANGED to the dispatcher, which forwards it to the
        procedure executor ONLY. It is never parsed, normalised, hashed,
        generated, truncated, combined, derived, semantically validated or
        inferred from ``conversation_id`` / ``message`` / ``history``, and it is
        never STORED on this boundary. Omitting it preserves the exact legacy
        behaviour and the exact legacy downstream call shape. A repeated id
        therefore yields a replayed (NOT recomputed, NOT re-worded)
        ``TURN_EXECUTED`` outcome carrying the persisted winner.

        Returns a :class:`SupportTurnOutcome`. Raises
        :class:`~support.procedure_state_store.ProcedureStateConflict`
        unchanged when the executor's CAS loses a race (no automatic retry
        and no provider re-run).
        """
        stored = await self._read_persisted_state(conversation_id)
        state = stored.state if stored is not None else None

        # --- CONTINUATION: EXACT identity rehydration only (fail closed) ---
        if state is not None and state.procedure_id is not None:
            version = state.procedure_version
            procedure = (
                None
                if version is None
                else self._procedure_catalog.get(state.procedure_id, version)
            )
            if procedure is None:
                # Unknown id OR wrong/unknown version: safe failure, never the
                # latest version and never another version of the same id. No
                # analysis runs and nothing is persisted.
                return SupportTurnOutcome(
                    status=PROCEDURE_NOT_FOUND,
                    conversation_id=conversation_id,
                )

        # --- COLD START: the catalog is the ONLY candidate mapper (may be None)
        else:
            selection = self._procedure_catalog.select(message)
            procedure = None if selection is None else selection.procedure

        # --- ONE dispatch: analysis -> global policy -> exactly one family ---
        result: DispatchResult = await self.turn_dispatcher.dispatch(
            conversation_id=conversation_id,
            message=message,
            history=history,
            state=state,
            procedure=procedure,
            execution_id=execution_id,
        )
        return self._project_outcome(conversation_id, result)

    def _project_outcome(
        self, conversation_id: str, result: DispatchResult
    ) -> SupportTurnOutcome:
        """Mechanically map the dispatcher's typed result to the public
        outcome. No wording and no decision are produced here."""
        if isinstance(result, NoProcedureDispatch):
            return SupportTurnOutcome(
                status=NO_PROCEDURE, conversation_id=conversation_id
            )
        if isinstance(result, GlobalTerminalDispatch):
            return SupportTurnOutcome(
                status=(
                    ESCALATE
                    if result.action is GlobalAction.ESCALATE
                    else OUT_OF_SCOPE
                ),
                conversation_id=conversation_id,
                text=result.text,
            )
        if isinstance(result, ReplayedDispatch):
            # HL-2b idempotent replay: the durable state was ALREADY advanced
            # under this execution id, so NOTHING was recomputed and NO new
            # user-facing text exists. The EXISTING TURN_EXECUTED vocabulary is
            # reused with the ACTUALLY persisted winner projected VERBATIM;
            # no wording and no new status are invented here.
            return SupportTurnOutcome(
                status=TURN_EXECUTED,
                conversation_id=conversation_id,
                persisted=result.procedure_result,
            )
        if isinstance(result, ExecutedDispatch):
            # Exactly one family capability ran; the family-native payload is
            # projected VERBATIM.
            if result.family is TurnFamily.PROCEDURE_SUPPORT:
                return SupportTurnOutcome(
                    status=TURN_EXECUTED,
                    conversation_id=conversation_id,
                    persisted=result.procedure_result,
                )
            if result.family is TurnFamily.KNOWLEDGE_QA:
                return SupportTurnOutcome(
                    status=KNOWLEDGE_QA,
                    conversation_id=conversation_id,
                    knowledge=result.knowledge_result,
                )
            return SupportTurnOutcome(
                status=CLARIFICATION,
                conversation_id=conversation_id,
                clarification=result.clarification_result,
            )
        raise AssertionError("unreachable dispatch result")

    async def _read_persisted_state(
        self, conversation_id: str
    ) -> Optional[PersistedProcedureState]:
        """ONE short, read-only, lock-free read of the persisted identity.

        The session is closed BEFORE any provider call, so no transaction is
        held across PROCESS. This is deliberately the only additional DB
        touch of this boundary: it explains WHICH procedure the conversation
        is bound to, and it never writes (the executor remains the single
        writer).
        """
        async with self.session_factory() as session:
            return await self._store_factory(session).read(conversation_id)


__all__ = [
    "CLARIFICATION",
    "ESCALATE",
    "KNOWLEDGE_QA",
    "NO_PROCEDURE",
    "OUT_OF_SCOPE",
    "PROCEDURE_NOT_FOUND",
    "TURN_EXECUTED",
    "SupportCore",
    "SupportTurnOutcome",
]
