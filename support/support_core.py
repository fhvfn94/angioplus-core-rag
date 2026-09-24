"""
Support Core -- production dependency COMPOSITION (Slice 11A.3a) and the
single support-turn EXECUTION flow (Slice 11A.3b).

Infrastructure only. This module is the single place where the
ALREADY-EXISTING Support Core building blocks are constructed and held
together, and -- since Slice 11A.3b -- the single place where ONE support
turn is driven end to end:

    conversation_id + raw message
      -> ONE short, read-only read of the persisted ProcedureState
      -> resolve the CURRENT Procedure IDENTITY (frozen collaborators)
      -> SupportTurnExecutor.execute_persisted_turn(...)
      -> SupportTurnOutcome

It stays a COMPOSITION / APPLICATION BOUNDARY: it owns no wording, no
policy and no decision logic.

PROCEDURE RESOLUTION (delegated -- this module has no selection algorithm)
-------------------------------------------------------------------------
  * COLD START (no persisted ``procedure_id``) -> ``ProcedureCatalog.select(
    message)``, the ONLY cold-start mapper: zero qualifying procedures ->
    ``NO_PROCEDURE``, exactly one -> that Procedure, MORE than one (ambiguous)
    -> ``NO_PROCEDURE``. No ranking, no best-match, no default procedure, no
    ``latest`` and no model-chosen ``procedure_id``.
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
    ``YandexExplanationProvider`` -- support.response_generator_yandex
    ``ProcedureStateStore``       -- support.procedure_state_store (factory)
    ``SupportTurnExecutor``       -- support.turn_executor

WHAT IT DELIBERATELY DOES NOT DO (later sub-slices / other slices own it)

  * no runtime wiring and no server / channel / transport integration;
  * NO execution POLICY beyond identity resolution: no retry, no automatic
    re-run, no procedure switching, no version upgrade/fallback and no
    safe-failure policy that invents a decision;
  * no HTTP, no FSM, no Tracker, no retries, no procedure switching and no
    conversation / message ownership ("cold start" here means ONLY "the
    persisted state carries no procedure identity yet");
  * no routing, no reducing, no retrieval and no response generation: the
    Router, the frozen orchestrator, the catalog and the renderer keep every
    one of those responsibilities;
  * it does NOT duplicate the frozen ``process_support_turn`` helper, does
    NOT call the orchestrator directly and does NOT re-implement the durable
    turn boundary -- it COMPOSES and calls ONLY the existing
    :class:`~support.turn_executor.SupportTurnExecutor`;
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
from support.db import get_session_factory
from support.knowledge import KnowledgeBackend
from support.knowledge_yandex import YandexKnowledgeBackend
from support.procedure_catalog import ProcedureCatalog
from support.procedure_state_store import (
    PersistedProcedureState,
    ProcedureStateStore,
)
from support.response_generator import ResponseProvider
from support.response_generator_yandex import YandexExplanationProvider
from support.turn_executor import PersistedTurnResult, SupportTurnExecutor


# ============================================================
# TURN OUTCOME (Slice 11A.3b)
# ============================================================

# The three ways ONE support turn can end AT THIS COMPOSITION BOUNDARY.
# They are composition outcomes ONLY: never an FSM event, never a Tracker
# action, never an HTTP status and never user-facing handoff policy.
TURN_EXECUTED = "TURN_EXECUTED"
NO_PROCEDURE = "NO_PROCEDURE"
PROCEDURE_NOT_FOUND = "PROCEDURE_NOT_FOUND"

_TURN_STATUSES = (TURN_EXECUTED, NO_PROCEDURE, PROCEDURE_NOT_FOUND)


@dataclass(frozen=True)
class SupportTurnOutcome:
    """What :meth:`SupportCore.handle_turn` produced for ONE user turn.

      * ``TURN_EXECUTED`` -- the executor ran the turn and persisted its
        next state; ``persisted`` carries the EXISTING
        :class:`~support.turn_executor.PersistedTurnResult` VERBATIM (the
        caller projects ``persisted.turn`` / ``persisted.state``).
      * ``NO_PROCEDURE`` -- cold start with no UNIQUE
        ``ProcedureCatalog.select`` match (no match, or ambiguous). No
        procedure was invented, defaulted or model-chosen, and no turn ran.
      * ``PROCEDURE_NOT_FOUND`` -- the persisted identity (the EXACT
        ``procedure_id`` / ``procedure_version`` pair) could not be
        rehydrated. No fallback, no version upgrade, no ``latest`` and no
        identity inference from the current text; no turn ran.

    A concurrency conflict is deliberately NOT modelled here: the existing
    typed :class:`~support.procedure_state_store.ProcedureStateConflict`
    crosses this boundary UNCHANGED, so no duplicate vocabulary is invented.
    """

    status: str
    conversation_id: str
    persisted: Optional[PersistedTurnResult] = None

    def __post_init__(self) -> None:
        if self.status not in _TURN_STATUSES:
            raise ValueError(f"unknown turn status {self.status!r}.")
        if self.status == TURN_EXECUTED:
            if not isinstance(self.persisted, PersistedTurnResult):
                raise ValueError(
                    f"{TURN_EXECUTED} requires the executor's "
                    "PersistedTurnResult."
                )
        elif self.persisted is not None:
            raise ValueError(
                f"{self.status} must not carry a persisted result."
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
            response_provider=...,
            session_factory=...,
            store_factory=...,
            turn_executor=...,
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
        response_provider: Optional[ResponseProvider] = None,
        session_factory: Optional[
            async_sessionmaker[AsyncSession]
        ] = None,
        store_factory: Callable[
            [AsyncSession], ProcedureStateStore
        ] = ProcedureStateStore,
        turn_executor: Optional[SupportTurnExecutor] = None,
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
        self._response_provider = (
            YandexExplanationProvider()
            if response_provider is None
            else response_provider
        )
        self._session_factory = session_factory
        self._store_factory = store_factory
        self._turn_executor = turn_executor

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

    # ---------------------------------------------------------
    # TURN EXECUTION (Slice 11A.3b)
    # ---------------------------------------------------------

    async def handle_turn(
        self,
        *,
        conversation_id: str,
        message: str,
        history: Sequence[Mapping[str, str]] = (),
    ) -> SupportTurnOutcome:
        """Resolve the CURRENT Procedure, then run + persist ONE support turn.

        The ONLY decision taken here is "cold start vs continuation", and
        both of its branches are delegated verbatim to frozen collaborators:

          * unbound conversation -> ``ProcedureCatalog.select(message)``;
          * bound conversation   -> ``ProcedureCatalog.get(procedure_id,
            procedure_version)`` -- the EXACT persisted pair (no ``select``
            re-run, no version upgrade, no fallback, no ``latest``).

        A resolved Procedure is handed to the composed
        :class:`~support.turn_executor.SupportTurnExecutor`, which owns the
        durable READ -> PROCESS -> CAS WRITE lifecycle. This method never
        writes ``ProcedureState`` itself and never calls the orchestrator.

        ``history`` is an opaque pass-through: it is forwarded unchanged.

        Returns a :class:`SupportTurnOutcome`. Raises
        :class:`~support.procedure_state_store.ProcedureStateConflict`
        unchanged when the executor's CAS loses a race (no automatic retry
        and no provider re-run).
        """
        stored = await self._read_persisted_state(conversation_id)

        # --- COLD START: no persisted identity -> the catalog is the mapper ---
        if stored is None or stored.state.procedure_id is None:
            selection = self._procedure_catalog.select(message)
            if selection is None:
                # No unique cold-start match (none, or ambiguous): never
                # invent/default a procedure and never let a model choose one.
                return SupportTurnOutcome(
                    status=NO_PROCEDURE, conversation_id=conversation_id
                )
            procedure = selection.procedure

        # --- CONTINUATION: EXACT identity rehydration only ---
        else:
            version = stored.state.procedure_version
            procedure = (
                None
                if version is None
                else self._procedure_catalog.get(
                    stored.state.procedure_id, version
                )
            )
            if procedure is None:
                # Unknown id OR wrong/unknown version: safe failure, never the
                # latest version and never another version of the same id.
                return SupportTurnOutcome(
                    status=PROCEDURE_NOT_FOUND,
                    conversation_id=conversation_id,
                )

        # --- ONE durable turn: the executor owns read -> process -> CAS write.
        persisted = await self.turn_executor.execute_persisted_turn(
            conversation_id=conversation_id,
            message=message,
            procedure=procedure,
            analyzer_provider=self._analyzer_provider,
            history=history,
            knowledge_backend=self._knowledge_backend,
            response_provider=self._response_provider,
        )
        return SupportTurnOutcome(
            status=TURN_EXECUTED,
            conversation_id=conversation_id,
            persisted=persisted,
        )

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
    "NO_PROCEDURE",
    "PROCEDURE_NOT_FOUND",
    "TURN_EXECUTED",
    "SupportCore",
    "SupportTurnOutcome",
]
