"""
SupportTurnExecutor -- durable application boundary for ONE support turn.

Infrastructure only. NOT imported by any production runtime path (server.py,
alice_support_agent.py, n8n, Tracker). It is a standalone application service,
not yet wired anywhere.

PURPOSE
-------
Compose the frozen, already-total :func:`support.orchestrator.process_support_turn`
with DURABLE, concurrency-safe persistence of the resulting
:class:`~support.procedure_state.ProcedureState`, without touching any
transport / FSM / Tracker concern. It is bookkeeping and transaction
boundaries ONLY: it makes no decisions, selects no procedure, renders no
wording, and adds no policy.

LIFECYCLE (each phase exactly once, in order)
---------------------------------------------
  1. READ   (short read-only transaction, NO lock)
       store.read(conversation_id) -> (previous_state, expected_revision)
  2. PROCESS (NO database transaction open)
       process(...)                -> TurnResult
  3. WRITE  (short CAS transaction, commit)
       store.save_cas(next_state, expected_revision=...) -> new revision

Both public entry points share this EXACT lifecycle (and the ONE CAS writer):

  * :meth:`SupportTurnExecutor.execute_persisted_turn` -- PROCESS runs the
    analyzer ONCE (the legacy path);
  * :meth:`SupportTurnExecutor.execute_persisted_turn_from_observation` --
    PROCESS consumes a PRECOMPUTED
    :class:`~support.observation.SemanticObservation` VERBATIM and takes NO
    ``analyzer_provider``, so the analyzer can never run a second time.

Because the read session is closed before PROCESS and the write session is
opened after it, no database lock or transaction is held across any
analyzer / knowledge / response-generator call.

FAILURE SEMANTICS
-----------------
  * Provider failures are already absorbed inside the frozen orchestrator
    (analyzer -> fail-safe CLARIFY, retrieval -> ERROR, generator ->
    deterministic fallback); this module adds NO ``try/except`` around it,
    so a degraded turn is still a normal, persisted revision bump.
  * A CAS miss raises :class:`ProcedureStateConflict` (no automatic retry,
    no LLM retry): the caller decides.
  * A database failure in the write transaction rolls back BOTH the state
    row and the attempt log (they are written atomically), so there is never
    a partial ProcedureState write.

NON-GOALS (deliberately absent)
-------------------------------
No channel/runtime wiring, no FSM, no Tracker, no idempotency service, no
procedure selection, no conversation/message ownership, no network calls
beyond the ones the orchestrator already makes through injected ports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from support.alice_analyzer import AnalyzerProvider
from support.knowledge import KnowledgeBackend
from support.observation import SemanticObservation
from support.orchestrator import (
    TurnResult,
    process_support_turn,
    process_support_turn_from_observation,
)
from support.procedure import Procedure
from support.procedure_state import ProcedureState
from support.procedure_state_store import (
    ProcedureStateConflict,
    ProcedureStateStore,
)
from support.response_generator import ResponseProvider


@dataclass(frozen=True)
class PersistedTurnResult:
    """A :class:`TurnResult` plus the durable coordinates of its persistence.

    ``state`` is exactly ``turn.state`` (the persisted next state) and
    ``revision`` is the NEW revision written by the CAS.
    """

    conversation_id: str
    turn: TurnResult
    state: ProcedureState
    revision: int


class SupportTurnExecutor:
    """Run + durably persist one support turn (load -> process -> CAS-save)."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        store_factory: Callable[
            [AsyncSession], ProcedureStateStore
        ] = ProcedureStateStore,
    ) -> None:
        self._session_factory = session_factory
        self._store_factory = store_factory

    async def execute_persisted_turn(
        self,
        *,
        conversation_id: str,
        message: str,
        procedure: Procedure,
        analyzer_provider: AnalyzerProvider,
        history: Sequence[Mapping[str, str]] = (),
        knowledge_backend: Optional[KnowledgeBackend] = None,
        response_provider: Optional[ResponseProvider] = None,
    ) -> PersistedTurnResult:
        """Process one turn (analyzer ONCE) and persist its state under CAS.

        ``procedure`` is RESOLVED and supplied by the caller; it is never
        selected or switched here. ``history`` is an opaque caller-supplied
        input (the executor never reads the message log). ``conversation_id``
        must reference an existing conversation (FK precondition); creating
        conversations is out of scope.
        """

        def _process(previous_state: Optional[ProcedureState]) -> TurnResult:
            return process_support_turn(
                message=message,
                history=history,
                state=previous_state,
                procedure=procedure,
                analyzer_provider=analyzer_provider,
                knowledge_backend=knowledge_backend,
                response_provider=response_provider,
            )

        return await self._execute_persisted(
            conversation_id=conversation_id, process=_process
        )

    async def execute_persisted_turn_from_observation(
        self,
        *,
        conversation_id: str,
        observation: SemanticObservation,
        procedure: Procedure,
        knowledge_backend: Optional[KnowledgeBackend] = None,
        response_provider: Optional[ResponseProvider] = None,
    ) -> PersistedTurnResult:
        """Persist one turn from a PRECOMPUTED observation (no re-analysis).

        Identical durable lifecycle to :meth:`execute_persisted_turn` (the SAME
        READ -> PROCESS -> CAS WRITE body, one persistence writer); the ONLY
        difference is the PROCESS stage, which consumes the caller-supplied
        :class:`~support.observation.SemanticObservation` VERBATIM instead of
        running the analyzer. Because this method has NO ``analyzer_provider``
        argument and delegates to
        :func:`support.orchestrator.process_support_turn_from_observation`, the
        analyzer can NEVER be called a second time for the same user turn.

        ``procedure`` is RESOLVED and supplied by the caller; ``conversation_id``
        must reference an existing conversation (FK precondition).
        """

        def _process(previous_state: Optional[ProcedureState]) -> TurnResult:
            return process_support_turn_from_observation(
                observation=observation,
                state=previous_state,
                procedure=procedure,
                knowledge_backend=knowledge_backend,
                response_provider=response_provider,
            )

        return await self._execute_persisted(
            conversation_id=conversation_id, process=_process
        )

    async def _execute_persisted(
        self,
        *,
        conversation_id: str,
        process: Callable[[Optional[ProcedureState]], TurnResult],
    ) -> PersistedTurnResult:
        """The SINGLE shared durable lifecycle (one persistence writer).

        Both public entry points delegate here, so the READ -> PROCESS -> CAS
        WRITE boundaries -- and the one CAS writer -- live in exactly one
        place. No transaction is held across PROCESS.
        """
        # 1. READ (short, read-only, lock-free, then closed).
        async with self._session_factory() as session:
            stored = await self._store_factory(session).read(conversation_id)

        previous_state = stored.state if stored is not None else None
        expected_revision = stored.revision if stored is not None else None

        # 2. PROCESS (no DB transaction open across provider calls).
        turn = process(previous_state)

        # 3. WRITE (short CAS transaction; commit).
        async with self._session_factory() as session:
            store = self._store_factory(session)
            try:
                new_revision = await store.save_cas(
                    conversation_id,
                    turn.state,
                    expected_revision=expected_revision,
                )
                await session.commit()
            except ProcedureStateConflict:
                await session.rollback()
                raise

        return PersistedTurnResult(
            conversation_id=conversation_id,
            turn=turn,
            state=turn.state,
            revision=new_revision,
        )


__all__ = [
    "PersistedTurnResult",
    "ProcedureStateConflict",
    "SupportTurnExecutor",
]
