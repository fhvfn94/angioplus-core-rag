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

IDEMPOTENT REPLAY (Slice HL-2b)
-------------------------------
Each entry point accepts an OPTIONAL, OPAQUE ``execution_id``. The executor
never interprets it and knows nothing about where it came from: it forwards
it to the store, which records it ATOMICALLY with the state so repeating the
SAME id can never advance the durable state twice (and never rewrites the
attempt log). There are TWO idempotent-hit paths and BOTH return the
ACTUALLY persisted state with ``replayed`` ``True`` and ``turn`` ``None``:

  * BEFORE PROCESS -- the persisted state was ALREADY advanced by this SAME
    id (a sequential replay); PROCESS and the CAS write are skipped; or
  * AFTER PROCESS -- this call's CAS lost a race to ANOTHER worker holding
    the SAME id (``CasResult.applied`` is ``False``); the recomputed turn is
    DISCARDED and the persisted winner state is returned. This is correct
    WITHOUT any external single-flight.

A DIFFERENT id with a stale revision still raises ``ProcedureStateConflict``.
No idempotency SERVICE is imported; the id is just an opaque value passed
through.

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
from support.domain import ConversationState
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
from support.turn_recovery import TurnRecoveryPayload
from support.turn_recovery_store import TurnRecoveryStore


@dataclass(frozen=True)
class PersistedTurnResult:
    """A :class:`TurnResult` plus the durable coordinates of its persistence.

    On a NORMAL turn, ``state`` is exactly ``turn.state`` (the persisted next
    state) and ``revision`` is the NEW revision written by the CAS.

    When ``replayed`` is ``True`` the turn was an IDEMPOTENT hit: the durable
    state had ALREADY been advanced by this execution id, so nothing was
    recomputed and nothing was written. In that case ``turn`` is ``None``,
    ``state`` is the ACTUALLY persisted state (never a recomputed candidate)
    and ``revision`` is the CURRENT, unchanged revision.

    ``recovery`` is the durable turn-recovery record (HL-2c.3a) belonging to
    the WINNING ``execution_id``, exposed ONLY on a replay (``None`` on a
    normal turn, which carries its own ``turn``). It is read VERBATIM from the
    dedicated ``turn_recovery`` table -- never recomputed -- and is the
    material from which the ORIGINAL runtime turn can be reproduced without
    re-running the LLM / retrieval / procedure. ``None`` means no record was
    stored (legacy caller) and the caller must NOT fabricate a turn.
    """

    conversation_id: str
    turn: Optional[TurnResult]
    state: ProcedureState
    revision: int
    replayed: bool = False
    recovery: Optional[TurnRecoveryPayload] = None

    def __post_init__(self) -> None:
        if self.replayed and self.turn is not None:
            raise ValueError("a replayed result carries no recomputed turn.")
        if not self.replayed and self.turn is None:
            raise ValueError("a non-replayed result requires a turn.")
        if not self.replayed and self.recovery is not None:
            raise ValueError("a recovery record is only exposed on a replay.")


class SupportTurnExecutor:
    """Run + durably persist one support turn (load -> process -> CAS-save)."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        store_factory: Callable[
            [AsyncSession], ProcedureStateStore
        ] = ProcedureStateStore,
        recovery_store_factory: Callable[
            [AsyncSession], TurnRecoveryStore
        ] = TurnRecoveryStore,
    ) -> None:
        self._session_factory = session_factory
        self._store_factory = store_factory
        self._recovery_store_factory = recovery_store_factory

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
        execution_id: Optional[str] = None,
        projection_state: Optional[ConversationState] = None,
    ) -> PersistedTurnResult:
        """Process one turn (analyzer ONCE) and persist its state under CAS.

        ``procedure`` is RESOLVED and supplied by the caller; it is never
        selected or switched here. ``history`` is an opaque caller-supplied
        input (the executor never reads the message log). ``conversation_id``
        must reference an existing conversation (FK precondition); creating
        conversations is out of scope.

        ``execution_id`` is an OPTIONAL, OPAQUE idempotency identity (the
        executor never interprets it); see :meth:`_execute_persisted`.
        ``projection_state`` is the OPTIONAL immutable
        :class:`~support.domain.ConversationState` the runtime projection will
        consult; when supplied TOGETHER WITH an ``execution_id`` the recovery
        record is written ATOMICALLY with the CAS (see
        :meth:`_execute_persisted`).
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

        kwargs: dict = {
            "conversation_id": conversation_id,
            "process": _process,
            "execution_id": execution_id,
        }
        if projection_state is not None:
            # HL-2b compatibility convention: OMIT the optional keyword when it
            # is absent, so a caller that does not project keeps the EXACT
            # previous ``_execute_persisted`` call shape (and any pre-existing
            # monkeypatch / subclass stays valid).
            kwargs["projection_state"] = projection_state
        return await self._execute_persisted(**kwargs)

    async def execute_persisted_turn_from_observation(
        self,
        *,
        conversation_id: str,
        observation: SemanticObservation,
        procedure: Procedure,
        knowledge_backend: Optional[KnowledgeBackend] = None,
        response_provider: Optional[ResponseProvider] = None,
        execution_id: Optional[str] = None,
        projection_state: Optional[ConversationState] = None,
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

        ``execution_id`` is an OPTIONAL, OPAQUE idempotency identity (the
        executor never interprets it); ``projection_state`` is the OPTIONAL
        immutable :class:`~support.domain.ConversationState` the runtime
        projection will consult. Both feed :meth:`_execute_persisted`.
        """

        def _process(previous_state: Optional[ProcedureState]) -> TurnResult:
            return process_support_turn_from_observation(
                observation=observation,
                state=previous_state,
                procedure=procedure,
                knowledge_backend=knowledge_backend,
                response_provider=response_provider,
            )

        kwargs: dict = {
            "conversation_id": conversation_id,
            "process": _process,
            "execution_id": execution_id,
        }
        if projection_state is not None:
            # HL-2b compatibility convention: OMIT the optional keyword when it
            # is absent, so a caller that does not project keeps the EXACT
            # previous ``_execute_persisted`` call shape (and any pre-existing
            # monkeypatch / subclass stays valid).
            kwargs["projection_state"] = projection_state
        return await self._execute_persisted(**kwargs)

    async def _execute_persisted(
        self,
        *,
        conversation_id: str,
        process: Callable[[Optional[ProcedureState]], TurnResult],
        execution_id: Optional[str] = None,
        projection_state: Optional[ConversationState] = None,
    ) -> PersistedTurnResult:
        """The SINGLE shared durable lifecycle (one persistence writer).

        Both public entry points delegate here, so the READ -> PROCESS -> CAS
        WRITE boundaries -- and the one CAS writer -- live in exactly one
        place. No transaction is held across PROCESS.

        ``execution_id`` is an OPTIONAL, OPAQUE idempotency identity. The
        executor never interprets it and knows nothing about where it came
        from; it only forwards it to the store so that repeating the SAME id
        cannot advance the durable state twice. When it is supplied and the
        persisted state was ALREADY advanced by that SAME id, the turn is an
        IDEMPOTENT HIT: PROCESS and the CAS are skipped entirely and the
        actually persisted state is returned. If instead the CAS itself loses
        a race to the SAME id, ``CasResult.applied`` is ``False`` and the
        recomputed turn is DISCARDED in favour of the persisted winner state.

        ``projection_state`` is the OPTIONAL immutable
        :class:`~support.domain.ConversationState` the runtime projection will
        consult. When it is supplied TOGETHER WITH ``execution_id``, a
        :class:`~support.turn_recovery.TurnRecoveryPayload` is captured from
        the PROCESS result and written on the SAME session / transaction as
        the CAS (step 3), so the procedure state and its recovery record
        COMMIT OR ROLL BACK TOGETHER: there is no crash window in which the
        CAS advanced but the recovery record is missing. Any idempotent replay
        loads the WINNER's STORED record VERBATIM -- never one recomputed on
        the hit.
        """
        # 1. READ (short, read-only, lock-free, then closed).
        async with self._session_factory() as session:
            stored = await self._store_factory(session).read(conversation_id)

            # IDEMPOTENT HIT: this execution already advanced the durable
            # state. Load the WINNER's durable recovery record on the SAME
            # read session, return the PERSISTED state (never a recomputed
            # candidate) and skip PROCESS and the CAS write entirely.
            if (
                execution_id is not None
                and stored is not None
                and stored.last_execution_id == execution_id
            ):
                return PersistedTurnResult(
                    conversation_id=conversation_id,
                    turn=None,
                    state=stored.state,
                    revision=stored.revision,
                    replayed=True,
                    recovery=await self._read_recovery(
                        session, conversation_id, execution_id
                    ),
                )

        previous_state = stored.state if stored is not None else None
        expected_revision = stored.revision if stored is not None else None

        # 2. PROCESS (no DB transaction open across provider calls).
        turn = process(previous_state)

        # Capture the recovery record (canonical engine result + projection
        # snapshot) BEFORE the write session opens -- pure scalar extraction.
        # It is keyed by ``execution_id``, so it is built ONLY when an id was
        # supplied AND a projection snapshot exists; otherwise NO record is
        # written (legacy behaviour, untouched).
        recovery: Optional[TurnRecoveryPayload] = None
        if execution_id is not None and projection_state is not None:
            recovery = TurnRecoveryPayload.capture(
                execution_id=execution_id,
                conversation_id=conversation_id,
                turn=turn,
                projection_state=projection_state,
            )

        # 3. WRITE (short CAS transaction; commit).
        async with self._session_factory() as session:
            store = self._store_factory(session)
            try:
                save_kwargs: dict = {"expected_revision": expected_revision}
                if execution_id is not None:
                    # Pass the opaque id ONLY when supplied, so a caller that
                    # does not use idempotency keeps the EXACT previous
                    # ``save_cas`` call shape (and any pre-existing custom
                    # store stays compatible).
                    save_kwargs["execution_id"] = execution_id
                cas = await store.save_cas(
                    conversation_id, turn.state, **save_kwargs
                )
                if cas.applied and recovery is not None:
                    # SAME session / transaction as the CAS, and ONLY for an
                    # APPLIED write (a losing idempotent race never rewrites
                    # the winner's record). A failure here rolls the WHOLE
                    # transaction back (below), including the CAS.
                    await self._recovery_store_factory(session).save(recovery)
                await session.commit()
            except Exception:
                await session.rollback()
                raise

        # POST-CAS IDEMPOTENT HIT: another worker with the SAME execution id
        # won the race between our READ and our CAS. Our write did NOT apply,
        # so DISCARD the recomputed turn/state we just produced and return the
        # ACTUALLY persisted state (plus the WINNER's stored recovery record)
        # instead. This makes the same-execution-id race correct WITHOUT
        # external single-flight. The winner's record is guaranteed visible:
        # its CAS and its record committed in ONE transaction, which is what
        # our non-applied CAS observed.
        if not cas.applied:
            async with self._session_factory() as session:
                winner_recovery = await self._read_recovery(
                    session, conversation_id, execution_id
                )
            return PersistedTurnResult(
                conversation_id=conversation_id,
                turn=None,
                state=cas.state,
                revision=cas.revision,
                replayed=True,
                recovery=winner_recovery,
            )

        return PersistedTurnResult(
            conversation_id=conversation_id,
            turn=turn,
            state=cas.state,
            revision=cas.revision,
        )

    async def _read_recovery(
        self,
        session: AsyncSession,
        conversation_id: str,
        execution_id: Optional[str],
    ) -> Optional[TurnRecoveryPayload]:
        """Load the stored recovery record of a replayed execution (or None).

        ``None`` when no ``execution_id`` was supplied (nothing to key on), or
        when the winner stored no record (a legacy caller that supplied no
        projection snapshot). A present-but-malformed record raises
        :class:`~support.turn_recovery.TurnRecoveryCorruption` from the store
        -- NEVER silently coerced to ``None``.
        """
        if execution_id is None:
            return None
        return await self._recovery_store_factory(session).read(
            execution_id=execution_id,
            conversation_id=conversation_id,
        )


__all__ = [
    "PersistedTurnResult",
    "ProcedureStateConflict",
    "SupportTurnExecutor",
]
