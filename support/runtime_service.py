"""
Stateful Turn Runtime -- the ONE application runtime boundary for ONE inbound
support turn (Slice HL-2c.3b.2).

Infrastructure for the composition root: this module is NOT imported by any
production runtime module yet (``yandex/chat/server.py``,
``alice_support_agent.py``, n8n, Tracker stay untouched). It is the LAST piece
BEFORE ``server.py`` is wired, and it exists so that wiring step is a short,
explicit, type-checked COMPOSITION rather than new policy.

THE THREE DELIVERY PATHS (one boundary, three typed outcomes)
------------------------------------------------------------
EVERY inbound turn resolves to EXACTLY ONE of three frozen deliveries, and all
three are produced by this single boundary:

  1. FRESH DELIVERY -- :meth:`StatefulTurnRuntime.execute_fresh_turn`, used
     when the inbound event was newly claimed (``claim.claimed`` is ``True``).
     The boundary invokes ``SupportCore.handle_turn`` EXACTLY ONCE (the
     sanctioned composition owner of turn execution -- see below) and adapts
     its outcome with :func:`support.runtime_outcome.adapt_turn_outcome`:

       * a genuine new turn -> :class:`FreshTurnDelivery`, wrapping the
         ``RuntimeTurn`` that projection produced;
       * an ENGINE REPLAY (HL-2b: the durable state was ALREADY advanced under
         this execution id, so nothing was recomputed and no wording exists) --
         the boundary does NOT treat it as a fresh turn. It consumes the
         DURABLE recovery payload and reproduces the ORIGINAL turn ->
         :class:`RecoveredTurnDelivery`. No second ``SupportCore`` call, no
         provider re-run;
       * an engine replay with NO durable recovery payload (a legacy caller
         that stored none) -> :class:`RuntimeRecoveryUnavailable`. Fail closed:
         re-running is forbidden and fabricating a turn is forbidden.

  2. W3 RECOVERY -- :meth:`StatefulTurnRuntime.resolve_duplicate` (LEVEL B),
     used for a duplicate delivery whose first attempt crashed AFTER the engine
     applied the turn but BEFORE the runtime finalized. Nothing was finalized,
     so the ORIGINAL turn is reproduced from the ``turn_recovery`` record that
     the engine wrote ATOMICALLY with its CAS (HL-2c.3a) via
     :func:`support.turn_recovery.reproduce_runtime_turn`. ``SupportCore`` and
     every provider are bypassed COMPLETELY; nothing is written.

  3. FINALIZED DUPLICATE -- :meth:`StatefulTurnRuntime.resolve_duplicate`
     (LEVEL A), used for a duplicate delivery whose first attempt finished: the
     inbound event was finalized with its snapshot (HL-2c.3b.1). The EXACT
     persisted body is reproduced from that snapshot via
     :meth:`IdempotencyService.load_finalized_replay` +
     :func:`support.finalized_replay.reproduce_finalized_response`.
     ``SupportCore`` is bypassed ENTIRELY; there is NO FSM, message, Tracker or
     provider side effect.

:meth:`StatefulTurnRuntime.handle_delivery` composes the fresh path and the two
duplicate paths behind the ONE claim: it claims the inbound event, then
dispatches to the appropriate path and returns the same closed
:data:`RuntimeDelivery` vocabulary.

A DAMAGED LEVEL A snapshot (``FinalizedReplayCorruption`` -- e.g. a legacy
finalized row whose replay snapshot was never persisted) is NOT served
best-effort. The boundary resumes path 2: if a VALID durable recovery record
exists it reproduces the ORIGINAL turn from THAT record; only when no durable
source can serve is the loud corruption error re-raised. "ABSENT"
(:class:`RuntimeRecoveryUnavailable`) and "CORRUPT" (``...Corruption``) never
collapse into each other.

RUNTIME IDENTITY
----------------
At this boundary the inbound event id IS the engine execution id: the durable
recovery record is keyed by it (that is exactly how
:meth:`StatefulTurnRuntime.resolve_duplicate` finds a crashed turn).
:meth:`handle_delivery` therefore forwards the SAME id to the engine by
construction, so a caller cannot claim / finalize under one id and execute
under another.

SANCTIONED SupportCore OWNER
----------------------------
This module is the ONE sanctioned application composition module allowed to
import ``support.support_core``. The repository architecture guard
(``tests/test_support_core.py``) pins EXACTLY this file as the sole owner, so no
other runtime module gains direct ``SupportCore`` access (in particular
``yandex/chat/server.py`` stays transport-only and constructs no provider).

This module still DECIDES nothing and re-derives NO lifecycle mapping:
``decision`` / ``reason`` / ``fsm_event`` / ``case_action`` come from
:func:`support.runtime_outcome.adapt_turn_outcome` (whose single owner is
:mod:`support.lifecycle_projection`), never from this file, which imports
neither the projection owner nor the router.

WHAT IT IS (and is not)
-----------------------
A small composition service over the pieces that already exist:

  * :class:`~support.support_core.SupportCore` -- the ONLY turn execution
    boundary (FRESH); invoked exactly once per fresh delivery;
  * :mod:`support.runtime_outcome` -- the pure ``SupportTurnOutcome`` ->
    ``RuntimeTurn | RuntimeReplay`` adapter (reused VERBATIM);
  * :class:`~support.idempotency.IdempotencyService` -- the claim and the
    durable finalized replay (LEVEL A);
  * :class:`DurableRecoveryStore` -- the recovery-record seam whose DEFAULT
    implementation is the EXISTING
    :class:`~support.turn_recovery_store.TurnRecoveryStore` (LEVEL B); no new
    table, no new persistence, no migration;
  * :mod:`support.finalized_replay` and :mod:`support.turn_recovery` -- the two
    pure reproducers (reused VERBATIM).

It owns no wording, no conversation / message / FSM / Tracker write and no
transport mapping. :meth:`resolve_duplicate` READS ONLY: resolving a duplicate
creates no message, moves no FSM state and writes no row.

``retrieval_used`` is DELIBERATELY not synthesized here: the authoritative
:class:`~support.runtime_outcome.RuntimeTurn` has no such field (it is a legacy
HTTP compatibility concern), so inventing it for a fresh / recovered turn would
be a silent lie. This boundary returns the typed ``RuntimeTurn`` and the later
server cutover / finalization layer (HL-2c.3b.3) supplies the HTTP-only
metadata. A FINALIZED duplicate is different and IS exact: it carries the
persisted :class:`~support.schemas.ChatResponse`, ``retrieval_used`` included.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Callable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from support.domain import ConversationState
from support.finalized_replay import (
    FinalizedReplayCorruption,
    FinalizedReplayUnavailable,
    FinalizedTurnSnapshot,
    reproduce_finalized_response,
)
from support.idempotency import IdempotencyService
from support.runtime_outcome import (
    RuntimeTurn,
    adapt_turn_outcome,
)
from support.schemas import ChatResponse, ClaimResult
from support.support_core import SupportCore, SupportTurnOutcome
from support.turn_recovery import TurnRecoveryPayload, reproduce_runtime_turn
from support.turn_recovery_store import TurnRecoveryStore


# ============================================================
# THE DURABLE SEAMS (structural contracts over EXISTING code)
# ============================================================

@runtime_checkable
class InboundEventService(Protocol):
    """The runtime-facing inbound-event seam.

    Declared STRUCTURALLY so the runtime never depends on the concrete
    implementation (and so it stays unit-testable offline). The DEFAULT is the
    EXISTING :class:`support.idempotency.IdempotencyService` -- this slice adds
    no dedup logic and no table.
    """

    async def claim(
        self, inbound_event_id: str, channel: str
    ) -> ClaimResult:
        ...

    async def load_finalized_replay(
        self, inbound_event_id: str
    ) -> FinalizedTurnSnapshot:
        ...


@runtime_checkable
class DurableRecoveryStore(Protocol):
    """The runtime-facing contract of the DURABLE recovery record (HL-2c.3a).

    This is the "recovery snapshot store": it reads / writes the
    :class:`~support.turn_recovery.TurnRecoveryPayload` that is written
    ATOMICALLY with the engine CAS. Declared STRUCTURALLY so the runtime can
    reproduce a crashed turn from the durable record WITHOUT importing -- or
    depending on the internals of -- the LEVEL B writer.

    The DEFAULT implementation is the EXISTING
    :class:`support.turn_recovery_store.TurnRecoveryStore`: this slice adds NO
    table and NO new persistence, only the named seam.
    """

    async def save(self, payload: TurnRecoveryPayload) -> None:
        ...

    async def read(
        self, *, execution_id: str, conversation_id: str
    ) -> Optional[TurnRecoveryPayload]:
        ...


# ============================================================
# TYPED FAILURE (fail closed, never fabricate, never re-run)
# ============================================================

class RuntimeRecoveryUnavailable(RuntimeError):
    """A delivery has NOTHING durable to reproduce -- ABSENT, not corrupt.

    Raised when

      * a duplicate delivery has no finalized snapshot (the inbound row is
        still ``received``) AND the durable recovery record is ABSENT (a
        legacy caller that stored none, or a delivery that crashed BEFORE the
        engine applied under its id); or
      * the engine itself reports an idempotent REPLAY but the winner stored
        no recovery record, so the original turn cannot be reproduced.

    It is a DISTINCT, legitimate condition: the runtime must neither fabricate
    a body nor re-run the model. A present-but-DAMAGED record is a different,
    loud condition (``TurnRecoveryCorruption`` / ``FinalizedReplayCorruption``)
    and never degrades to THIS one.
    """


# ============================================================
# CLOSED DELIVERY VOCABULARY
# ============================================================

@dataclass(frozen=True, slots=True)
class FreshTurnDelivery:
    """Path 1a: a GENUINE new turn, executed ONCE through ``SupportCore``.

    ``turn`` is the :class:`~support.runtime_outcome.RuntimeTurn` produced by
    :func:`support.runtime_outcome.adapt_turn_outcome` from the
    ``SupportTurnOutcome`` that ``SupportCore.handle_turn`` returned. The
    boundary adds nothing to it: the HTTP-only metadata (``retrieval_used``,
    the finalized body) is supplied by the later server cutover, never here.
    """

    turn: RuntimeTurn


@dataclass(frozen=True, slots=True)
class RecoveredTurnDelivery:
    """Path 1b (engine replay) / path 2 (W3 recovery): the ORIGINAL runtime turn
    of a turn whose engine part applied but which the runtime never finalized.

    ``turn`` is reproduced from the
    :class:`~support.turn_recovery.TurnRecoveryPayload` written atomically with
    the engine CAS. Carrying the typed
    :class:`~support.runtime_outcome.RuntimeTurn` (rather than a synthesized
    HTTP body) is deliberate: the recovered record does not carry
    ``retrieval_used``, so inventing a body here would be a silent lie. The
    transport maps ``turn`` exactly as it maps a fresh turn.
    """

    turn: RuntimeTurn


@dataclass(frozen=True, slots=True)
class FinalizedDelivery:
    """Path 3 (LEVEL A): the EXACT original HTTP body of a FINALIZED turn.

    ``body`` is reproduced VERBATIM from the durable finalized snapshot --
    ``retrieval_used`` and the lifecycle triple included, never re-derived from
    the (mutable) current conversation. ``deduplicated`` is ``True`` by
    construction.
    """

    body: ChatResponse


# The closed runtime DELIVERY vocabulary of ONE inbound turn: a fresh turn, a
# recovered turn, or a finalized duplicate -- never more than one, never
# anything else. A bare "engine replay" with nothing durable behind it is NOT a
# delivery: there is nothing to serve, so that case is a typed FAILURE
# (``RuntimeRecoveryUnavailable``), not a fourth value.
RuntimeDelivery = FreshTurnDelivery | RecoveredTurnDelivery | FinalizedDelivery


# ============================================================
# THE RUNTIME BOUNDARY
# ============================================================

class StatefulTurnRuntime:
    """Resolve the runtime outcome of ONE inbound support turn, idempotently.

    The composition root (``yandex/chat/server.py``) drives the inbound-event
    lifecycle through it:

      * :meth:`handle_delivery` -- the ONE composed entry point: claim, then
        dispatch to exactly one of the paths below;
      * :meth:`claim` -- the short TX1 claim (adapter over
        :meth:`IdempotencyService.claim`); ``claimed`` is ``True`` for the FIRST
        delivery and ``False`` for a duplicate. A caller that wants to resolve
        the conversation ONLY when the claim is won can claim first and
        dispatch itself;
      * :meth:`execute_fresh_turn` -- the FRESH path (path 1), run when the
        claim was won: ONE ``SupportCore.handle_turn`` call, adapted to the
        runtime vocabulary;
      * :meth:`resolve_duplicate` -- the duplicate branch: reproduces the
        ALREADY-APPLIED turn (paths 2 / 3) or fails closed.

    Every method takes ONLY the identity the runtime already owns: the inbound
    event id, the channel, the conversation id, the message, the history and the
    current projection state. The boundary never resolves a conversation, never
    builds the history, never finalizes an inbound event and never touches the
    messages / FSM / Tracker.
    """

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        inbound_event_factory: Callable[
            [AsyncSession], InboundEventService
        ] = IdempotencyService,
        recovery_store_factory: Callable[
            [AsyncSession], DurableRecoveryStore
        ] = TurnRecoveryStore,
        support_core: Optional[SupportCore] = None,
    ) -> None:
        self._session_factory = session_factory
        # Every seam defaults to the EXISTING production implementation;
        # injecting a double keeps the boundary unit-testable without a
        # database / network. ``support_core`` is the ONE sanctioned
        # composition dependency of this module (see the module docstring).
        self._inbound_event_factory = inbound_event_factory
        self._recovery_store_factory = recovery_store_factory
        self._support_core = support_core

    # ---------------------------------------------------------
    # COMPOSED DEPENDENCIES (lazy, side-effect free)
    # ---------------------------------------------------------

    @property
    def support_core(self) -> SupportCore:
        """The composed ``SupportCore``, built once on first use.

        Built from THIS boundary's session factory, so the turn engine's
        durable store and its recovery store share the one configured database.
        Constructing it touches no database and reads no credential (see
        ``SupportCore``'s own construction contract); nothing is resolved until
        the FIRST fresh delivery.
        """
        if self._support_core is None:
            self._support_core = SupportCore(
                session_factory=self._session_factory
            )
        return self._support_core

    # ---------------------------------------------------------
    # CLAIM (TX1)
    # ---------------------------------------------------------

    async def claim(self, *, inbound_event_id: str, channel: str) -> ClaimResult:
        """Claim the inbound event on its OWN short, committed transaction.

        Returns ``claimed=True`` for the first delivery (the runtime then
        resolves the conversation, runs ONE fresh turn and finalizes it), or
        ``claimed=False`` for a duplicate (the runtime resolves it through
        :meth:`resolve_duplicate` and serves the durable result).
        """
        async with self._session_factory() as session:
            result = await self._inbound_event_factory(session).claim(
                inbound_event_id, channel
            )
            await session.commit()
            return result

    # ---------------------------------------------------------
    # PATH 1 -- FRESH DELIVERY (SupportCore, exactly once)
    # ---------------------------------------------------------

    async def execute_fresh_turn(
        self,
        *,
        conversation_id: str,
        message: str,
        projection_state: ConversationState,
        history: Sequence[Mapping[str, str]] = (),
        execution_id: Optional[str] = None,
    ) -> RuntimeDelivery:
        """Execute ONE genuinely new turn and adapt it to the runtime contract.

        ``SupportCore.handle_turn`` is invoked EXACTLY ONCE, with
        ``conversation_id`` / ``message`` / ``history`` / ``projection_state``
        and the OPAQUE ``execution_id`` forwarded UNCHANGED (never parsed,
        normalized, generated or derived here). ``projection_state`` is
        REQUIRED: it is the immutable current FSM triple the projection
        consults, it makes the turn W3-recoverable (the engine records the
        recovery payload atomically with its CAS when it is supplied together
        with an ``execution_id``), and without it no ``RuntimeTurn`` can be
        projected at all.

        The returned ``SupportTurnOutcome`` is adapted by
        :func:`support.runtime_outcome.adapt_turn_outcome`:

          * a genuine new turn -> :class:`FreshTurnDelivery`;
          * an ENGINE REPLAY -> :class:`RecoveredTurnDelivery`, reproduced from
            the durable recovery payload the engine loaded VERBATIM from the
            winner's record. The replay is NEVER treated as a fresh turn, and
            ``SupportCore`` is NOT called a second time;
          * an engine replay with no durable payload -> fail closed with
            :class:`RuntimeRecoveryUnavailable`.

        ``execution_id=None`` keeps the legacy no-idempotency call shape; such a
        turn is NOT W3-recoverable (the engine stores no record to find).

        A ``ProcedureStateConflict`` raised by the engine propagates UNCHANGED
        (no retry, no provider re-run).
        """
        outcome = await self.support_core.handle_turn(
            conversation_id=conversation_id,
            message=message,
            history=history,
            execution_id=execution_id,
            projection_state=projection_state,
        )

        projected = adapt_turn_outcome(outcome, projection_state)

        if isinstance(projected, RuntimeTurn):
            return FreshTurnDelivery(turn=projected)

        # ENGINE REPLAY (HL-2b / HL-2c.2): the durable state was ALREADY
        # advanced under this execution id, so there is NO new turn and NO
        # wording. NOT a fresh delivery -- reproduce the ORIGINAL turn.
        return RecoveredTurnDelivery(
            turn=self._reproduce_engine_replay(outcome, execution_id)
        )

    @staticmethod
    def _reproduce_engine_replay(
        outcome: SupportTurnOutcome,
        execution_id: Optional[str],
    ) -> RuntimeTurn:
        """Reproduce the ORIGINAL turn of an engine replay from durable state.

        The payload is the one the engine ALREADY loaded from the
        ``turn_recovery`` table when it detected the replay, so this boundary
        performs NO additional read (and cannot race it). ``None`` means the
        winner stored no record (legacy caller): fail closed rather than
        fabricate or re-run.
        """
        persisted = outcome.persisted
        payload = None if persisted is None else persisted.recovery
        if payload is None:
            raise RuntimeRecoveryUnavailable(
                "the engine reported an idempotent replay for execution "
                f"{execution_id!r} but no durable recovery record was stored; "
                "the original turn cannot be reproduced and re-running is "
                "forbidden."
            )
        return reproduce_runtime_turn(payload)

    # ---------------------------------------------------------
    # PATHS 2 / 3 -- DUPLICATE RESOLUTION (read only)
    # ---------------------------------------------------------

    async def resolve_duplicate(
        self,
        *,
        inbound_event_id: str,
        conversation_id: str,
    ) -> RuntimeDelivery:
        """Reproduce an ALREADY-APPLIED turn from durable state (READ ONLY).

        Resolution order (each step exists to keep a FACT, never a guess):

        1. LEVEL A. Load the finalized replay snapshot. On success the EXACT
           original body is reproduced -> :class:`FinalizedDelivery`. The
           snapshot's conversation must be the one asking (a replay can never
           be served across conversations).
        2. LEVEL B. If LEVEL A has nothing (still in flight) OR is DAMAGED
           (legacy finalized row / partial snapshot), read the durable recovery
           record and reproduce the ORIGINAL turn from it ->
           :class:`RecoveredTurnDelivery`.
        3. Otherwise fail closed:
             * a DAMAGED LEVEL A record with NO durable recovery record
               re-raises :class:`FinalizedReplayCorruption` (loud);
             * otherwise :class:`RuntimeRecoveryUnavailable` (nothing to
               reproduce).

        ``SupportCore`` is NEVER consulted here, so a duplicate delivery can
        never trigger a provider call. This method also never creates a message
        and mutates no row: it is a pure read over durable state.
        """
        async with self._session_factory() as session:
            finalized: Optional[FinalizedDelivery] = None
            level_a_failure: Optional[RuntimeError] = None

            try:
                snapshot = await self._inbound_event_factory(
                    session
                ).load_finalized_replay(inbound_event_id)
            except (
                FinalizedReplayUnavailable,
                FinalizedReplayCorruption,
            ) as exc:
                level_a_failure = exc
            else:
                if snapshot.conversation_id != conversation_id:
                    # A replay is only ever served to the conversation that
                    # owns the finalized turn -- never across conversations.
                    raise FinalizedReplayCorruption(
                        f"finalized event {inbound_event_id!r} belongs to "
                        f"conversation {snapshot.conversation_id!r}, not "
                        f"{conversation_id!r}."
                    )
                finalized = FinalizedDelivery(
                    body=reproduce_finalized_response(snapshot)
                )

            if finalized is not None:
                return finalized

            # LEVEL B / CRASH RECOVERY: the durable record written atomically
            # with the engine CAS. ABSENT is legitimate (``None``); a
            # present-but-damaged record raises ``TurnRecoveryCorruption`` from
            # the store and is never coerced to ``None``.
            payload = await self._recovery_store_factory(session).read(
                execution_id=inbound_event_id,
                conversation_id=conversation_id,
            )
            if payload is not None:
                return RecoveredTurnDelivery(
                    turn=reproduce_runtime_turn(payload)
                )

            if isinstance(level_a_failure, FinalizedReplayCorruption):
                # A damaged LEVEL A record stays LOUD when no durable source
                # could serve: damage is never hidden and never degraded to a
                # best-effort body.
                raise level_a_failure
            raise RuntimeRecoveryUnavailable(
                f"duplicate delivery {inbound_event_id!r} has no finalized "
                "replay and no durable recovery record; nothing to reproduce "
                "and re-running is forbidden."
            ) from level_a_failure

    # ---------------------------------------------------------
    # THE ONE COMPOSED ENTRY POINT
    # ---------------------------------------------------------

    async def handle_delivery(
        self,
        *,
        inbound_event_id: str,
        channel: str,
        conversation_id: str,
        message: str,
        projection_state: ConversationState,
        history: Sequence[Mapping[str, str]] = (),
    ) -> RuntimeDelivery:
        """Claim ONE inbound delivery, then resolve EXACTLY ONE of the paths.

        This is the ONE boundary: it claims the inbound event (TX1) and then

          * ``claimed`` -> the FRESH path (:meth:`execute_fresh_turn`), with the
            SAME id forwarded to the engine as its ``execution_id``; or
          * a duplicate -> the durable replay / recovery
            (:meth:`resolve_duplicate`).

        ``message`` / ``history`` / ``projection_state`` are used ONLY by the
        fresh path; a caller that wants to avoid building them for a duplicate
        can call :meth:`claim` first and dispatch itself. Finalizing the inbound
        event (TX2, messages, FSM, Tracker) stays with the composition root: this
        boundary resolves the turn, it does not own the conversation.
        """
        claim = await self.claim(
            inbound_event_id=inbound_event_id, channel=channel
        )

        if not claim.claimed:
            return await self.resolve_duplicate(
                inbound_event_id=inbound_event_id,
                conversation_id=conversation_id,
            )

        return await self.execute_fresh_turn(
            conversation_id=conversation_id,
            message=message,
            history=history,
            execution_id=inbound_event_id,
            projection_state=projection_state,
        )


__all__ = [
    "DurableRecoveryStore",
    "FinalizedDelivery",
    "FreshTurnDelivery",
    "InboundEventService",
    "RecoveredTurnDelivery",
    "RuntimeDelivery",
    "RuntimeRecoveryUnavailable",
    "StatefulTurnRuntime",
]
