"""
Finalized Runtime Replay (LEVEL A) -- SLICE HL-2c.3b.1.

Infrastructure only. NOT wired into any production runtime path yet
(``yandex/chat/server.py``, ``alice_support_agent.py``, n8n, Tracker stay
untouched). PURE value object + PURE reproducer; the durable persistence that
stores the snapshot lives in :mod:`support.models` (``inbound_events``) and is
written / read by :class:`support.idempotency.IdempotencyService`.

THE TWO REPLAY LEVELS (why this module exists)
----------------------------------------------
:class:`~support.turn_executor.SupportTurnExecutor` commits the ProcedureState
CAS in ONE short transaction; the runtime (a LATER slice) then persists the
user / assistant messages, applies the FSM event and finalizes the inbound
event in its OWN transaction. That gives TWO independent duplicate-delivery
cases:

  * LEVEL B -- ENGINE / PROCEDURE replay (HL-2b, surfaced by HL-2c.1/2): the
    CAS itself was already applied, so nothing was recomputed.

  * LEVEL A -- FINALIZED runtime replay (THIS module): the turn was FULLY
    finalized, so the runtime must reproduce the EXACT HTTP body it returned
    the first time -- WITHOUT re-running the LLM / retrieval / wording / FSM /
    Tracker. Today that reproduction is lossy: the legacy HTTP compatibility
    field ``retrieval_used`` is not stored, so the duplicate branch hardcodes
    ``retrieval_used: False`` and re-derives ``external_case_id`` from the
    (mutable) current conversation.

WHAT IS THE MINIMUM DURABLE SNAPSHOT
------------------------------------
The user-facing body of a finalized chat turn is derived from exactly:

  * the answer text -- stored ONCE, in ``messages`` (referenced by
    ``inbound_events.assistant_message_id``); it is NEVER duplicated into the
    replay snapshot, the resolved text is loaded from the message row;
  * ``decision`` / ``reason`` / ``case_action`` -- already persisted control
    metadata;
  * ``retrieval_used`` -- NOW persisted on ``inbound_events.retrieval_used``;
  * the lifecycle SNAPSHOT -- ``conversation_mode`` / ``support_status`` /
    ``handoff_status`` as they STOOD when the turn was finalized (the live
    conversation may have moved on since);
  * ``external_case_id`` -- the Tracker key the body echoed, snapshotted so a
    later change to the conversation cannot rewrite a replay.

The snapshot is written ATOMICALLY with the finalize (the SAME transaction)
and read back VERBATIM. NO value is re-derived from mutable current state, so
an ``ESCALATE`` that created a case cannot later "replay" as a ``RESUME``.

SEMANTICS (non-negotiable)
--------------------------
  * NO LLM / retrieval rerun, NO wording regeneration, NO FSM transition, NO
    second procedure execution, NO new message -- replay is a PURE function of
    the stored snapshot (plus the already-persisted answer text).
  * ABSENT and CORRUPT are DIFFERENT:
      - an inbound event whose ``outcome`` is still ``received`` is IN FLIGHT:
        nothing was finalized, so there is nothing to replay. That is a
        LEGITIMATE condition -> :class:`FinalizedReplayUnavailable`.
      - an EXISTING finalized record that is partial / malformed (including a
        legacy row whose ``retrieval_used`` was never persisted, ``None``)
        raises :class:`FinalizedReplayCorruption` (fail closed LOUDLY). A
        malformed durable record is NEVER silently coerced into a replayable
        value, because a wrong body is worse than a loud failure.

This module does NOT read the raw inbound message, a
:class:`~support.observation.SemanticObservation`, a procedure state or any
conversation row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from support.domain import (
    ALLOWED_TRISTATE_COMBINATIONS,
    ConversationMode,
    ConversationState,
    HandoffStatus,
    SupportStatus,
)
from support.schemas import ChatResponse


# The FINALIZED inbound-event outcomes (``inbound_events.outcome``) for which a
# runtime response EXISTS and must be reproduced verbatim.
FINALIZED_OUTCOMES = frozenset({"answered", "escalated", "forwarded"})

# The COMPLETE inbound-event outcome vocabulary (mirrors the
# ``ck_inbound_events_outcome`` CHECK constraint). Used to tell a DAMAGED row
# (an unrecognized value) from a legitimate not-yet / non-replayable record.
KNOWN_OUTCOMES = frozenset(
    {
        "received",
        "answered",
        "escalated",
        "rejected",
        "duplicate",
        "forwarded",
    }
)

# The ONE in-flight outcome (claimed, not yet finalized).
_IN_FLIGHT_OUTCOME = "received"

# The ONE finalized outcome that legitimately carries NO AI answer (a
# HUMAN_FORWARD turn is the operator's reply, not the model's).
_NO_ANSWER_OUTCOME = "forwarded"


class FinalizedReplayUnavailable(RuntimeError):
    """There is nothing to replay at LEVEL A -- a LEGITIMATE condition.

    Raised when the inbound event is still IN FLIGHT (``outcome``
    ``received``), is a non-replayable record (``rejected`` / ``duplicate``) or
    produced NO assistant answer (e.g. the ``START_NEW_SESSION`` bookkeeping
    row). It is NOT an error in the record: the runtime must RECOVER
    (HL-2c.3a) or re-run, never fabricate a body. Kept as a distinct type so a
    caller can never confuse "nothing finalized yet" with "a finalized record
    is damaged".
    """


class FinalizedReplayCorruption(RuntimeError):
    """A FINALIZED replay snapshot EXISTS but is partial / malformed.

    Raised LOUDLY (never swallowed, never degraded to a best-effort body).
    This includes a legacy finalized row whose ``retrieval_used`` was never
    persisted (``None``): without it an EXACT replay is impossible, and a
    guessed value would be a silent lie.
    """


def is_finalized_outcome(outcome: object) -> bool:
    """True when ``outcome`` names a FINALIZED, replayable inbound event."""
    return isinstance(outcome, str) and outcome in FINALIZED_OUTCOMES


def _require_non_blank(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FinalizedReplayCorruption(
            f"finalized field {field!r} must be a non-blank string."
        )
    return value


@dataclass(frozen=True, slots=True)
class FinalizedTurnSnapshot:
    """The canonical DURABLE snapshot of ONE finalized chat turn.

    Every field is an EXPLICIT, STABLE SCALAR (an enum ``.value`` string, an
    int, a bool, ``None`` or -- for ``answer`` -- the resolved message text) --
    no Python object, no ``repr``, nothing that could drift across versions.

    The DURABLE columns (written atomically with the finalize) are everything
    EXCEPT ``answer``: the answer text is resolved at load time from the
    assistant message referenced by ``assistant_message_id`` and is NEVER
    duplicated into ``inbound_events``. ``inbound_event_id`` is the delivery
    identity (the key); ``conversation_id`` is carried so a reader can refuse a
    record that does not belong to the conversation asking.
    """

    inbound_event_id: str
    conversation_id: str
    outcome: str

    # Durable link to the single source of truth for the answer text. ``None``
    # only for a HUMAN_FORWARD turn (no AI reply exists).
    assistant_message_id: Optional[int]

    # The RESOLVED answer text (loaded from ``messages`` by the DB reader; not
    # persisted on ``inbound_events``). Empty string for a forwarded turn.
    answer: str

    decision: Optional[str]
    reason: Optional[str]
    case_action: str
    retrieval_used: bool
    conversation_mode: str
    support_status: str
    handoff_status: Optional[str] = None
    external_case_id: Optional[str] = None

    def __post_init__(self) -> None:
        _require_non_blank(self.inbound_event_id, "inbound_event_id")
        _require_non_blank(self.conversation_id, "conversation_id")
        _require_non_blank(self.outcome, "outcome")
        _require_non_blank(self.case_action, "case_action")
        _require_non_blank(self.conversation_mode, "conversation_mode")
        _require_non_blank(self.support_status, "support_status")

        # A snapshot describes a FINALIZED turn; in-flight is never a snapshot.
        if self.outcome == _IN_FLIGHT_OUTCOME:
            raise FinalizedReplayUnavailable(
                "inbound event is still in flight ('received'); nothing has "
                "been finalized and there is no response to replay."
            )
        if self.outcome not in FINALIZED_OUTCOMES:
            raise FinalizedReplayCorruption(
                f"unknown finalized outcome {self.outcome!r}."
            )

        # `retrieval_used` must be a REAL, PERSISTED boolean: ``None`` is the
        # legacy / not-persisted case and cannot be replayed faithfully.
        if not isinstance(self.retrieval_used, bool):
            raise FinalizedReplayCorruption(
                "finalized field 'retrieval_used' must be a persisted "
                f"boolean, got {self.retrieval_used!r}."
            )

        if not isinstance(self.answer, str):
            raise FinalizedReplayCorruption("'answer' must be a string.")

        if self.outcome == _NO_ANSWER_OUTCOME:
            # A HUMAN_FORWARD turn is the operator's reply: NO assistant
            # message, NO AI answer.
            if self.assistant_message_id is not None:
                raise FinalizedReplayCorruption(
                    "a forwarded turn must not reference an assistant message."
                )
            if self.answer != "":
                raise FinalizedReplayCorruption(
                    "a forwarded turn carries no AI answer."
                )
            for name, value in (
                ("decision", self.decision),
                ("reason", self.reason),
            ):
                if value is not None and not isinstance(value, str):
                    raise FinalizedReplayCorruption(
                        f"{name!r} must be a string or None."
                    )
        else:
            # An AI turn: the answer LIVES in the referenced assistant message.
            if (
                not isinstance(self.assistant_message_id, int)
                or isinstance(self.assistant_message_id, bool)
                or self.assistant_message_id < 1
            ):
                raise FinalizedReplayCorruption(
                    "an AI turn requires a positive 'assistant_message_id'."
                )
            _require_non_blank(self.answer, "answer")
            _require_non_blank(self.decision, "decision")
            _require_non_blank(self.reason, "reason")

        if self.handoff_status is not None:
            _require_non_blank(self.handoff_status, "handoff_status")
        if self.external_case_id is not None:
            _require_non_blank(self.external_case_id, "external_case_id")

        # Fail closed on a snapshot that is not a real lifecycle state: the
        # triple is re-validated here so a caller cannot replay a body that
        # contradicts the FSM whitelist.
        self.conversation_state()

    # ---------------------------------------------------------
    # TYPED RECONSTRUCTION
    # ---------------------------------------------------------

    def conversation_state(self) -> ConversationState:
        """The snapshotted FSM triple, as a validated domain value object."""
        try:
            state: ConversationState = (
                ConversationMode(self.conversation_mode),
                SupportStatus(self.support_status),
                (
                    None
                    if self.handoff_status is None
                    else HandoffStatus(self.handoff_status)
                ),
            )
        except ValueError as exc:
            raise FinalizedReplayCorruption(
                "finalized lifecycle snapshot is not a valid "
                f"ConversationState: {exc}"
            ) from exc

        if state not in ALLOWED_TRISTATE_COMBINATIONS:
            raise FinalizedReplayCorruption(
                f"finalized lifecycle snapshot {state!r} is not an allowed "
                "conversation state."
            )
        return state


# ============================================================
# REPRODUCTION
# ============================================================

def reproduce_finalized_response(
    snapshot: FinalizedTurnSnapshot,
) -> ChatResponse:
    """Rebuild the EXACT HTTP response of an ALREADY-finalized chat turn.

    Pure, deterministic and total over a VALID snapshot: every field is copied
    VERBATIM from the snapshot -- NO LLM, NO retrieval, NO wording, NO FSM
    transition, NO new message and NO read of the CURRENT conversation state.
    ``deduplicated`` is ``True`` by construction: the only caller is a duplicate
    delivery (the first delivery finalized the snapshot).

    A malformed snapshot never reaches here: :class:`FinalizedTurnSnapshot`
    fails closed at construction time.
    """
    return ChatResponse(
        ok=True,
        answer=snapshot.answer,
        retrieval_used=snapshot.retrieval_used,
        decision=snapshot.decision or "",
        reason=snapshot.reason or "",
        conversation_id=snapshot.conversation_id,
        conversation_mode=snapshot.conversation_mode,
        support_status=snapshot.support_status,
        handoff_status=snapshot.handoff_status,
        case_action=snapshot.case_action,
        deduplicated=True,
        # Echoed verbatim; declared as an allowed extra on ``ChatResponse``.
        external_case_id=snapshot.external_case_id,
    )


__all__ = [
    "FINALIZED_OUTCOMES",
    "KNOWN_OUTCOMES",
    "FinalizedReplayCorruption",
    "FinalizedReplayUnavailable",
    "FinalizedTurnSnapshot",
    "is_finalized_outcome",
    "reproduce_finalized_response",
]
