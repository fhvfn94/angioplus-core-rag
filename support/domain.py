"""
Channel-agnostic support domain.

This module contains the *pure* part of the support domain:

  - the closed vocabulary of modes / statuses / roles / channels / events
  - the canonical conversation state as a triple
  - the explicit transition table (FSM)
  - ``next_state`` -- a pure function used by the service layer

No I/O, no ORM, no framework imports live here so that the FSM can be
unit-tested in isolation and reasoned about by a human.

State model (FINAL DELTA v2 + v3):
    state = (conversation_mode, support_status, handoff_status)

All three axes are stored explicitly. The FSM and the DB CHECK constraint
share the same whitelist of *real* combinations
(``ALLOWED_TRISTATE_COMBINATIONS``); everything else is forbidden.

Notable semantics (per approved design):
    * waiting_user is AI-owned: while the operator waits for the user,
      the user keeps talking to the AI. When the user is ready,
      ``ESCALATE_RESUME`` returns the SAME case to the operator.
    * There is a single canonical closed state: ``(ai, closed, closed)``.
"""

from __future__ import annotations

from enum import Enum
from typing import FrozenSet, Optional, Tuple


# ============================================================
# ENUMS
# ============================================================

class _StrEnum(str, Enum):
    """Python 3.10 compatible str-enum (values are plain strings)."""

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return str(self.value)


class ConversationMode(_StrEnum):
    """Who currently owns the conversation."""

    AI = "ai"
    HUMAN = "human"


class SupportStatus(_StrEnum):
    """Lifecycle of the conversation, independent of ownership."""

    ACTIVE = "active"
    CLOSED = "closed"


class HandoffStatus(_StrEnum):
    """Ownership reason / sub-status. ``None`` means 'AI, no handoff'."""

    WAITING_OPERATOR = "waiting_operator"
    WAITING_USER = "waiting_user"
    WAITING_RESOLUTION_CONFIRMATION = "waiting_resolution_confirmation"
    CLOSED = "closed"


class MessageRole(_StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    OPERATOR = "operator"


class Channel(_StrEnum):
    """Planned channels. Phase 1 implements telegram + web only."""

    TELEGRAM = "telegram"
    WEB = "web"
    EMAIL = "email"
    VOICE = "voice"


class SupportEvent(_StrEnum):
    """All events that may drive the support FSM."""

    USER_MESSAGE_AI = "USER_MESSAGE_AI"
    USER_MESSAGE_HUMAN = "USER_MESSAGE_HUMAN"
    AI_RESOLVED = "AI_RESOLVED"

    ESCALATE_NEW = "ESCALATE_NEW"
    ESCALATE_RESUME = "ESCALATE_RESUME"

    OPERATOR_COMMENT = "OPERATOR_COMMENT"
    TRACKER_NEED_INFO = "TRACKER_NEED_INFO"
    TRACKER_RESOLVED = "TRACKER_RESOLVED"

    USER_CONFIRMED_RESOLVED = "USER_CONFIRMED_RESOLVED"
    USER_REJECTED_RESOLVED = "USER_REJECTED_RESOLVED"
    RESOLUTION_TIMEOUT = "RESOLUTION_TIMEOUT"

    START_NEW_SESSION = "START_NEW_SESSION"


# ============================================================
# STATE
# ============================================================

ConversationState = Tuple[ConversationMode, SupportStatus, Optional[HandoffStatus]]


# ============================================================
# CANONICAL STATES AND ALLOWED COMBINATIONS
# ============================================================
#
# This whitelist is the single source of truth shared by the FSM and the
# DB CHECK constraint. Any triple not listed here is invalid and must be
# rejected both by ``next_state`` and by PostgreSQL.

AI_ACTIVE: ConversationState = (ConversationMode.AI, SupportStatus.ACTIVE, None)
AI_ACTIVE_WAITING_USER: ConversationState = (
    ConversationMode.AI,
    SupportStatus.ACTIVE,
    HandoffStatus.WAITING_USER,
)
HUMAN_ACTIVE_WAITING_OPERATOR: ConversationState = (
    ConversationMode.HUMAN,
    SupportStatus.ACTIVE,
    HandoffStatus.WAITING_OPERATOR,
)
HUMAN_ACTIVE_WAITING_RESOLUTION: ConversationState = (
    ConversationMode.HUMAN,
    SupportStatus.ACTIVE,
    HandoffStatus.WAITING_RESOLUTION_CONFIRMATION,
)
CLOSED: ConversationState = (ConversationMode.AI, SupportStatus.CLOSED, HandoffStatus.CLOSED)


ALLOWED_TRISTATE_COMBINATIONS: FrozenSet[ConversationState] = frozenset(
    {
        AI_ACTIVE,
        AI_ACTIVE_WAITING_USER,
        HUMAN_ACTIVE_WAITING_OPERATOR,
        HUMAN_ACTIVE_WAITING_RESOLUTION,
        CLOSED,
    }
)


# ============================================================
# TRANSITION TABLE
# ============================================================
#
# Explicit table: (from_state, event) -> to_state.
# This is the minimal own implementation justified by the design review:
# mature FSM libraries assume a single state value, while our state is a
# genuine triple with zero behaviour inside states.

TRANSITIONS: dict[Tuple[ConversationState, SupportEvent], ConversationState] = {
    # --- AI self-service path ------------------------------------------
    (AI_ACTIVE, SupportEvent.USER_MESSAGE_AI): AI_ACTIVE,
    (AI_ACTIVE, SupportEvent.AI_RESOLVED): AI_ACTIVE,

    # --- escalation: AI -> operator ------------------------------------
    (AI_ACTIVE, SupportEvent.ESCALATE_NEW): HUMAN_ACTIVE_WAITING_OPERATOR,

    # --- operator thread ------------------------------------------------
    (
        HUMAN_ACTIVE_WAITING_OPERATOR,
        SupportEvent.USER_MESSAGE_HUMAN,
    ): HUMAN_ACTIVE_WAITING_OPERATOR,
    (
        HUMAN_ACTIVE_WAITING_OPERATOR,
        SupportEvent.OPERATOR_COMMENT,
    ): HUMAN_ACTIVE_WAITING_OPERATOR,

    # operator asks for information -> AI-owned, user keeps chatting
    (
        HUMAN_ACTIVE_WAITING_OPERATOR,
        SupportEvent.TRACKER_NEED_INFO,
    ): AI_ACTIVE_WAITING_USER,

    # operator resolved -> user must confirm (still human-owned)
    (
        HUMAN_ACTIVE_WAITING_OPERATOR,
        SupportEvent.TRACKER_RESOLVED,
    ): HUMAN_ACTIVE_WAITING_RESOLUTION,

    # --- AI-owned waiting_user -----------------------------------------
    # user keeps talking to the AI while the operator waits
    (AI_ACTIVE_WAITING_USER, SupportEvent.USER_MESSAGE_AI): AI_ACTIVE_WAITING_USER,
    # user is ready again -> resume the SAME case, back to operator
    (AI_ACTIVE_WAITING_USER, SupportEvent.ESCALATE_RESUME): HUMAN_ACTIVE_WAITING_OPERATOR,

    # --- resolution confirmation ---------------------------------------
    (
        HUMAN_ACTIVE_WAITING_RESOLUTION,
        SupportEvent.USER_CONFIRMED_RESOLVED,
    ): CLOSED,
    (
        HUMAN_ACTIVE_WAITING_RESOLUTION,
        SupportEvent.USER_REJECTED_RESOLVED,
    ): HUMAN_ACTIVE_WAITING_OPERATOR,
    (
        HUMAN_ACTIVE_WAITING_RESOLUTION,
        SupportEvent.RESOLUTION_TIMEOUT,
    ): CLOSED,
}


# START_NEW_SESSION is defined for *every* state (see §B.2 of the design):
# it closes the current active conversation and opens a fresh one. It is
# handled by SessionService.close_and_open_new(), not by the table above.
STATE_INDEPENDENT_EVENTS: FrozenSet[SupportEvent] = frozenset(
    {SupportEvent.START_NEW_SESSION}
)


# ============================================================
# ERRORS
# ============================================================

class InvalidTransition(Exception):
    """Raised when an event is not valid for the current conversation state."""

    def __init__(
        self,
        state: ConversationState,
        event: SupportEvent,
        reason: str = "",
    ) -> None:
        self.state = state
        self.event = event
        self.reason = reason
        super().__init__(
            f"Invalid transition: event={event.value} "
            f"from state="
            f"{tuple(s.value if s is not None else None for s in state)}"
            + (f" ({reason})" if reason else "")
        )


class CaseAlreadyAttached(Exception):
    """Raised when attaching a Tracker case to a conversation that already
    points at a DIFFERENT case.

    The conversation -> case pointer is immutable once set: re-pointing it
    would silently orphan the operator thread that already owns the previous
    case. ``attach_case`` therefore refuses the overwrite and this error
    carries BOTH identity values so the caller (the /case handler) can return
    a precise 409 without re-querying.
    """

    def __init__(self, conversation_id: str, existing_case_id: str) -> None:
        self.conversation_id = conversation_id
        self.existing_case_id = existing_case_id
        super().__init__(
            f"Conversation {conversation_id!r} already attached to case "
            f"{existing_case_id!r}"
        )


# ============================================================
# PURE FSM FUNCTIONS
# ============================================================

def is_transition_allowed(
    state: ConversationState,
    event: SupportEvent,
) -> bool:
    """Return True if ``event`` is legal in ``state`` (or is state-independent)."""
    if event in STATE_INDEPENDENT_EVENTS:
        return True
    return (state, event) in TRANSITIONS


def next_state(
    state: ConversationState,
    event: SupportEvent,
) -> ConversationState:
    """
    Pure transition function.

    Returns the resulting canonical state, or raises ``InvalidTransition``.

    START_NEW_SESSION is not resolved here (it is state-independent and
    transactional); callers handle it via SessionService. If it is passed
    to this function we raise, to avoid returning a misleading state.
    """
    if event in STATE_INDEPENDENT_EVENTS:
        raise InvalidTransition(
            state,
            event,
            "state-independent event; handled by SessionService.close_and_open_new",
        )

    try:
        result = TRANSITIONS[(state, event)]
    except KeyError:
        raise InvalidTransition(state, event) from None

    # Defensive: the FSM must never emit a combination outside the whitelist.
    assert result in ALLOWED_TRISTATE_COMBINATIONS, (
        f"FSM produced a forbidden state combination: {result!r}"
    )
    return result
