"""FSM unit tests: exhaustive transition table coverage (pure, no DB)."""

from __future__ import annotations

import pytest

from support.domain import (
    AI_ACTIVE,
    AI_ACTIVE_WAITING_USER,
    ALLOWED_TRISTATE_COMBINATIONS,
    CLOSED,
    HUMAN_ACTIVE_WAITING_OPERATOR,
    HUMAN_ACTIVE_WAITING_RESOLUTION,
    ConversationMode,
    HandoffStatus,
    InvalidTransition,
    SupportEvent,
    SupportStatus,
    TRANSITIONS,
    is_transition_allowed,
    next_state,
)


def test_whitelist_is_canonical_closed_single():
    # Only one canonical closed state; (human, closed, closed) is NOT allowed.
    assert CLOSED == (
        ConversationMode.AI,
        SupportStatus.CLOSED,
        HandoffStatus.CLOSED,
    )
    assert (
        ConversationMode.HUMAN,
        SupportStatus.CLOSED,
        HandoffStatus.CLOSED,
    ) not in ALLOWED_TRISTATE_COMBINATIONS


def test_every_transition_lands_in_whitelist():
    for (state, event), target in TRANSITIONS.items():
        assert state in ALLOWED_TRISTATE_COMBINATIONS, state
        assert target in ALLOWED_TRISTATE_COMBINATIONS, target


def test_waiting_user_is_ai_owned():
    # TRACKER_NEED_INFO -> (ai, active, waiting_user)
    result = next_state(
        HUMAN_ACTIVE_WAITING_OPERATOR, SupportEvent.TRACKER_NEED_INFO
    )
    assert result == AI_ACTIVE_WAITING_USER


def test_waiting_user_keeps_ai_chat():
    result = next_state(
        AI_ACTIVE_WAITING_USER, SupportEvent.USER_MESSAGE_AI
    )
    assert result == AI_ACTIVE_WAITING_USER


def test_waiting_user_resume_returns_to_operator():
    result = next_state(
        AI_ACTIVE_WAITING_USER, SupportEvent.ESCALATE_RESUME
    )
    assert result == HUMAN_ACTIVE_WAITING_OPERATOR


def test_escalate_new_from_ai():
    result = next_state(AI_ACTIVE, SupportEvent.ESCALATE_NEW)
    assert result == HUMAN_ACTIVE_WAITING_OPERATOR


def test_resolution_confirmed_closes():
    result = next_state(
        HUMAN_ACTIVE_WAITING_RESOLUTION,
        SupportEvent.USER_CONFIRMED_RESOLVED,
    )
    assert result == CLOSED


def test_resolution_rejected_returns_to_operator():
    result = next_state(
        HUMAN_ACTIVE_WAITING_RESOLUTION,
        SupportEvent.USER_REJECTED_RESOLVED,
    )
    assert result == HUMAN_ACTIVE_WAITING_OPERATOR


def test_resolution_timeout_closes():
    result = next_state(
        HUMAN_ACTIVE_WAITING_RESOLUTION,
        SupportEvent.RESOLUTION_TIMEOUT,
    )
    assert result == CLOSED


@pytest.mark.parametrize(
    "state,event",
    [
        (AI_ACTIVE, SupportEvent.USER_MESSAGE_HUMAN),
        (AI_ACTIVE, SupportEvent.ESCALATE_RESUME),
        (HUMAN_ACTIVE_WAITING_OPERATOR, SupportEvent.ESCALATE_NEW),
        (CLOSED, SupportEvent.USER_MESSAGE_AI),
        (AI_ACTIVE_WAITING_USER, SupportEvent.ESCALATE_NEW),
    ],
)
def test_invalid_transitions_raise(state, event):
    with pytest.raises(InvalidTransition):
        next_state(state, event)


def test_start_new_session_is_state_independent():
    for state in ALLOWED_TRISTATE_COMBINATIONS:
        assert is_transition_allowed(state, SupportEvent.START_NEW_SESSION)


def test_start_new_session_not_resolved_by_next_state():
    with pytest.raises(InvalidTransition):
        next_state(AI_ACTIVE, SupportEvent.START_NEW_SESSION)
