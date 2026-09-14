"""
Channel- and provider-agnostic support domain.

This package owns the canonical lifecycle of support conversations:
their persistence, finite-state machine, message log and inbound-event
idempotency. It has no dependency on any specific channel (Telegram,
Web, Email, Voice) or LLM provider (YandexGPT, ...).

Consumers:
  - yandex/chat/server.py   (Yandex text path)
  - n8n workflows           (transport / integration layer)
  - future web / email / voice adapters
"""

from __future__ import annotations

from support.domain import (
    ALLOWED_TRISTATE_COMBINATIONS,
    Channel,
    ConversationMode,
    ConversationState,
    HandoffStatus,
    InvalidTransition,
    MessageRole,
    SupportEvent,
    SupportStatus,
    is_transition_allowed,
    next_state,
)

__all__ = [
    "ConversationMode",
    "SupportStatus",
    "HandoffStatus",
    "MessageRole",
    "Channel",
    "SupportEvent",
    "ConversationState",
    "InvalidTransition",
    "next_state",
    "is_transition_allowed",
    "ALLOWED_TRISTATE_COMBINATIONS",
]
