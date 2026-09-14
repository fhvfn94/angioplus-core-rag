"""
Pydantic v2 schemas for the support service layer.

These are framework-agnostic value objects used between the HTTP layer
(``yandex/chat/server.py``) and the support services. They intentionally
do not depend on aiohttp or any other web framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, ConfigDict


# ============================================================
# CORE VALUE OBJECTS
# ============================================================

@dataclass(frozen=True)
class ConversationView:
    """Read-only view of a conversation row."""

    conversation_id: str
    channel: str
    external_user_id: str
    external_conversation_id: str
    conversation_mode: str
    support_status: str
    handoff_status: Optional[str]
    operator_provider: Optional[str]
    external_case_id: Optional[str]


@dataclass(frozen=True)
class ClaimResult:
    """Result of claiming an inbound event."""

    claimed: bool
    outcome: Optional[str]
    prior: Optional["ProcessedTurnResult"] = None


@dataclass(frozen=True)
class ProcessedTurnResult:
    """
    Cached metadata of a previously processed turn.

    ``answer`` is not stored here; it lives in ``messages`` and is fetched
    via ``assistant_message_id``. This object only carries the small
    decision metadata.
    """

    answer: Optional[str]
    decision: Optional[str]
    reason: Optional[str]
    case_action: Optional[str]
    conversation_id: Optional[str]


# ============================================================
# REQUEST / RESPONSE
# ============================================================

class ChatRequest(BaseModel):
    """
    Stateless-extended request for POST /api/chat.

    ``message`` is the only required field; everything else is optional so
    that legacy ``{"message": "..."}`` callers keep working unchanged.
    """

    model_config = ConfigDict(extra="ignore")

    message: str

    channel: Optional[str] = None
    external_user_id: Optional[str] = None
    external_conversation_id: Optional[str] = None
    conversation_id: Optional[str] = None

    inbound_event_id: Optional[str] = None
    inbound_message_id: Optional[str] = None

    @property
    def stateful(self) -> bool:
        return bool(self.channel)


class ChatResponse(BaseModel):
    """Response for POST /api/chat (additive over the legacy shape)."""

    model_config = ConfigDict(extra="allow")

    ok: bool
    answer: str
    retrieval_used: bool
    decision: str
    reason: str

    # stateful extras (absent in legacy mode)
    conversation_id: Optional[str] = None
    conversation_mode: Optional[str] = None
    support_status: Optional[str] = None
    handoff_status: Optional[str] = None
    case_action: Optional[str] = None
    deduplicated: Optional[bool] = None
