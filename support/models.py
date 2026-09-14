"""
SQLAlchemy 2.x ORM models for the support domain.

Tables (FINAL DESIGN v2 + DELTA v3):

  conversations    -- lifecycle / FSM state of a support conversation
  messages         -- append-only message log
  inbound_events   -- inbound delivery idempotency (user -> backend)
  processed_events -- external provider callback idempotency (e.g. Tracker)

Key invariants enforced at the DB level:

  * ``conversations.external_conversation_id`` is NOT NULL (no NULL loophole
    in the partial unique index).
  * partial unique active-session index on
    (channel, external_user_id, external_conversation_id) WHERE active.
  * the triple (conversation_mode, support_status, handoff_status) is
    constrained to the same whitelist of real combinations used by the FSM.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from support.db import Base


# ============================================================
# CONSTANTS SHARED WITH THE DOMAIN
# ============================================================

# The valid triple whitelist as a SQL expression. Mirrors
# support.domain.ALLOWED_TRISTATE_COMBINATIONS. Kept here as a literal so the
# migration and the model agree; the unit test cross-checks the two.
_VALID_TRISTATE_SQL = (
    "("
    "(conversation_mode = 'ai'     AND support_status = 'active' AND handoff_status IS NULL)"
    " OR (conversation_mode = 'ai'    AND support_status = 'active' AND handoff_status = 'waiting_user')"
    " OR (conversation_mode = 'human' AND support_status = 'active' AND handoff_status = 'waiting_operator')"
    " OR (conversation_mode = 'human' AND support_status = 'active' AND handoff_status = 'waiting_resolution_confirmation')"
    " OR (conversation_mode = 'ai'    AND support_status = 'closed' AND handoff_status = 'closed')"
    ")"
)


# ============================================================
# conversations
# ============================================================

class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
        autoincrement=True,
    )

    # Business key, produced by the caller/adapter (e.g. "tg-<uid>-<ts>").
    conversation_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)

    channel: Mapped[str] = mapped_column(Text, nullable=False)
    external_user_id: Mapped[str] = mapped_column(Text, nullable=False)

    # Distinguishes "who" from "which thread". NOT NULL by design: channels
    # without a thread concept must set this to external_user_id.
    external_conversation_id: Mapped[str] = mapped_column(Text, nullable=False)

    conversation_mode: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'ai'")
    )
    support_status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'active'")
    )
    handoff_status: Mapped[str | None] = mapped_column(Text, nullable=True)

    operator_provider: Mapped[str | None] = mapped_column(Text, nullable=True)
    external_case_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolution_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        CheckConstraint(
            "conversation_mode IN ('ai','human')",
            name="ck_conversations_mode",
        ),
        CheckConstraint(
            "support_status IN ('active','closed')",
            name="ck_conversations_support_status",
        ),
        CheckConstraint(
            "handoff_status IS NULL OR handoff_status IN "
            "('waiting_operator','waiting_user',"
            "'waiting_resolution_confirmation','closed')",
            name="ck_conversations_handoff_status",
        ),
        CheckConstraint(
            _VALID_TRISTATE_SQL,
            name="ck_conversations_valid_tristate",
        ),
        Index("ix_conversations_external_case_id", "external_case_id"),
        Index(
            "ix_conversations_ext_conv",
            "channel",
            "external_conversation_id",
        ),
        # At most ONE active conversation per (channel, user, thread).
        # Closed conversations are unlimited.
        Index(
            "uq_conversations_active_session",
            "channel",
            "external_user_id",
            "external_conversation_id",
            unique=True,
            postgresql_where=text("support_status = 'active'"),
        ),
    )


# ============================================================
# messages
# ============================================================

class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
        autoincrement=True,
    )
    conversation_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("conversations.conversation_id", ondelete="CASCADE"),
        nullable=False,
    )
    external_user_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "role IN ('user','assistant','operator')",
            name="ck_messages_role",
        ),
        Index(
            "ix_messages_conversation_created",
            "conversation_id",
            "created_at",
        ),
    )


# ============================================================
# inbound_events
# ============================================================

class InboundEvent(Base):
    __tablename__ = "inbound_events"

    # Per-channel delivery key:
    #   telegram -> tg:<update_id>
    #   web      -> web:<request_uuid>
    #   email    -> email:<Message-ID>
    #   voice    -> voice:<call_id>:<turn_id>
    inbound_event_id: Mapped[str] = mapped_column(Text, primary_key=True)

    channel: Mapped[str] = mapped_column(Text, nullable=False)
    conversation_id: Mapped[str | None] = mapped_column(Text, nullable=True)

    # received | answered | escalated | rejected | duplicate
    outcome: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'received'")
    )

    # Reference to the produced answer (single source of truth for answer text).
    assistant_message_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Small result metadata of the turn (answer text is NOT stored here).
    decision: Mapped[str | None] = mapped_column(Text, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    case_action: Mapped[str | None] = mapped_column(Text, nullable=True)

    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        CheckConstraint(
            "outcome IN "
            "('received','answered','escalated','rejected','duplicate')",
            name="ck_inbound_events_outcome",
        ),
        Index("ix_inbound_events_conversation", "conversation_id"),
    )


# ============================================================
# processed_events (external provider callbacks)
# ============================================================

class ProcessedEvent(Base):
    __tablename__ = "processed_events"

    # Provider event key, e.g. "tracker:ANGIO-9:6aa30..."
    event_key: Mapped[str] = mapped_column(Text, primary_key=True)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    external_case_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    conversation_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_processed_events_conversation", "conversation_id"),
    )
