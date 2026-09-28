"""
SQLAlchemy 2.x ORM models for the support domain.

Tables (FINAL DESIGN v2 + DELTA v3):

  conversations            -- lifecycle / FSM state of a support conversation
  messages                 -- append-only message log
  inbound_events           -- inbound delivery idempotency (user -> backend)
  processed_events         -- external provider callback idempotency (e.g. Tracker)
  procedure_states         -- current ProcedureState per conversation (1:1)
  procedure_state_attempts -- append-only log of recorded step attempts

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
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
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
            "('received','answered','escalated','rejected','duplicate','forwarded')",
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


# ============================================================
# procedure_states (current per-conversation procedure state)
# ============================================================

class ProcedureStateRow(Base):
    """The CURRENT :class:`support.procedure_state.ProcedureState`.

    The domain value object is a frozen dataclass with a tuple of
    :class:`support.procedure_state.StepAttempt`; it is not itself
    persistable. This table (plus :class:`ProcedureStateAttemptRow`) is its
    durable projection, written and read by
    :class:`support.procedure_state_service.ProcedureStateService`.

    One row per conversation: ``conversation_id`` is both the primary key
    and a FK, so a conversation owns at most one state row.
    """

    __tablename__ = "procedure_states"

    conversation_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("conversations.conversation_id", ondelete="CASCADE"),
        primary_key=True,
    )

    # --- which procedure is being walked (identity only; no cursor) ---
    issue: Mapped[str | None] = mapped_column(Text, nullable=True)
    procedure_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    procedure_version: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # --- what the assistant last advised (a fact about the dialogue) ---
    last_recommended_step: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- problem-level fact ---
    problem_status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'unknown'")
    )

    # --- clarification fact ---
    pending_clarification: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    clarification_topic: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- schema ---
    schema_version: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("1")
    )

    # --- persistence concurrency (Slice 9) ---
    # Monotonic per-conversation write revision for OPTIMISTIC concurrency
    # control (compare-and-swap), owned by
    # :class:`support.procedure_state_store.ProcedureStateStore`. The FIRST
    # persisted state has revision 1 (the server default keeps the frozen
    # ProcedureStateService -- which never sets it -- on the same semantic
    # model); every successful CAS write sets
    # ``revision = expected_revision + 1``. This is a persistence concern and
    # is deliberately NOT part of the frozen ``ProcedureState`` value object.
    revision: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("1")
    )

    # --- per-turn idempotency identity (HL-2b) ---
    # An OPAQUE execution id recorded on the LAST CAS write, written
    # atomically with the state (same UPDATE / INSERT). It carries NO
    # channel or runtime meaning: the store never interprets it, and a
    # caller that supplies none leaves it NULL (so callers that do not use
    # idempotency are entirely unaffected). The store uses it ONLY to make
    # a repeated write of the SAME execution id an idempotent no-op -- no
    # second revision bump, no attempt-log rewrite. GLOBALLY UNIQUE (see
    # __table_args__) so ONE execution can advance at most ONE
    # conversation's state. Not part of the frozen ``ProcedureState`` value
    # object; a persistence concern like ``revision``.
    last_execution_id: Mapped[str | None] = mapped_column(Text, nullable=True)

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
            "problem_status IN ('unknown','ongoing','resolved')",
            name="ck_procedure_states_problem_status",
        ),
        CheckConstraint(
            "schema_version >= 1",
            name="ck_procedure_states_schema_version",
        ),
        CheckConstraint(
            "revision >= 1",
            name="ck_procedure_states_revision",
        ),
        # Globally unique: one execution id advances at most one state row.
        # NULLs are distinct in PostgreSQL, so any number of non-idempotent
        # writers may leave ``last_execution_id`` NULL.
        Index(
            "uq_procedure_states_last_execution_id",
            "last_execution_id",
            unique=True,
        ),
    )


# ============================================================
# procedure_state_attempts (append-only step-attempt log)
# ============================================================

class ProcedureStateAttemptRow(Base):
    """One recorded :class:`support.procedure_state.StepAttempt`.

    Append-only: rows are only ever inserted (or replaced wholesale as a set
    by :class:`support.procedure_state_service.ProcedureStateService`); the
    ``sequence`` mirrors ``StepAttempt.sequence`` (0-based position within
    the state's attempt log, for ordering only).
    """

    __tablename__ = "procedure_state_attempts"

    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
        autoincrement=True,
    )
    conversation_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("procedure_states.conversation_id", ondelete="CASCADE"),
        nullable=False,
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    step_id: Mapped[str] = mapped_column(Text, nullable=False)
    outcome: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "outcome IN ('failed','succeeded','skipped','done_unknown')",
            name="ck_procedure_state_attempts_outcome",
        ),
        UniqueConstraint(
            "conversation_id",
            "sequence",
            name="uq_procedure_state_attempts_seq",
        ),
        Index(
            "ix_procedure_state_attempts_conversation",
            "conversation_id",
            "sequence",
        ),
    )


# ============================================================
# TURN RECOVERY (HL-2c.3a)
# ============================================================

class TurnRecoveryRow(Base):
    """The DURABLE recovery record of ONE applied turn execution.

    A SEPARATE persistence concern from :class:`ProcedureStateRow`. This table
    is keyed by the OPAQUE ``execution_id`` (the idempotency identity already
    used by ``ProcedureStateStore.save_cas``) and holds the MINIMUM material
    needed to reproduce the ORIGINAL runtime turn of that execution WITHOUT
    re-running the LLM / retrieval / procedure:

      * the engine's canonical result -- ``text`` / ``action`` /
        ``escalate_reason`` / ``clarify_reason``;
      * the runtime PROJECTION SNAPSHOT -- ``projection_mode`` /
        ``projection_support_status`` / ``projection_handoff_status``, i.e. the
        ``ConversationState`` triple as it stood at CAS time. Those are runtime
        / human-lifecycle state, NOT procedure progress, so they are
        deliberately NOT columns of ``procedure_states``. They are captured
        here because the FSM projection is state-dependent and the CURRENT
        conversation state may have moved on by recovery time.

    Written by ``support.turn_recovery_store.TurnRecoveryStore`` on the SAME
    session / transaction as the ProcedureState CAS, so the two commit or roll
    back together: there is never a crash window where the procedure state
    advanced but this record is missing.
    """

    __tablename__ = "turn_recovery"

    execution_id: Mapped[str] = mapped_column(Text, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(Text, nullable=False)

    # --- canonical engine result (what the projection renders) ---
    text: Mapped[str] = mapped_column(Text, nullable=False)
    action: Mapped[str] = mapped_column(Text, nullable=False)
    escalate_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    clarify_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- runtime projection snapshot (ConversationState at CAS time) ---
    projection_mode: Mapped[str] = mapped_column(Text, nullable=False)
    projection_support_status: Mapped[str] = mapped_column(
        Text, nullable=False
    )
    projection_handoff_status: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
