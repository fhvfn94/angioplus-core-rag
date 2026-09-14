"""support state: conversations, messages, inbound_events, processed_events

Revision ID: 0001_support_state
Revises:
Create Date: 2026-01-01

Phase 1 / Changeset #1. Creates the channel-agnostic support tables with:
  * NOT NULL external_conversation_id (no NULL loophole)
  * partial unique active-session index
  * CHECK constraints whitelisting the real FSM state combinations
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0001_support_state"
down_revision = None
branch_labels = None
depends_on = None


_VALID_TRISTATE_SQL = (
    "("
    "(conversation_mode = 'ai'     AND support_status = 'active' AND handoff_status IS NULL)"
    " OR (conversation_mode = 'ai'    AND support_status = 'active' AND handoff_status = 'waiting_user')"
    " OR (conversation_mode = 'human' AND support_status = 'active' AND handoff_status = 'waiting_operator')"
    " OR (conversation_mode = 'human' AND support_status = 'active' AND handoff_status = 'waiting_resolution_confirmation')"
    " OR (conversation_mode = 'ai'    AND support_status = 'closed' AND handoff_status = 'closed')"
    ")"
)


def upgrade() -> None:
    # ------------------------------------------------------------
    # conversations
    # ------------------------------------------------------------
    op.create_table(
        "conversations",
        sa.Column(
            "id",
            sa.BigInteger(),
            sa.Identity(),
            primary_key=True,
            autoincrement=True,
        ),
        sa.Column("conversation_id", sa.Text(), nullable=False, unique=True),
        sa.Column("channel", sa.Text(), nullable=False),
        sa.Column("external_user_id", sa.Text(), nullable=False),
        sa.Column("external_conversation_id", sa.Text(), nullable=False),
        sa.Column(
            "conversation_mode",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'ai'"),
        ),
        sa.Column(
            "support_status",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'active'"),
        ),
        sa.Column("handoff_status", sa.Text(), nullable=True),
        sa.Column("operator_provider", sa.Text(), nullable=True),
        sa.Column("external_case_id", sa.Text(), nullable=True),
        sa.Column("resolution_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            "conversation_mode IN ('ai','human')",
            name="ck_conversations_mode",
        ),
        sa.CheckConstraint(
            "support_status IN ('active','closed')",
            name="ck_conversations_support_status",
        ),
        sa.CheckConstraint(
            "handoff_status IS NULL OR handoff_status IN "
            "('waiting_operator','waiting_user',"
            "'waiting_resolution_confirmation','closed')",
            name="ck_conversations_handoff_status",
        ),
        sa.CheckConstraint(
            _VALID_TRISTATE_SQL,
            name="ck_conversations_valid_tristate",
        ),
    )
    op.create_index(
        "ix_conversations_external_case_id",
        "conversations",
        ["external_case_id"],
    )
    op.create_index(
        "ix_conversations_ext_conv",
        "conversations",
        ["channel", "external_conversation_id"],
    )
    op.create_index(
        "uq_conversations_active_session",
        "conversations",
        ["channel", "external_user_id", "external_conversation_id"],
        unique=True,
        postgresql_where=sa.text("support_status = 'active'"),
    )

    # ------------------------------------------------------------
    # messages
    # ------------------------------------------------------------
    op.create_table(
        "messages",
        sa.Column(
            "id",
            sa.BigInteger(),
            sa.Identity(),
            primary_key=True,
            autoincrement=True,
        ),
        sa.Column("conversation_id", sa.Text(), nullable=False),
        sa.Column("external_user_id", sa.Text(), nullable=True),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            "role IN ('user','assistant','operator')",
            name="ck_messages_role",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.conversation_id"],
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_messages_conversation_created",
        "messages",
        ["conversation_id", "created_at"],
    )

    # ------------------------------------------------------------
    # inbound_events
    # ------------------------------------------------------------
    op.create_table(
        "inbound_events",
        sa.Column("inbound_event_id", sa.Text(), primary_key=True),
        sa.Column("channel", sa.Text(), nullable=False),
        sa.Column("conversation_id", sa.Text(), nullable=True),
        sa.Column(
            "outcome",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'received'"),
        ),
        sa.Column("assistant_message_id", sa.BigInteger(), nullable=True),
        sa.Column("decision", sa.Text(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("case_action", sa.Text(), nullable=True),
        sa.Column(
            "received_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "outcome IN "
            "('received','answered','escalated','rejected','duplicate')",
            name="ck_inbound_events_outcome",
        ),
        sa.ForeignKeyConstraint(
            ["assistant_message_id"],
            ["messages.id"],
            ondelete="SET NULL",
        ),
    )
    op.create_index(
        "ix_inbound_events_conversation",
        "inbound_events",
        ["conversation_id"],
    )

    # ------------------------------------------------------------
    # processed_events
    # ------------------------------------------------------------
    op.create_table(
        "processed_events",
        sa.Column("event_key", sa.Text(), primary_key=True),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("external_case_id", sa.Text(), nullable=True),
        sa.Column("conversation_id", sa.Text(), nullable=True),
        sa.Column(
            "processed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_processed_events_conversation",
        "processed_events",
        ["conversation_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_processed_events_conversation", table_name="processed_events")
    op.drop_table("processed_events")

    op.drop_index("ix_inbound_events_conversation", table_name="inbound_events")
    op.drop_table("inbound_events")

    op.drop_index("ix_messages_conversation_created", table_name="messages")
    op.drop_table("messages")

    op.drop_index("uq_conversations_active_session", table_name="conversations")
    op.drop_index("ix_conversations_ext_conv", table_name="conversations")
    op.drop_index("ix_conversations_external_case_id", table_name="conversations")
    op.drop_table("conversations")
