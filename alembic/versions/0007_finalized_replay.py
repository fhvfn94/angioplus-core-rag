"""inbound events: finalized replay snapshot (HL-2c.3b.1)

Revision ID: 0007_finalized_replay
Revises: 0006_turn_recovery
Create Date: 2026-01-09

Slice HL-2c.3b.1 (LEVEL A finalized runtime replay, support/runtime boundary
ONLY).

Adds the MINIMUM durable snapshot columns to ``inbound_events`` so a duplicate
delivery of an ALREADY-finalized chat turn reproduces the EXACT original HTTP
body WITHOUT re-running the LLM / retrieval / wording / FSM / Tracker:

  * ``retrieval_used`` -- the legacy HTTP compatibility field that was never
    persisted before (the documented fidelity gap);
  * ``conversation_mode`` / ``support_status`` / ``handoff_status`` -- the
    lifecycle SNAPSHOT as it stood when the turn was finalized (the live
    conversation may have moved on since);
  * ``external_case_id`` -- the Tracker key the body echoed.

The answer text is deliberately NOT duplicated here: it stays in ``messages``
and is linked by the existing ``inbound_events.assistant_message_id``.

Purely additive and NULLABLE: existing (legacy) rows keep their current
columns and get NULL for all five new ones. There is no backfill -- a legacy
finalized row therefore cannot be replayed silently; the reader fails closed
with ``FinalizedReplayCorruption`` instead of fabricating a lossy body.

NOTE ON THE IDENTIFIER: the revision id is abbreviated to
``0007_finalized_replay`` (21 characters) so that it FITS Alembic's default
``alembic_version.version_num`` column (``VARCHAR(32)``). Guarded offline by
``tests/test_alembic_revision_id_length.py``.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0007_finalized_replay"
down_revision = "0006_turn_recovery"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "inbound_events",
        sa.Column("retrieval_used", sa.Boolean(), nullable=True),
    )
    op.add_column(
        "inbound_events",
        sa.Column("conversation_mode", sa.Text(), nullable=True),
    )
    op.add_column(
        "inbound_events",
        sa.Column("support_status", sa.Text(), nullable=True),
    )
    op.add_column(
        "inbound_events",
        sa.Column("handoff_status", sa.Text(), nullable=True),
    )
    op.add_column(
        "inbound_events",
        sa.Column("external_case_id", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("inbound_events", "external_case_id")
    op.drop_column("inbound_events", "handoff_status")
    op.drop_column("inbound_events", "support_status")
    op.drop_column("inbound_events", "conversation_mode")
    op.drop_column("inbound_events", "retrieval_used")
