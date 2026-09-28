"""turn recovery: durable per-execution recovery record (HL-2c.3a)

Revision ID: 0006_turn_recovery
Revises: 0005_procedure_state_exec_id
Create Date: 2026-01-09

Slice HL-2c.3a (W3 crash-window recovery, support/runtime boundary ONLY).

Adds a DEDICATED ``turn_recovery`` table keyed by the OPAQUE
``execution_id`` -- the SAME identity ``procedure_states.last_execution_id``
(HL-2b) records for idempotency -- holding the MINIMUM material needed to
reproduce the ORIGINAL runtime turn of that execution WITHOUT re-running the
LLM / retrieval / procedure:

  * the engine's canonical result: ``text`` / ``action`` /
    ``escalate_reason`` / ``clarify_reason``;
  * the runtime PROJECTION SNAPSHOT: ``projection_mode`` /
    ``projection_support_status`` / ``projection_handoff_status``.

The projection triple is ``ConversationState`` (runtime / human lifecycle),
NOT ``ProcedureState``: it is snapshotted here precisely so the FSM projection
can be replayed EXACTLY as it was, even if the CURRENT conversation state has
moved on. It is deliberately NOT a column of ``procedure_states``.

OWNERSHIP: written by ``support.turn_recovery_store.TurnRecoveryStore`` on the
SAME session / transaction as the ProcedureState CAS, so the two COMMIT OR
ROLL BACK TOGETHER and no crash window can leave the procedure state advanced
without its recovery record. ``ProcedureStateStore`` is NOT modified by this
revision and stays unaware of recovery / ``ConversationState``.

Purely additive: no existing table or earlier revision is altered and there is
no backfill (rows exist only for executions that recorded a recovery record).

NOTE ON THE IDENTIFIER: the revision id is abbreviated to
``0006_turn_recovery`` (18 characters) so that it FITS Alembic's default
``alembic_version.version_num`` column (``VARCHAR(32)``). Guarded offline by
``tests/test_alembic_revision_id_length.py``.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0006_turn_recovery"
down_revision = "0005_procedure_state_exec_id"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "turn_recovery",
        sa.Column("execution_id", sa.Text(), nullable=False),
        sa.Column("conversation_id", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("escalate_reason", sa.Text(), nullable=True),
        sa.Column("clarify_reason", sa.Text(), nullable=True),
        sa.Column("projection_mode", sa.Text(), nullable=False),
        sa.Column("projection_support_status", sa.Text(), nullable=False),
        sa.Column("projection_handoff_status", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("execution_id", name="pk_turn_recovery"),
    )


def downgrade() -> None:
    op.drop_table("turn_recovery")
