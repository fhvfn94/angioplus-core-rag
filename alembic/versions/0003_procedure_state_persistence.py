"""procedure state persistence: procedure_states, procedure_state_attempts

Revision ID: 0003_procedure_state_persistence
Revises: 6d2f871c8a35
Create Date: 2026-01-02

Durable projection of ``support.procedure_state.ProcedureState``:

  * ``procedure_states``         -- the CURRENT state, one row per conversation
    (``conversation_id`` is both PK and FK; ON DELETE CASCADE).
  * ``procedure_state_attempts`` -- the append-only attempt log, unique per
    ``(conversation_id, sequence)``, ON DELETE CASCADE from its state row.

The CHECK constraints whitelist the same enum values the domain uses
(``ProblemStatus`` / ``StepOutcome``), mirroring how the FSM triple is pinned
in ``0001_support_state``.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0003_procedure_state_persistence"
down_revision = "6d2f871c8a35"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ------------------------------------------------------------
    # procedure_states
    # ------------------------------------------------------------
    op.create_table(
        "procedure_states",
        sa.Column("conversation_id", sa.Text(), nullable=False),
        sa.Column("issue", sa.Text(), nullable=True),
        sa.Column("procedure_id", sa.Text(), nullable=True),
        sa.Column("procedure_version", sa.Integer(), nullable=True),
        sa.Column("last_recommended_step", sa.Text(), nullable=True),
        sa.Column(
            "problem_status",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'unknown'"),
        ),
        sa.Column(
            "pending_clarification",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("clarification_topic", sa.Text(), nullable=True),
        sa.Column(
            "schema_version",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("1"),
        ),
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
            "problem_status IN ('unknown','ongoing','resolved')",
            name="ck_procedure_states_problem_status",
        ),
        sa.CheckConstraint(
            "schema_version >= 1",
            name="ck_procedure_states_schema_version",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.conversation_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("conversation_id"),
    )

    # ------------------------------------------------------------
    # procedure_state_attempts
    # ------------------------------------------------------------
    op.create_table(
        "procedure_state_attempts",
        sa.Column(
            "id",
            sa.BigInteger(),
            sa.Identity(),
            primary_key=True,
            autoincrement=True,
        ),
        sa.Column("conversation_id", sa.Text(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("step_id", sa.Text(), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            "outcome IN ('failed','succeeded','skipped','done_unknown')",
            name="ck_procedure_state_attempts_outcome",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["procedure_states.conversation_id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "conversation_id",
            "sequence",
            name="uq_procedure_state_attempts_seq",
        ),
    )
    op.create_index(
        "ix_procedure_state_attempts_conversation",
        "procedure_state_attempts",
        ["conversation_id", "sequence"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_procedure_state_attempts_conversation",
        table_name="procedure_state_attempts",
    )
    op.drop_table("procedure_state_attempts")
    op.drop_table("procedure_states")
