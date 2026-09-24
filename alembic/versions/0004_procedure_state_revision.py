"""procedure state revision: optimistic-concurrency column

Revision ID: 0004_procedure_state_revision
Revises: 0003_procedure_state_persistence
Create Date: 2026-01-08

Slice 9 (application boundary). Adds a monotonic per-conversation write
revision to ``procedure_states`` for compare-and-swap (CAS) concurrency
control, used by ``support.procedure_state_store.ProcedureStateStore``.

Single semantic model:

  * The FIRST persisted state of a conversation has ``revision = 1`` -- this
    includes rows written by the frozen ``ProcedureStateService``, which never
    sets the column and therefore relies on the ``server_default`` kept below.
  * Every successful CAS write sets ``revision = expected_revision + 1``.
  * ``revision`` is therefore ALWAYS >= 1; the CHECK constraint pins that.

Safety for existing rows: ``procedure_states`` is not written by any runtime
path (``ProcedureStateService`` is not imported by server.py / orchestrator /
n8n / Tracker), so production has no rows. If any row does exist, backfilling
it to ``1`` is lossless: it is indistinguishable from a fresh cold-start row,
and the next CAS write simply moves it to ``2``. No row ever holds ``0``.

This migration is purely additive: it does NOT alter ``0003`` or any earlier
revision.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0004_procedure_state_revision"
down_revision = "0003_procedure_state_persistence"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "procedure_states",
        sa.Column(
            "revision",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("1"),
        ),
    )
    op.create_check_constraint(
        "ck_procedure_states_revision",
        "procedure_states",
        "revision >= 1",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_procedure_states_revision",
        "procedure_states",
        type_="check",
    )
    op.drop_column("procedure_states", "revision")
