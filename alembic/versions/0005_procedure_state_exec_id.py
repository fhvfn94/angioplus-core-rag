"""procedure state: opaque per-turn execution identity (idempotent CAS)

Revision ID: 0005_procedure_state_exec_id
Revises: 0004_procedure_state_revision
Create Date: 2026-01-09

Slice HL-2b (idempotent turn execution, primitives only). Adds an OPAQUE
execution identity to ``procedure_states`` so a repeated write of the SAME
execution id is an idempotent no-op instead of a second revision bump.

  * The column is NULLable and carries NO channel/runtime meaning: the store
    (``support.procedure_state_store.ProcedureStateStore``) records it on the
    CAS write and compares it only for equality. Callers that never supply an
    execution id keep the previous behaviour exactly (value stays NULL).
  * The store writes it ATOMICALLY with the state (same conditional UPDATE /
    INSERT), so no second transaction is introduced and provider calls still
    happen outside any DB transaction.
  * The UNIQUE index makes ONE execution id advance at most ONE
    conversation's state. PostgreSQL treats NULLs as distinct, so any number
    of writers that do not use idempotency may share NULL.

Purely additive: it does NOT alter ``0004`` or any earlier revision, and it
adds NO backfill (existing rows -- none exist in production, since
``procedure_states`` is not written by any runtime path -- simply keep NULL).

NOTE ON THE IDENTIFIER: the revision id is abbreviated to
``0005_procedure_state_exec_id`` so that it FITS Alembic's default
``alembic_version.version_num`` column (``VARCHAR(32)``). The earlier, longer
id ``0005_procedure_state_last_execution_id`` (38 characters) made
``alembic upgrade head`` fail on a clean database while persisting the
revision, with ``asyncpg.StringDataRightTruncationError: value too long for
type character varying(32)``. The abbreviated id still identifies migration
0005 and keeps ``down_revision = "0004_procedure_state_revision"``. Guarded
offline by ``tests/test_alembic_revision_id_length.py``.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0005_procedure_state_exec_id"
down_revision = "0004_procedure_state_revision"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "procedure_states",
        sa.Column("last_execution_id", sa.Text(), nullable=True),
    )
    op.create_index(
        "uq_procedure_states_last_execution_id",
        "procedure_states",
        ["last_execution_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(
        "uq_procedure_states_last_execution_id",
        table_name="procedure_states",
    )
    op.drop_column("procedure_states", "last_execution_id")
