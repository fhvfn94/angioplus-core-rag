"""allow forwarded inbound outcome

Revision ID: 6d2f871c8a35
Revises: 0001_support_state
Create Date: 2026-09-17 10:24:53.507685
"""

from __future__ import annotations

from alembic import op


revision = "6d2f871c8a35"
down_revision = "0001_support_state"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "ck_inbound_events_outcome",
        "inbound_events",
        type_="check",
    )

    op.create_check_constraint(
        "ck_inbound_events_outcome",
        "inbound_events",
        "outcome IN "
        "('received','answered','escalated','rejected','duplicate','forwarded')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_inbound_events_outcome",
        "inbound_events",
        type_="check",
    )

    op.create_check_constraint(
        "ck_inbound_events_outcome",
        "inbound_events",
        "outcome IN "
        "('received','answered','escalated','rejected','duplicate')",
    )