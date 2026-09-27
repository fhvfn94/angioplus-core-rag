"""
Regression guard: every Alembic revision id must fit ``version_num``.

Background (real defect, HL-2b)
-------------------------------
The uncommitted ``0005`` migration declared
``revision = "0005_procedure_state_last_execution_id"`` (38 characters).
Alembic persists the revision id into ``alembic_version.version_num``, whose
effective type is ``VARCHAR(32)``, so a manual clean-database
``alembic upgrade head`` failed while recording the revision:

    asyncpg.exceptions.StringDataRightTruncationError:
    value too long for type character varying(32)

This module is a pure, OFFLINE guard (no database, no network). It:

  * DERIVES the effective ``version_num`` capacity from Alembic's own
    version-table definition. The repository overrides neither the version
    table nor its structure (``alembic.ini`` sets no ``version_table``;
    ``alembic/env.py`` passes no custom version-table impl), so the dialect
    default -- ``VARCHAR(32)`` -- applies; and
  * enumerates every revision declared in ``alembic/versions`` (via Alembic's
    own script loader, so it is exactly the revision map Alembic builds) and
    asserts each identifier fits that capacity.

It therefore fails if any revision id -- present or future -- would overflow
``alembic_version.version_num``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from alembic.config import Config
from alembic.ddl.postgresql import PostgresqlImpl
from alembic.script import ScriptDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = REPO_ROOT / "alembic.ini"
ALEMBIC_DIR = REPO_ROOT / "alembic"

# The revision the HL-2b migration chains onto (must stay unchanged).
HL2B_PREVIOUS_REVISION = "0004_procedure_state_revision"
# The corrected HL-2b revision identifier.
HL2B_REVISION = "0005_procedure_state_exec_id"
# The invalid identifier that triggered this guard.
HL2B_INVALID_REVISION = "0005_procedure_state_last_execution_id"


def _effective_version_num_capacity() -> int:
    """Length limit Alembic applies to ``alembic_version.version_num``.

    Derived from Alembic's own version-table definition for the PostgreSQL
    dialect (this repository installs no custom version table and overrides
    no structure), so it tracks Alembic rather than a hard-coded number.
    """
    impl = object.__new__(PostgresqlImpl)
    table = impl.version_table_impl(
        version_table="alembic_version",
        version_table_schema=None,
        version_table_pk=True,
    )
    return int(table.c.version_num.type.length)


def _declared_revisions() -> dict[str, object]:
    """Map every declared ``revision`` id to its ``down_revision``.

    Loads the repository's real ``alembic.ini`` (with an absolute
    ``script_location`` so the result is independent of the current working
    directory) and uses Alembic's own script loader, so it reflects exactly
    the revision map Alembic would build. No database connection is made.
    """
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(ALEMBIC_DIR))
    # The repository's ini uses the legacy ``prepend_sys_path`` key, which
    # makes Alembic emit an unrelated DeprecationWarning on load; keep the
    # guard output focused on revision-id length.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        script = ScriptDirectory.from_config(config)
        return {
            rev.revision: rev.down_revision for rev in script.walk_revisions()
        }


def test_derived_capacity_is_a_positive_integer():
    capacity = _effective_version_num_capacity()
    assert isinstance(capacity, int)
    assert capacity > 0


def test_every_revision_id_fits_version_num():
    capacity = _effective_version_num_capacity()
    revisions = _declared_revisions()

    assert revisions, "no Alembic revisions were discovered"

    too_long = {rev: len(rev) for rev in revisions if len(rev) > capacity}
    assert not too_long, (
        "Alembic revision id(s) exceed alembic_version.version_num "
        f"capacity ({capacity}): {too_long}. Shorten the id so that "
        "`alembic upgrade head` can persist it."
    )


def test_hl2b_revision_id_is_concise_and_chained():
    capacity = _effective_version_num_capacity()
    revisions = _declared_revisions()

    assert len(HL2B_REVISION) <= capacity
    assert HL2B_REVISION in revisions, "HL-2b revision id not found"
    # The abbreviated id still chains onto 0004, unchanged.
    assert revisions[HL2B_REVISION] == HL2B_PREVIOUS_REVISION
    # The offending long id is gone from the revision map.
    assert HL2B_INVALID_REVISION not in revisions


def test_guard_would_have_caught_the_original_defect():
    # Documents WHY this guard exists: the original id did not fit.
    capacity = _effective_version_num_capacity()
    assert len(HL2B_INVALID_REVISION) > capacity
