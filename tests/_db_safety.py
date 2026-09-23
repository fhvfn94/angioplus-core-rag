"""
Fail-closed safety guard for destructive DB-backed tests.

Background
----------
The DB-backed integration suite exercises real PostgreSQL and its fixtures
perform ``Base.metadata.drop_all`` + ``create_all`` before each test. A
forensic audit established that two pytest runs had ``TEST_DATABASE_URL``
pointing directly at the *production* database ``angioplus``, so the
destructive fixtures wiped live data.

This module makes that failure mode impossible by refusing to proceed
(raising :class:`TestDatabaseSafetyError`, which surfaces as a pytest
ERROR *before* any ``drop_all``/``create_all``) unless the configured
target is unambiguously a dedicated test database.

The guard is fail-closed: any uncertainty (missing URL, unparseable URL,
no explicit allowlist match, or a collision with ``DATABASE_URL``) refuses
to run rather than guessing.

Rules enforced
--------------
1. ``TEST_DATABASE_URL`` must be set and parseable.
2. Its database name must NOT be a forbidden/production name
   (``angioplus``, ``postgres``, ``template0``, ``template1``).
3. It must NOT resolve to the SAME physical database as ``DATABASE_URL``
   (same host/port/database, or same database name).
4. Its database name must be on the explicit allowlist
   (``{"angioplus_test"}`` by default; extend deliberately via the
   ``ALLOWED_TEST_DATABASES`` env var).

Nothing here connects to any database: it only inspects DSNs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError


TEST_DATABASE_URL_ENV = "TEST_DATABASE_URL"
DATABASE_URL_ENV = "DATABASE_URL"
ALLOWED_TEST_DATABASES_ENV = "ALLOWED_TEST_DATABASES"


class TestDatabaseSafetyError(RuntimeError):
    """Raised when a DB-backed test would run in an unsafe configuration.

    Raising (instead of skipping) is deliberate: it is better to hard-fail
    a test run than to silently skip and let a misconfigured environment
    look green.
    """

    # Tell pytest not to try to collect this as a test class (the name
    # starts with "Test" but it is an exception, not a test).
    __test__ = False


# Database names that must never be used as a destructive test target.
# ``angioplus`` is the production database; the others are system catalogs.
FORBIDDEN_DATABASE_NAMES = frozenset(
    {"angioplus", "postgres", "template0", "template1"}
)

# Explicit allowlist of acceptable test database names.
_DEFAULT_ALLOWED = frozenset({"angioplus_test"})


@dataclass(frozen=True)
class _ParsedUrl:
    driver: str
    host: str
    port: Optional[int]
    database: str

    @property
    def physical_key(self) -> str:
        """Identity of the *physical* database (ignores user/password)."""
        return f"{self.host}:{self.port or ''}/{self.database}"


def _allowed_names() -> frozenset[str]:
    raw = os.getenv(ALLOWED_TEST_DATABASES_ENV, "").strip()
    if not raw:
        return _DEFAULT_ALLOWED
    extra = {name.strip() for name in raw.split(",") if name.strip()}
    return frozenset(_DEFAULT_ALLOWED | extra)


def _parse(label: str, url: str) -> _ParsedUrl:
    try:
        parsed = make_url(url)
    except ArgumentError as exc:
        raise TestDatabaseSafetyError(
            f"{label} is not a valid SQLAlchemy URL: {url!r} ({exc})"
        ) from exc
    if not parsed.database:
        raise TestDatabaseSafetyError(
            f"{label} does not name a database: {url!r}"
        )
    return _ParsedUrl(
        driver=parsed.drivername,
        host=parsed.host or "",
        port=parsed.port,
        database=parsed.database,
    )


def get_test_database_url() -> str:
    """Return the raw ``TEST_DATABASE_URL`` (may be empty).

    Reading the value is intentionally separate from validating it; call
    :func:`guard_test_database` before any destructive use.
    """
    return os.getenv(TEST_DATABASE_URL_ENV, "").strip()


def guard_test_database(
    test_url: Optional[str] = None,
    prod_url: Optional[str] = None,
) -> str:
    """Validate the test database target and return the resolved URL.

    Fail-closed: raises :class:`TestDatabaseSafetyError` for any unsafe or
    ambiguous configuration. When the arguments are omitted the values are
    read from ``TEST_DATABASE_URL`` / ``DATABASE_URL``.

    Must be called *before* ``drop_all``/``create_all`` (the fixtures do).
    """
    if test_url is None:
        test_url = get_test_database_url()
    if prod_url is None:
        prod_url = os.getenv(DATABASE_URL_ENV, "").strip()

    # Rule 1: TEST_DATABASE_URL must be present.
    if not test_url:
        raise TestDatabaseSafetyError(
            "TEST_DATABASE_URL is not set. DB-backed tests are fail-closed: "
            "point it at a dedicated test database, e.g. "
            "'postgresql+asyncpg://angioplus_test:...@127.0.0.1:5432/angioplus_test'."
        )

    test = _parse(TEST_DATABASE_URL_ENV, test_url)

    # Rule 2: forbid known production/system database names.
    if test.database in FORBIDDEN_DATABASE_NAMES:
        raise TestDatabaseSafetyError(
            f"TEST_DATABASE_URL points at forbidden database "
            f"{test.database!r}. Destructive test setup (drop_all/create_all) "
            f"must never run against it."
        )

    # Rule 3: never collide with the configured production database.
    if prod_url:
        prod = _parse(DATABASE_URL_ENV, prod_url)
        if test.physical_key == prod.physical_key or test.database == prod.database:
            raise TestDatabaseSafetyError(
                "TEST_DATABASE_URL resolves to the SAME database as "
                f"DATABASE_URL ({test.database!r} @ {test.host}:{test.port}). "
                "Refusing to run destructive tests against the production "
                "database."
            )

    # Rule 4: explicit allowlist.
    allowed = _allowed_names()
    if test.database not in allowed:
        raise TestDatabaseSafetyError(
            f"TEST_DATABASE_URL database {test.database!r} is not an "
            f"explicitly allowed test database {sorted(allowed)}. Extend "
            f"{ALLOWED_TEST_DATABASES_ENV} deliberately if this is intended."
        )

    return test_url
