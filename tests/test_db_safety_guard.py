"""
Regression tests for the fail-closed test-database safety guard.

These are PURE unit tests: they never connect to any database. They lock in
the behaviour that protects the production database from the destructive
DB-backed fixtures (``drop_all``/``create_all``).

The core invariant under test: the guard MUST refuse to run (raise) for the
production database and any ambiguous configuration, and MUST allow only an
explicitly allowlisted dedicated test database.
"""

from __future__ import annotations

import pytest

from _db_safety import (
    FORBIDDEN_DATABASE_NAMES,
    TestDatabaseSafetyError,
    guard_test_database,
)


PROD_URL = "postgresql+asyncpg://angioplus:secret@127.0.0.1:5432/angioplus"
TEST_URL = (
    "postgresql+asyncpg://angioplus_test:secret@127.0.0.1:5432/angioplus_test"
)


# ----------------------------------------------------------------
# The guard must BLOCK the production database.
# ----------------------------------------------------------------

def test_blocks_production_database():
    with pytest.raises(TestDatabaseSafetyError):
        guard_test_database(test_url=PROD_URL, prod_url=PROD_URL)


def test_blocks_production_database_even_with_no_prod_url():
    # The forbidden-name rule applies regardless of DATABASE_URL.
    with pytest.raises(TestDatabaseSafetyError):
        guard_test_database(test_url=PROD_URL, prod_url="")


@pytest.mark.parametrize("name", sorted(FORBIDDEN_DATABASE_NAMES))
def test_blocks_every_forbidden_name(name):
    url = f"postgresql+asyncpg://angioplus:x@127.0.0.1:5432/{name}"
    with pytest.raises(TestDatabaseSafetyError):
        guard_test_database(test_url=url, prod_url="")


# ----------------------------------------------------------------
# The guard must ALLOW a correctly configured test database.
# ----------------------------------------------------------------

def test_allows_angioplus_test():
    resolved = guard_test_database(test_url=TEST_URL, prod_url=PROD_URL)
    assert resolved == TEST_URL


def test_allows_angioplus_test_without_prod_url():
    resolved = guard_test_database(test_url=TEST_URL, prod_url="")
    assert resolved == TEST_URL


# ----------------------------------------------------------------
# Fail-closed for missing / malformed configuration.
# ----------------------------------------------------------------

def test_blocks_missing_test_url():
    with pytest.raises(TestDatabaseSafetyError):
        guard_test_database(test_url="", prod_url=PROD_URL)


def test_blocks_unparseable_test_url():
    with pytest.raises(TestDatabaseSafetyError):
        guard_test_database(test_url="not-a-valid-url", prod_url=PROD_URL)


def test_blocks_non_allowlisted_database_name():
    # A random unrelated DB must not be silently accepted.
    url = "postgresql+asyncpg://u:p@127.0.0.1:5432/some_other_db"
    with pytest.raises(TestDatabaseSafetyError):
        guard_test_database(test_url=url, prod_url=PROD_URL)


# ----------------------------------------------------------------
# Critical collision rule: test URL == production URL (same DB name,
# possibly different host) must be refused.
# ----------------------------------------------------------------

def test_blocks_same_physical_database_as_prod():
    same = "postgresql+asyncpg://other:other@127.0.0.1:5432/angioplus_test"
    prod = "postgresql+asyncpg://angioplus:angioplus@127.0.0.1:5432/angioplus_test"
    # Different user, same host/port/database -> same physical DB.
    with pytest.raises(TestDatabaseSafetyError):
        guard_test_database(test_url=same, prod_url=prod)


def test_blocks_test_url_equal_to_prod_url():
    with pytest.raises(TestDatabaseSafetyError):
        guard_test_database(test_url=PROD_URL, prod_url=PROD_URL)


def test_blocks_by_database_name_collision():
    # Same database name on a different host is still refused (name rule).
    test = "postgresql+asyncpg://u:p@db.example:5432/angioplus_test"
    prod = "postgresql+asyncpg://u:p@127.0.0.1:5432/angioplus_test"
    with pytest.raises(TestDatabaseSafetyError):
        guard_test_database(test_url=test, prod_url=prod)


# ----------------------------------------------------------------
# Env-driven behaviour (reads TEST_DATABASE_URL / DATABASE_URL).
# ----------------------------------------------------------------

def test_reads_from_environment(monkeypatch):
    monkeypatch.setenv("TEST_DATABASE_URL", TEST_URL)
    monkeypatch.setenv("DATABASE_URL", PROD_URL)
    assert guard_test_database() == TEST_URL


def test_fails_closed_when_env_missing(monkeypatch):
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", PROD_URL)
    with pytest.raises(TestDatabaseSafetyError):
        guard_test_database()


def test_allowlist_is_extensible_via_env(monkeypatch):
    extra = "postgresql+asyncpg://u:p@127.0.0.1:5432/angioplus_ci"
    monkeypatch.setenv("TEST_DATABASE_URL", extra)
    monkeypatch.setenv("DATABASE_URL", PROD_URL)
    # Not allowed by default.
    with pytest.raises(TestDatabaseSafetyError):
        guard_test_database()
    # Allowed once explicitly allowlisted.
    monkeypatch.setenv("ALLOWED_TEST_DATABASES", "angioplus_ci")
    assert guard_test_database() == extra
