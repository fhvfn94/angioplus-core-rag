"""
Pytest fixtures for the support-domain integration tests.

These tests REQUIRE a real PostgreSQL (no SQLite): the design relies on
partial unique indexes, CHECK constraints and SELECT ... FOR UPDATE.

DB-backed tests are marked with ``@pytest.mark.db`` and MUST run against a
dedicated test database, configured via ``TEST_DATABASE_URL`` (async DSN):

    postgresql+asyncpg://angioplus_test:angioplus_test@127.0.0.1:5432/angioplus_test

Safety is fail-closed: the shared :func:`guard_test_database` refuses to
run (raises -> pytest ERROR, BEFORE any ``drop_all``/``create_all``) unless
the target is unambiguously a test database. It rejects a missing
``TEST_DATABASE_URL``, the production database name (``angioplus``), a
target equal to ``DATABASE_URL``, and any name outside the explicit
allowlist. See ``tests/_db_safety.py``.

Destroying the schema before each test is centralized HERE (the single
autouse ``_reset_db_schema`` fixture). Test modules must NOT define their
own destructive ``drop_all``/``create_all`` fixtures: doing so would bypass
this guard. Use the ``engine`` / ``session_factory`` / ``session`` fixtures
instead, or request the ``db`` marker and rely on the shared reset.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from support.db import Base
from support.models import (  # noqa: F401  (register tables)
    Conversation,
    InboundEvent,
    Message,
    ProcessedEvent,
)
from _db_safety import (
    TestDatabaseSafetyError,
    get_test_database_url,
    guard_test_database,
)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Opt-in gate for real Yandex Cloud tests.

    Cloud-marked tests hit the network (Alice / File Search) and must never
    run in an ordinary ``pytest`` invocation. They require the explicit
    ``--run-cloud`` flag.
    """
    parser.addoption(
        "--run-cloud",
        action="store_true",
        default=False,
        help=(
            "Run @pytest.mark.cloud tests against real Yandex Cloud. "
            "Off by default; cloud tests are skipped without it."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "db: DB-backed test that runs destructive schema setup (fail-closed "
        "safety guard applies).",
    )
    config.addinivalue_line(
        "markers",
        "cloud: real Yandex Cloud call (Alice / File Search). Skipped "
        "unless --run-cloud is passed; never runs in ordinary pytest.",
    )


@pytest.fixture(autouse=True)
def _cloud_opt_in_guard(request: pytest.FixtureRequest) -> None:
    """Skip every ``@pytest.mark.cloud`` test unless ``--run-cloud`` is set.

    Mirrors the fail-closed DB guard: destructive/networked tests are
    isolated from the default run. ``pytest`` therefore NEVER executes a
    cloud test; manual execution is::

        pytest tests/test_alice_support_semantics_cloud.py --run-cloud -q
    """
    if request.node.get_closest_marker("cloud") is None:
        return
    if not request.config.getoption("--run-cloud"):
        pytest.skip(
            "cloud test skipped: pass --run-cloud to run against real "
            "Yandex Cloud."
        )


@pytest.fixture(autouse=True)
def _db_safety_guard(request: pytest.FixtureRequest) -> None:
    """Fail-closed gate for every ``@pytest.mark.db`` test.

    Runs before any DB fixture and raises (-> pytest ERROR) if the test
    database configuration is unsafe, so destructive operations can never
    reach a production database.
    """
    if request.node.get_closest_marker("db") is not None:
        guard_test_database()


@pytest_asyncio.fixture(autouse=True)
async def _reset_db_schema(request: pytest.FixtureRequest):
    """The single, centralized destructive schema setup.

    Applies only to ``@pytest.mark.db`` tests. Guarded, then
    ``drop_all`` + ``create_all`` on the validated test database. Any other
    test is a no-op.
    """
    if request.node.get_closest_marker("db") is None:
        return
    url = guard_test_database()
    eng = create_async_engine(url, poolclass=None)
    try:
        async with eng.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
    finally:
        await eng.dispose()


@pytest_asyncio.fixture
async def engine(request: pytest.FixtureRequest):
    """Yield an engine bound to the (already reset) test database.

    Requires the ``db`` marker: a test that requests this fixture without
    the marker would silently skip the centralized reset + guard, so we
    refuse instead of allowing a safety bypass.
    """
    if request.node.get_closest_marker("db") is None:
        raise TestDatabaseSafetyError(
            "A test requested the DB 'engine' fixture without the "
            "@pytest.mark.db marker. Refusing to run so the centralized "
            "database safety guard cannot be bypassed."
        )
    eng = create_async_engine(get_test_database_url(), poolclass=None)
    try:
        yield eng
    finally:
        await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@pytest_asyncio.fixture
async def session(session_factory):
    async with session_factory() as sess:
        yield sess
