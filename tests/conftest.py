"""
Pytest fixtures for the support-domain integration tests.

These tests REQUIRE a real PostgreSQL (no SQLite): the design relies on
partial unique indexes, CHECK constraints and SELECT ... FOR UPDATE.

Point the suite at a scratch database via the environment variable
``TEST_DATABASE_URL`` (async DSN), e.g.:

    postgresql+asyncpg://angioplus:angioplus@127.0.0.1:5432/angioplus_test

If it is not set, the DB-backed tests are skipped (the pure FSM tests
still run).
"""

from __future__ import annotations

import os

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


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "").strip()


def _db_required() -> None:
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL is not set; skipping DB-backed test")


@pytest_asyncio.fixture
async def engine():
    _db_required()
    eng = create_async_engine(TEST_DATABASE_URL, poolclass=None)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
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
