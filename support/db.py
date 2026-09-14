"""
Async database setup for the support domain.

PostgreSQL + SQLAlchemy 2.x (async) + asyncpg.

``DATABASE_URL`` must be an async DSN, e.g.:

    postgresql+asyncpg://user:pass@localhost:5432/angioplus

This module deliberately contains no business logic: it only wires the
engine, the session factory and the declarative ``Base``.
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


load_dotenv()


class Base(DeclarativeBase):
    """Declarative base for all support ORM models."""


_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError(
            "В .env отсутствует DATABASE_URL (ожидается async DSN, "
            "например postgresql+asyncpg://user:pass@host:5432/db)"
        )
    if not url.startswith("postgresql+asyncpg://"):
        raise RuntimeError(
            "DATABASE_URL должен быть async DSN вида "
            "'postgresql+asyncpg://...'"
        )
    return url


def get_engine() -> AsyncEngine:
    """Lazily create the process-wide async engine."""
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            get_database_url(),
            pool_pre_ping=True,
            future=True,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


async def dispose_engine() -> None:
    """Dispose the engine (used on shutdown and in tests)."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
