"""Async engine + session helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Request
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def _to_async_dsn(dsn: str) -> str:
    """Normalise a Postgres DSN for asyncpg.

    Railway hands out `postgres://...` and `postgresql://...`; SQLAlchemy's
    async layer wants `postgresql+asyncpg://...`. Pass through other schemes
    (sqlite+aiosqlite://) untouched for tests.
    """
    if dsn.startswith("postgresql+asyncpg://"):
        return dsn
    if dsn.startswith("postgresql://"):
        return "postgresql+asyncpg://" + dsn[len("postgresql://") :]
    if dsn.startswith("postgres://"):
        return "postgresql+asyncpg://" + dsn[len("postgres://") :]
    return dsn


def make_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(_to_async_dsn(database_url), pool_pre_ping=True, future=True)


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


@asynccontextmanager
async def session_scope(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    async with factory() as s:
        try:
            yield s
            await s.commit()
        except Exception:
            await s.rollback()
            raise


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: one DB session per request, committed on success.

    Lives in db.py (rather than main.py) so non-main modules — `web_auth.py`'s
    `current_user`, for one — can take `Depends(get_session)` without import
    cycles.
    """
    factory = request.app.state.sessions
    async with session_scope(factory) as s:
        yield s
