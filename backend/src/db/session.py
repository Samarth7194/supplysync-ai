"""SQLAlchemy engine and session construction.

No route should call SQLAlchemy directly. Runtime code should depend on
repositories that receive a Session from this module.
"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import load_settings


def build_engine(database_url: str | None = None) -> Engine:
    """Create a SQLAlchemy engine from an explicit URL or runtime settings."""
    url = database_url or load_settings().database.database_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(
        url,
        connect_args=connect_args,
        pool_pre_ping=True,
        future=True,
    )


engine = build_engine()
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    future=True,
)


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency-style session scope.

    This is intentionally not imported by main.py yet. The next migration step
    can inject repositories through services while keeping route handlers clean.
    """
    with SessionLocal() as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
