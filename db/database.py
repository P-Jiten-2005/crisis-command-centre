"""SQLAlchemy engine and session management.

The database URL is read from ``DATABASE_URL`` (see ``.env.example``).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative base class shared by all ORM models."""


class Database:
    """Owns the SQLAlchemy engine and session factory for one database URL."""

    def __init__(self, url: str, echo: bool = False) -> None:
        """Create the engine and session factory.

        Args:
            url: SQLAlchemy database URL, e.g. ``postgresql+psycopg2://...``.
            echo: Whether to log every SQL statement.
        """
        self.url = url
        self.engine: Engine = self._create_engine(url, echo)
        self.session_factory: sessionmaker[Session] = sessionmaker(
            bind=self.engine, autoflush=False, expire_on_commit=False
        )

    @staticmethod
    def _create_engine(url: str, echo: bool) -> Engine:
        """Build an engine with driver-appropriate pooling options.

        Args:
            url: SQLAlchemy database URL.
            echo: Whether to log SQL statements.

        Returns:
            A configured :class:`Engine`.
        """
        if url.startswith("sqlite"):
            return create_engine(url, echo=echo, connect_args={"check_same_thread": False})
        return create_engine(url, echo=echo, pool_pre_ping=True, pool_size=5, max_overflow=10)

    def create_all(self, retries: int = 10, delay_seconds: float = 2.0) -> None:
        """Create all tables, retrying while the database starts up.

        Args:
            retries: Maximum number of connection attempts.
            delay_seconds: Pause between attempts.

        Raises:
            OperationalError: If the database is still unreachable after all retries.
        """
        import db.models  # noqa: F401  (register models on Base.metadata)

        for attempt in range(1, retries + 1):
            try:
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                Base.metadata.create_all(self.engine)
                logger.info("Database schema ready")
                return
            except OperationalError:
                if attempt == retries:
                    raise
                logger.warning("Database not ready (attempt %d/%d), retrying in %.1fs", attempt, retries, delay_seconds)
                time.sleep(delay_seconds)

    @contextmanager
    def session_scope(self) -> Iterator[Session]:
        """Provide a transactional session that commits on success and rolls back on error.

        Yields:
            An open :class:`Session`.
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


@lru_cache(maxsize=1)
def get_database() -> Database:
    """Return the process-wide :class:`Database` built from ``DATABASE_URL``.

    Returns:
        The singleton database wrapper.
    """
    return Database(get_settings().require_database_url())


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session.

    Yields:
        An open :class:`Session` closed after the request.
    """
    session = get_database().session_factory()
    try:
        yield session
    finally:
        session.close()
