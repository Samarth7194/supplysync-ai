"""Test configuration - ensure backend/src is on the import path."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("DATABASE_URL", "sqlite:///./pytest_supplysync.db")


@pytest.fixture(scope="session", autouse=True)
def _prepare_default_sqlalchemy_schema():
    """Create tables for tests that exercise the real FastAPI dependency graph.

    Production and Docker use Alembic migrations. Some API tests import
    ``main.app`` directly against the default local SQLite URL, so the test
    harness creates the schema once instead of weakening runtime persistence
    errors.
    """
    from db.models import Base
    from db.session import engine

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
